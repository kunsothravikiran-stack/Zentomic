"""Synthetic whole-call deadlines, without real time or service operations."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.budget import CallBudget, resolve_call_budget
from zentomic.twiml import render_hangup


class CallBudgetTests(unittest.TestCase):
    def test_exact_deadline_and_later_are_terminal(self):
        for now in (999, 1000, 1001, 10**20):
            with self.subTest(now=now):
                expected = CallBudget("continue", 1) if now == 999 else CallBudget("hangup", 0)
                self.assertEqual(resolve_call_budget(deadline_ms=1000, now_ms=now), expected)

    def test_zero_and_large_timestamps_use_exact_integer_arithmetic(self):
        self.assertEqual(resolve_call_budget(deadline_ms=0, now_ms=0), CallBudget("hangup", 0))
        self.assertEqual(resolve_call_budget(deadline_ms=1, now_ms=0), CallBudget("continue", 1))
        self.assertEqual(resolve_call_budget(deadline_ms=10**20 + 1, now_ms=10**20),
                         CallBudget("continue", 1))

    def test_step_reserve_boundary_preserves_actual_remaining_time(self):
        for remaining, action in ((4999, "hangup"), (5000, "continue"),
                                  (5001, "continue"), (0, "hangup")):
            with self.subTest(remaining=remaining):
                self.assertEqual(resolve_call_budget(
                    deadline_ms=10_000, now_ms=10_000 - remaining,
                    minimum_remaining_ms=5000,
                ), CallBudget(action, remaining))

    def test_step_reserves_share_the_original_deadline(self):
        deadline = 10**20
        self.assertEqual(resolve_call_budget(
            deadline_ms=deadline, now_ms=deadline - 6000,
            minimum_remaining_ms=5000,
        ), CallBudget("continue", 6000))
        # A later operation cannot restart the deadline to obtain its reserve.
        self.assertEqual(resolve_call_budget(
            deadline_ms=deadline, now_ms=deadline - 4000,
            minimum_remaining_ms=5000,
        ), CallBudget("hangup", 4000))
        self.assertEqual(resolve_call_budget(
            deadline_ms=deadline, now_ms=deadline + 1,
            minimum_remaining_ms=5000,
        ), CallBudget("hangup", 0))

    def test_step_reserve_must_be_positive_integer_even_after_expiry(self):
        for value in (None, True, False, 0, -1, 1.0, float("nan"),
                      float("inf"), "synthetic-private-value", [], {}):
            for now in (0, 1000):
                with self.subTest(value=value, now=now):
                    with self.assertRaises(ValueError) as caught:
                        resolve_call_budget(deadline_ms=1000, now_ms=now,
                                            minimum_remaining_ms=value)
                    self.assertNotIn("synthetic-private-value", str(caught.exception))

    def test_explicit_default_reserve_preserves_existing_behavior(self):
        for deadline, now in ((0, 0), (1, 0), (100, 99), (100, 101)):
            with self.subTest(deadline=deadline, now=now):
                self.assertEqual(
                    resolve_call_budget(deadline_ms=deadline, now_ms=now),
                    resolve_call_budget(deadline_ms=deadline, now_ms=now,
                                        minimum_remaining_ms=1),
                )

    def test_shared_deadline_does_not_restart_between_steps(self):
        # One trusted deadline covers collection, classification, confirmation,
        # and fallback. These labels describe a synthetic adapter's checkpoints.
        deadline = 601_000
        for step, now, remaining in (("collection", 1_000, 600_000),
                                     ("classification", 101_000, 500_000),
                                     ("confirmation", 501_000, 100_000),
                                     ("fallback", 601_000, 0)):
            with self.subTest(step=step):
                result = resolve_call_budget(deadline_ms=deadline, now_ms=now)
                self.assertEqual(result.remaining_ms, remaining)
                self.assertEqual(result.action, "continue" if remaining else "hangup")
        self.assertEqual([node.tag for node in fromstring(render_hangup())], ["Hangup"])

    def test_timestamps_are_strict_even_when_the_other_value_is_zero(self):
        for name in ("deadline_ms", "now_ms"):
            for value in (None, True, False, -1, 0.0, 1.5, float("nan"), float("inf"),
                          "1000", b"1000", [], {}):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    resolve_call_budget(**{"deadline_ms": 0, "now_ms": 0, name: value})

    def test_decisions_are_immutable_and_errors_do_not_reflect_input(self):
        result = resolve_call_budget(deadline_ms=10, now_ms=1)
        with self.assertRaises(FrozenInstanceError):
            result.remaining_ms = 100
        with self.assertRaises(ValueError) as caught:
            resolve_call_budget(deadline_ms="synthetic-private-value", now_ms=0)
        self.assertNotIn("synthetic-private-value", str(caught.exception))

    def test_clock_environment_and_network_are_not_read(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "time.time", side_effect=AssertionError("Clock must be injected")
        ), patch("socket.socket", side_effect=AssertionError("Network forbidden")), patch(
            "builtins.print"
        ) as output:
            self.assertEqual(resolve_call_budget(deadline_ms=10, now_ms=1), CallBudget("continue", 9))
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
