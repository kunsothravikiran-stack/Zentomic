"""Offline admission of voice transitions, including rapid redirect loops."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.budget import resolve_call_budget, resolve_transition_budget
from zentomic.twiml import render_hangup, render_redirect


class TransitionBudgetTests(unittest.TestCase):
    def test_each_admission_consumes_one_transition_including_the_last(self):
        for count, action, expected in (
            (0, "continue", 1), (1, "continue", 2), (2, "continue", 3),
            (3, "hangup", 3), (4, "hangup", 4),
        ):
            with self.subTest(count=count):
                result = resolve_transition_budget(transitions=count, max_transitions=3)
                self.assertEqual((result.action, result.transitions), (action, expected))

    def test_zero_limit_disables_all_transitions(self):
        result = resolve_transition_budget(transitions=0, max_transitions=0)
        self.assertEqual((result.action, result.transitions), ("hangup", 0))

    def test_configuration_is_strict_and_errors_do_not_reflect_values(self):
        for name in ("transitions", "max_transitions"):
            for value in (None, True, False, -1, 1.0, float("nan"), float("inf"),
                          "synthetic-private-value", [], {}):
                kwargs = dict(transitions=0, max_transitions=0)
                kwargs[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError) as caught:
                        resolve_transition_budget(**kwargs)
                    self.assertEqual(str(caught.exception),
                                     f"{name} must be a nonnegative integer")

    def test_rapid_loop_stops_even_before_whole_call_deadline(self):
        count = 0
        rendered = []
        # Synthetic trusted state with an unchanged clock: a time budget alone
        # would keep admitting this alternating redirect loop.
        for path in ("/voice/menu", "/voice/confirm") * 3:
            self.assertEqual(resolve_call_budget(deadline_ms=1000, now_ms=1).action,
                             "continue")
            decision = resolve_transition_budget(transitions=count, max_transitions=3)
            count = decision.transitions
            xml = (render_redirect(action_path=path) if decision.action == "continue"
                   else render_hangup())
            rendered.append(fromstring(xml)[0].tag)
        self.assertEqual(rendered, ["Redirect"] * 3 + ["Hangup"] * 3)
        self.assertEqual(count, 3)

    def test_result_is_immutable_and_does_not_read_runtime_state(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("time.time", side_effect=AssertionError("Clock forbidden")):
            first = resolve_transition_budget(transitions=0, max_transitions=2)
            self.assertEqual(first, resolve_transition_budget(transitions=0, max_transitions=2))
        with self.assertRaises(FrozenInstanceError):
            first.transitions = 0

    def test_large_integer_counts_remain_exact(self):
        count = 10**30
        result = resolve_transition_budget(transitions=count, max_transitions=count + 1)
        self.assertEqual(result.transitions, count + 1)
        self.assertEqual(result.action, "continue")


if __name__ == "__main__":
    unittest.main()
