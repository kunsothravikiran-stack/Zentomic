"""Combined time/count admission must not spend transitions on denied work."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from zentomic.budget import resolve_voice_step_budget


class VoiceStepBudgetTests(unittest.TestCase):
    def test_both_budgets_must_admit_and_only_admission_increments(self):
        for now, count, action, remaining, next_count in (
            (0, 0, "continue", 100, 1),
            (90, 1, "continue", 10, 2),  # Both exact admission boundaries.
            (91, 0, "hangup", 9, 0),
            (100, 0, "hangup", 0, 0),
            (101, 0, "hangup", 0, 0),
            (0, 2, "hangup", 100, 2),
            (0, 3, "hangup", 100, 3),
            (100, 2, "hangup", 0, 2),
        ):
            with self.subTest(now=now, count=count):
                result = resolve_voice_step_budget(
                    deadline_ms=100, now_ms=now, minimum_remaining_ms=10,
                    transitions=count, max_transitions=2,
                )
                self.assertEqual(
                    (result.action, result.remaining_ms, result.transitions),
                    (action, remaining, next_count),
                )

    def test_default_threshold_and_zero_transition_limit(self):
        result = resolve_voice_step_budget(
            deadline_ms=1, now_ms=0, transitions=0, max_transitions=1,
        )
        self.assertEqual((result.action, result.transitions), ("continue", 1))
        result = resolve_voice_step_budget(
            deadline_ms=1, now_ms=0, transitions=0, max_transitions=0,
        )
        self.assertEqual((result.action, result.transitions), ("hangup", 0))

    def test_invalid_configuration_is_rejected_even_when_other_budget_denies(self):
        for name in ("deadline_ms", "now_ms", "minimum_remaining_ms",
                     "transitions", "max_transitions"):
            for value in (None, True, False, -1, 1.0, float("nan"),
                          float("inf"), "private-input", [], {}):
                with self.subTest(name=name, value=value):
                    args = dict(deadline_ms=0, now_ms=0, minimum_remaining_ms=1,
                                transitions=0, max_transitions=0)
                    args[name] = value
                    with self.assertRaises(ValueError) as caught:
                        resolve_voice_step_budget(**args)
                    self.assertNotIn("private-input", str(caught.exception))
        with self.assertRaises(ValueError):
            resolve_voice_step_budget(deadline_ms=0, now_ms=0,
                                      minimum_remaining_ms=0,
                                      transitions=0, max_transitions=0)

    def test_sequence_keeps_count_on_time_denial_and_stops_at_count_limit(self):
        count = 0
        outcomes = []
        for now in (0, 95, 96):
            result = resolve_voice_step_budget(
                deadline_ms=100, now_ms=now, minimum_remaining_ms=10,
                transitions=count, max_transitions=4,
            )
            count = result.transitions
            outcomes.append(result.action)
        self.assertEqual(outcomes, ["continue", "hangup", "hangup"])
        self.assertEqual(count, 1)

        count = 0
        outcomes = []
        for _ in range(4):
            result = resolve_voice_step_budget(
                deadline_ms=100, now_ms=0, transitions=count, max_transitions=2,
            )
            count = result.transitions
            outcomes.append(result.action)
        self.assertEqual(outcomes, ["continue", "continue", "hangup", "hangup"])
        self.assertEqual(count, 2)

    def test_large_integers_immutable_result_and_no_runtime_dependencies(self):
        big = 10**30
        args = dict(deadline_ms=big + 1, now_ms=big,
                    transitions=big, max_transitions=big + 1)
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("time.time", side_effect=AssertionError("Clock forbidden")):
            result = resolve_voice_step_budget(**args)
            self.assertEqual(result, resolve_voice_step_budget(**args))
        self.assertEqual((result.action, result.remaining_ms, result.transitions),
                         ("continue", 1, big + 1))
        with self.assertRaises(FrozenInstanceError):
            result.transitions = 0


if __name__ == "__main__":
    unittest.main()
