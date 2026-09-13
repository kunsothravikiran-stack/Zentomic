"""Offline application policy for repeated and out-of-order status observations."""

import unittest
from unittest.mock import patch

from zentomic.call_status import advance_call_status


class CallStatusTransitionTests(unittest.TestCase):
    active = ("queued", "ringing", "in-progress")
    terminal = ("completed", "busy", "failed", "no-answer", "canceled")

    def test_first_observation_accepts_every_known_status(self):
        for incoming in self.active + self.terminal:
            with self.subTest(incoming=incoming):
                self.assertEqual(advance_call_status(None, incoming), incoming)

    def test_active_progress_never_regresses_and_may_skip_states(self):
        for current_index, current in enumerate(self.active):
            for incoming_index, incoming in enumerate(self.active):
                with self.subTest(current=current, incoming=incoming):
                    self.assertEqual(
                        advance_call_status(current, incoming),
                        self.active[max(current_index, incoming_index)],
                    )

    def test_every_active_state_can_end_with_every_terminal_outcome(self):
        for current in self.active:
            for incoming in self.terminal:
                with self.subTest(current=current, incoming=incoming):
                    self.assertEqual(advance_call_status(current, incoming), incoming)

    def test_first_terminal_outcome_survives_delays_duplicates_and_conflicts(self):
        for current in self.terminal:
            for incoming in self.active + self.terminal:
                with self.subTest(current=current, incoming=incoming):
                    self.assertEqual(advance_call_status(current, incoming), current)

    def test_invalid_incoming_status_is_rejected_even_after_terminal_state(self):
        invalid = (None, "", "private-unknown", " completed", "COMPLETED",
                   "initiated", "answered", True, 1, [], {}, b"completed", "\ud800")
        for current in (None,) + self.active + self.terminal:
            for incoming in invalid:
                with self.subTest(current=current, incoming=repr(incoming)):
                    with self.assertRaisesRegex(ValueError, "^unsupported call status$"):
                        advance_call_status(current, incoming)

    def test_invalid_stored_state_is_not_silently_repaired(self):
        for current in ("", "private-unknown", " completed", "x" * 10_000,
                        True, 1, [], {}):
            with self.subTest(current=repr(current)):
                with self.assertRaisesRegex(ValueError, "^unsupported call status$"):
                    advance_call_status(current, "completed")

    def test_offline_event_sequence_retains_terminal_state(self):
        current = None
        history = []
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("builtins.print") as output:
            for incoming in ("ringing", "queued", "in-progress", "completed",
                             "ringing", "completed", "failed"):
                current = advance_call_status(current, incoming)
                history.append(current)
            output.assert_not_called()
        self.assertEqual(history, ["ringing", "ringing", "in-progress"] + ["completed"] * 4)


if __name__ == "__main__":
    unittest.main()
