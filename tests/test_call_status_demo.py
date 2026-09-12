"""Offline lifecycle replay keeps delayed callbacks from regressing state."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_call_status


class CallStatusDemoTests(unittest.TestCase):
    def test_empty_history_does_not_invent_an_observation(self):
        self.assertEqual(simulate_call_status([]), [])

    def test_replay_exposes_observation_and_previous_state(self):
        observations = ["in-progress", "in-progress", "ringing", "completed", "busy"]
        with offline_guard():
            steps = simulate_call_status(iter(observations))
        self.assertEqual([step["incoming_status"] for step in steps], observations)
        self.assertEqual([step["previous_status"] for step in steps],
                         [None, "in-progress", "in-progress", "in-progress", "completed"])
        self.assertEqual([step["status"] for step in steps],
                         ["in-progress", "in-progress", "in-progress", "completed", "completed"])
        self.assertEqual(observations,
                         ["in-progress", "in-progress", "ringing", "completed", "busy"])

    def test_cli_exposes_resumed_transition_without_an_extra_step(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main([
                "--initial-call-status", "completed", "--call-status", "failed",
            ]), 0)
        self.assertEqual(json.loads(output.getvalue())["steps"], [{
            "action": "observe-call-status", "previous_status": "completed",
            "incoming_status": "failed", "status": "completed",
            "changed": False, "terminal": True,
        }])

    def test_initial_state_does_not_invent_an_observation(self):
        for initial in ("ringing", "completed"):
            with self.subTest(initial=initial):
                self.assertEqual(simulate_call_status([], initial_status=initial), [])

    def test_replay_can_resume_from_active_or_terminal_state(self):
        for initial, expected, changes in (
            ("in-progress", ["in-progress", "in-progress", "completed"], [False, False, True]),
            ("busy", ["busy", "busy", "busy"], [False, False, False]),
        ):
            with self.subTest(initial=initial), offline_guard():
                steps = simulate_call_status(
                    iter(["queued", "ringing", "completed"]), initial_status=initial,
                )
                self.assertEqual([step["status"] for step in steps], expected)
                self.assertEqual([step["changed"] for step in steps], changes)

    def test_invalid_initial_state_is_rejected_before_consuming_input(self):
        def unread():
            self.fail("invalid initial state must not consume observations")
            yield "completed"

        for initial in ("", "private-marker", "COMPLETED", True, 1, [], {}, "\ud800"):
            with self.subTest(initial=repr(initial)):
                with self.assertRaisesRegex(ValueError, "^unsupported call status$"):
                    simulate_call_status(unread(), initial_status=initial)

    def test_terminal_initial_state_still_validates_incoming_status(self):
        with self.assertRaisesRegex(ValueError, "^unsupported call status$"):
            simulate_call_status(["private-marker"], initial_status="completed")

    def test_cli_initial_state_matches_helper(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main([
                "--initial-call-status", "in-progress", "--call-status", "ringing",
            ]), 0)
        self.assertEqual(json.loads(output.getvalue())["steps"],
                         simulate_call_status(["ringing"], initial_status="in-progress"))

    def test_replay_preserves_progress_and_first_terminal_outcome(self):
        statuses = ["ringing", "queued", "in-progress", "in-progress",
                    "completed", "ringing", "failed"]
        with offline_guard():
            steps = simulate_call_status(iter(statuses))
        self.assertEqual([step["status"] for step in steps],
                         ["ringing", "ringing", "in-progress", "in-progress",
                          "completed", "completed", "completed"])
        self.assertEqual([step["changed"] for step in steps],
                         [True, False, True, False, True, False, False])
        self.assertEqual([step["terminal"] for step in steps],
                         [False, False, False, False, True, True, True])
        self.assertTrue(all(step["action"] == "observe-call-status" for step in steps))
        self.assertTrue(all("twiml" not in step for step in steps))

    def test_each_terminal_outcome_can_be_the_first_observation(self):
        for status in ("completed", "busy", "failed", "no-answer", "canceled"):
            with self.subTest(status=status):
                self.assertEqual(simulate_call_status([status]), [{
                    "action": "observe-call-status", "status": status,
                    "previous_status": None, "incoming_status": status,
                    "changed": True, "terminal": True,
                }])

    def test_unknown_status_is_rejected_even_after_terminal_state(self):
        for status in (None, [], "", "answered", "COMPLETED", "private-marker"):
            for prefix in ([], ["completed"]):
                with self.subTest(status=status, prefix=prefix):
                    with self.assertRaisesRegex(ValueError, "^unsupported call status$"):
                        simulate_call_status([*prefix, status])

    def test_cli_matches_helper_without_services(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(["--call-status", "completed", "--call-status", "ringing"]), 0)
        self.assertEqual(json.loads(output.getvalue()), {
            "mode": "offline-demo", "steps": simulate_call_status(["completed", "ringing"]),
        })

    def test_cli_rejects_mixed_modes_and_invalid_status_without_partial_json(self):
        cases = (
            ["--initial-call-status", "ringing"],
            ["--initial-call-status", "ringing", "--intent", "sales"],
            ["--initial-call-status", "ringing", "--dial-result", "busy"],
            ["--initial-call-status", "private-marker", "--call-status", "ringing"],
            ["--call-status", "ringing", "1"],
            ["--call-status", "ringing", "--speech", "synthetic"],
            ["--call-status", "ringing", "--intent", "sales"],
            ["--call-status", "ringing", "--classifier-response", "{}"],
            ["--call-status", "ringing", "--dial-result", "busy"],
            ["--call-status", "private-marker"],
            ["--call-status", "completed", "--call-status", "private-marker"],
        )
        for argv in cases:
            output, error = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(error):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("private-marker", error.getvalue())


if __name__ == "__main__":
    unittest.main()
