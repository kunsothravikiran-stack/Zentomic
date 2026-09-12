"""Synthetic forwarding results, without dialing or callback authentication."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from xml.etree.ElementTree import fromstring

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_forwarding


class ForwardingDemoTests(unittest.TestCase):
    def test_no_result_leaves_the_first_attempt_pending(self):
        with offline_guard():
            self.assertEqual(simulate_forwarding([]), [
                {"action": "await-dial-result", "target": "demo-sales"},
            ])

    def test_unsuccessful_first_attempt_allows_one_reception_fallback(self):
        for status in ("busy", "no-answer", "failed"):
            with self.subTest(status=status), offline_guard():
                self.assertEqual(simulate_forwarding([status]), [
                    {"action": "await-dial-result", "target": "demo-sales"},
                    {"action": "fallback", "target": "demo-reception"},
                    {"action": "await-dial-result", "target": "demo-reception"},
                ])

    def test_every_second_result_ends_without_repeating_fallback(self):
        for first in ("busy", "no-answer", "failed"):
            for second in ("busy", "no-answer", "failed", "completed", "canceled"):
                with self.subTest(first=first, second=second), offline_guard():
                    steps = simulate_forwarding([first, second])
                self.assertEqual([step["action"] for step in steps], [
                    "await-dial-result", "fallback", "await-dial-result", "hangup",
                ])
                self.assertEqual([node.tag for node in fromstring(steps[-1]["twiml"])],
                                 ["Hangup"])
                self.assertNotIn("target", steps[-1])

    def test_completed_and_canceled_do_not_fall_back(self):
        for status in ("completed", "canceled"):
            with self.subTest(status=status), offline_guard():
                steps = simulate_forwarding([status])
            self.assertEqual([step["action"] for step in steps],
                             ["await-dial-result", "hangup"])

    def test_stops_reading_after_hangup_or_second_result(self):
        for statuses in (("completed",), ("canceled",), ("busy", "failed")):
            def inputs():
                yield from statuses
                raise AssertionError("consumed input after terminal forwarding result")

            with self.subTest(statuses=statuses), offline_guard():
                self.assertEqual(simulate_forwarding(inputs())[-1]["action"], "hangup")

    def test_invalid_consumed_results_fail_without_reflection(self):
        for status in (None, "", "answered", "ringing", "COMPLETED", " busy", [],
                       "synthetic-private-marker"):
            for prefix in ([], ["busy"]):
                with self.subTest(status=status, prefix=prefix), offline_guard():
                    with self.assertRaisesRegex(ValueError, "^unsupported Number dial result status$"):
                        simulate_forwarding([*prefix, status])

    def test_cli_matches_helper_and_is_offline(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(["--dial-result", "busy", "--dial-result", "completed"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["mode"], "offline-demo")
        self.assertEqual(result["steps"], simulate_forwarding(["busy", "completed"]))

    def test_cli_rejects_mixed_modes_and_invalid_results_without_partial_json(self):
        cases = (
            ["--dial-result", "busy", "1"],
            ["--dial-result", "busy", "--intent", "sales"],
            ["--dial-result", "busy", "--classifier-response", "{}"],
            ["--dial-result", "busy", "--speech", "synthetic"],
            ["--dial-result", "synthetic-private-marker"],
            ["--dial-result", "busy", "--dial-result", "synthetic-private-marker"],
        )
        for argv in cases:
            output, error = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(error):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("synthetic-private-marker", error.getvalue())


if __name__ == "__main__":
    unittest.main()
