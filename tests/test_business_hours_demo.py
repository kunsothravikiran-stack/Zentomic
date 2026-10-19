"""Runnable business-hours preview uses only fixed synthetic configuration."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from unittest.mock import patch

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_business_hours


class BusinessHoursDemoTests(unittest.TestCase):
    def invoke(self, argv):
        output, errors = io.StringIO(), io.StringIO()
        with offline_guard(), redirect_stdout(output), redirect_stderr(errors):
            try:
                code = main(argv)
            except SystemExit as error:
                code = error.code
        return code, output.getvalue(), errors.getvalue()

    def test_empty_samples_do_not_invent_a_clock_read_or_decision(self):
        with offline_guard():
            self.assertEqual(simulate_business_hours(iter([])), [])

    def test_cli_preserves_order_and_half_open_weekday_boundaries(self):
        argv = []
        for instant in ("2026-01-05T13:59:59Z", "2026-01-05T14:00:00Z",
                        "2026-01-05T21:59:59Z", "2026-01-05T22:00:00Z"):
            argv.extend(["--business-hours-at", instant])
        code, output, errors = self.invoke(argv)
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output), {"mode": "offline-demo", "steps": [
            {"action": "route", "target": "demo-" + target}
            for target in ("voicemail", "team", "team", "voicemail")
        ]})
        self.assertNotIn("2026", output)

    def test_summer_offsets_weekends_and_local_holiday_fixture(self):
        instants = [datetime.fromisoformat(value) for value in (
            "2026-07-06T13:00:00Z", "2026-07-06T18:30:00+05:30",
            "2026-07-05T13:00:00Z", "2026-01-01T15:00:00Z",
            "2026-01-01T01:00:00Z",  # Still Wednesday Dec 31 at 20:00 local.
            "2026-01-02T01:00:00Z",  # Still the closed local Jan 1.
        )]
        with offline_guard():
            steps = simulate_business_hours(iter(instants))
        self.assertEqual([step["target"] for step in steps],
                         ["demo-team"] * 2 + ["demo-voicemail"] * 4)

    def test_holiday_uses_workspace_date_not_input_offset_date(self):
        # Both samples are during local office hours on the previous NY date.
        steps = simulate_business_hours([datetime.fromisoformat(value) for value in (
            "2026-01-01T00:00:00+05:30",  # Dec 31 13:30: open.
            "2026-01-02T00:00:00+05:30",  # Jan 1 13:30: holiday closure.
        )])
        self.assertEqual([step["target"] for step in steps],
                         ["demo-team", "demo-voicemail"])

    def test_invalid_or_naive_sample_prints_no_partial_result_or_value(self):
        for value in ("private-marker", "2026-01-05", "2026-01-05T14:00:00",
                      "0001-01-01T00:00:00Z", "2026-13-01T14:00:00Z"):
            with self.subTest(value=value):
                code, output, errors = self.invoke([
                    "--business-hours-at", "2026-01-05T14:00:00Z",
                    "--business-hours-at", value,
                ])
                self.assertEqual((code, output), (2, ""))
                self.assertIn("invalid synthetic business-hours instant", errors)
                self.assertNotIn(value, errors)

    def test_missing_timezone_data_does_not_fallback_to_open(self):
        with patch("zentomic.business_hours.ZoneInfo", side_effect=OSError("private-path")):
            code, output, errors = self.invoke([
                "--business-hours-at", "2026-01-05T14:00:00Z",
            ])
        self.assertEqual((code, output), (2, ""))
        self.assertIn("unavailable demo timezone", errors)
        self.assertNotIn("private-path", errors)

    def test_mode_rejects_unrelated_inputs_before_simulation(self):
        for extra in (["1"], ["--speech", "hello"], ["--intent", "sales"],
                      ["--classifier-response", "{}"], ["--dial-result", "busy"],
                      ["--call-status", "ringing"], ["--initial-call-status", "queued"]):
            with self.subTest(extra=extra), patch(
                "zentomic.demo.simulate_business_hours",
            ) as simulate:
                code, output, _ = self.invoke([
                    "--business-hours-at", "2026-01-05T14:00:00Z", *extra,
                ])
            self.assertEqual((code, output), (2, ""))
            simulate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
