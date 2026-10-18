"""Explicit clock samples and local zoneinfo data only; no network or clock reads."""

from datetime import datetime, timedelta, timezone, tzinfo
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from zentomic.business_hours import resolve_business_hours_target_at


class BusinessHoursTimeTests(unittest.TestCase):
    def resolve(self, instant, hours=None, **kwargs):
        config = dict(timezone_name="America/New_York",
                      open_target="team", closed_target="voicemail")
        config.update(kwargs)
        return resolve_business_hours_target_at(
            {0: [(540, 1020)]} if hours is None else hours,
            instant=instant, **config,
        )

    def test_winter_and_summer_use_workspace_offset_not_server_time(self):
        for month, hour in ((1, 14), (7, 13)):
            # Both dates are Mondays; 9 AM local is inclusive in either season.
            day = 5 if month == 1 else 6
            with self.subTest(month=month):
                start = datetime(2026, month, day, hour, tzinfo=timezone.utc)
                self.assertEqual(self.resolve(start - timedelta(seconds=1)), "voicemail")
                self.assertEqual(self.resolve(start), "team")
                self.assertEqual(self.resolve(start + timedelta(hours=8)), "voicemail")

    def test_offset_input_and_local_week_year_rollover(self):
        instant = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
        offset = instant.astimezone(timezone(timedelta(hours=5, minutes=30)))
        hours = {2: [(1200, 1201)]}  # Wednesday Dec 31, 2025, 8 PM in New York.
        self.assertEqual(self.resolve(instant, hours), "team")
        self.assertEqual(self.resolve(offset, hours), "team")
        sunday = datetime(2026, 7, 6, 3, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(self.resolve(sunday, {6: [(1439, 1440)]}), "team")
        self.assertEqual(self.resolve(sunday + timedelta(seconds=1),
                                      {6: [(1439, 1440)]}), "voicemail")

    def test_dst_fallback_repeated_minutes_have_same_policy(self):
        hours = {6: [(90, 91)]}
        for hour in (5, 6):  # Both UTC instants map to Sunday 1:30 AM.
            with self.subTest(hour=hour):
                instant = datetime(2026, 11, 1, hour, 30, tzinfo=timezone.utc)
                self.assertEqual(self.resolve(instant, hours), "team")
                local = instant.astimezone(ZoneInfo("America/New_York"))
                self.assertEqual(self.resolve(local, hours), "team")
                self.assertEqual(self.resolve(instant + timedelta(minutes=1), hours),
                                 "voicemail")

    def test_dst_spring_forward_skips_nonexistent_minutes(self):
        before = datetime(2026, 3, 8, 6, 59, 59, tzinfo=timezone.utc)
        after = before + timedelta(seconds=1)
        self.assertEqual(self.resolve(before, {6: [(60, 120)]}), "team")
        self.assertEqual(self.resolve(after, {6: [(120, 180)]}), "voicemail")
        self.assertEqual(self.resolve(after, {6: [(180, 181)]}), "team")
        # Even an imaginary same-zone input is interpreted by its UTC instant,
        # not the astimezone(same_zone) identity shortcut.
        imaginary = datetime(2026, 3, 8, 2, 30, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(self.resolve(imaginary, {6: [(120, 180)]}), "voicemail")

    def test_naive_and_invalid_instants_cannot_use_host_timezone(self):
        class NoOffset(tzinfo):
            def utcoffset(self, value):
                return None

        for instant in (None, True, "private-instant", 0, datetime(2026, 1, 5),
                        datetime(2026, 1, 5, tzinfo=NoOffset())):
            with self.subTest(instant=repr(instant)), self.assertRaises(ValueError):
                self.resolve(instant)

    def test_invalid_and_missing_zones_fail_without_echo_or_utc_fallback(self):
        instant = datetime(2026, 1, 5, 14, tzinfo=timezone.utc)
        for zone in (None, True, [], "", " ", "x" * 257, "\ud800",
                     "/private-zone", "../private-zone", "Private/Missing-Zone"):
            with self.subTest(zone=repr(zone)), self.assertRaises(ValueError) as caught:
                self.resolve(instant, timezone_name=zone)
            self.assertNotIn("private", str(caught.exception).lower())
        with patch("zentomic.business_hours.ZoneInfo",
                   side_effect=ZoneInfoNotFoundError("private-zone")):
            with self.assertRaises(ValueError) as caught:
                self.resolve(instant)
            self.assertNotIn("private-zone", str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
            self.assertTrue(caught.exception.__suppress_context__)

    def test_datetime_boundary_overflow_fails_closed(self):
        for instant, zone in ((datetime.min.replace(tzinfo=timezone.utc),
                               "America/New_York"),
                              (datetime.max.replace(tzinfo=timezone.utc),
                               "Asia/Kolkata")):
            with self.subTest(zone=zone), self.assertRaises(ValueError):
                self.resolve(instant, timezone_name=zone)

    def test_complete_policy_validation_and_exact_targets_are_preserved(self):
        instant = datetime(2026, 1, 5, 14, tzinfo=timezone.utc)
        self.assertEqual(self.resolve(instant, open_target=" équipe "), " équipe ")
        self.assertEqual(self.resolve(instant, {}, closed_target=" 音声 "), " 音声 ")
        for hours in ({0: [(540, 1020)], 6: [(2, 1)]}, {0: [(1, 1)]}):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                self.resolve(instant, hours)
        with self.assertRaises(ValueError):
            self.resolve(instant, closed_target="")


if __name__ == "__main__":
    unittest.main()
