"""Synthetic local-date exceptions; no holiday service or clock reads."""

import copy
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from zentomic.business_hours import resolve_business_hours_target_at


class BusinessHoursDateTests(unittest.TestCase):
    def resolve(self, overrides, *, instant=None, hours=None, **kwargs):
        config = dict(timezone_name="America/New_York",
                      open_target="team", closed_target="voicemail")
        config.update(kwargs)
        return resolve_business_hours_target_at(
            {0: [(540, 1020)]} if hours is None else hours,
            instant=instant if instant is not None else datetime(
                2026, 1, 5, 15, tzinfo=timezone.utc),
            date_overrides=overrides, **config,
        )

    def test_closed_date_replaces_weekly_open_hours_without_annual_repetition(self):
        overrides = {date(2026, 1, 5): []}
        self.assertEqual(self.resolve(overrides), "voicemail")
        self.assertEqual(self.resolve({date(2025, 1, 5): []}), "team")
        self.assertEqual(self.resolve(None), "team")
        self.assertEqual(self.resolve({}), "team")

    def test_special_hours_replace_not_union_with_weekly_hours(self):
        overrides = {date(2026, 1, 5): [(720, 780), (1200, 1260)]}
        for hour, minute, second, expected in (
                (16, 59, 59, "voicemail"), (17, 0, 0, "team"),
                (17, 59, 59, "team"), (18, 0, 0, "voicemail"),
                (20, 0, 0, "voicemail")):
            with self.subTest(hour=hour, minute=minute, second=second):
                instant = datetime(2026, 1, 5, hour, minute, second, tzinfo=timezone.utc)
                self.assertEqual(self.resolve(overrides, instant=instant), expected)
        # A date can open a weekday that is otherwise entirely closed.
        self.assertEqual(self.resolve({date(2026, 1, 5): [(0, 1440)]}, hours={}), "team")

    def test_override_uses_workspace_date_across_utc_year_boundary(self):
        instant = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
        hours = {2: [(0, 1440)]}
        self.assertEqual(self.resolve({date(2025, 12, 31): []},
                                      instant=instant, hours=hours), "voicemail")
        self.assertEqual(self.resolve({date(2026, 1, 1): []},
                                      instant=instant, hours=hours), "team")
        offset = instant.astimezone(timezone(timedelta(hours=5, minutes=30)))
        self.assertEqual(self.resolve({date(2025, 12, 31): [(1200, 1201)]},
                                      instant=offset, hours={}), "team")

    def test_override_obeys_dst_folds_and_skipped_minutes(self):
        for hour in (5, 6):
            instant = datetime(2026, 11, 1, hour, 30, tzinfo=timezone.utc)
            self.assertEqual(self.resolve({date(2026, 11, 1): [(90, 91)]},
                                          instant=instant, hours={}), "team")
        instant = datetime(2026, 3, 8, 7, tzinfo=timezone.utc)
        self.assertEqual(self.resolve({date(2026, 3, 8): [(120, 180)]},
                                      instant=instant, hours={}), "voicemail")

    def test_overnight_override_must_be_split_across_dates(self):
        overrides = {date(2026, 1, 4): [(1320, 1440)], date(2026, 1, 5): [(0, 120)]}
        for hour, expected in ((4, "team"), (5, "team"), (6, "team"), (7, "voicemail")):
            instant = datetime(2026, 1, 5, hour, tzinfo=timezone.utc)
            self.assertEqual(self.resolve(overrides, instant=instant, hours={}), expected)
        with self.assertRaises(ValueError):
            self.resolve({date(2026, 1, 5): [(1320, 120)]})

    def test_invalid_inactive_dates_and_windows_cannot_hide_behind_match(self):
        class DateSubclass(date):
            pass

        for bad in (True, None, "private-date", datetime(2026, 1, 5),
                    DateSubclass(2026, 1, 6)):
            with self.subTest(key=repr(bad)), self.assertRaises(ValueError):
                self.resolve({date(2026, 1, 5): [], bad: []})
        for windows in (None, {}, iter(()), [(0, 0)], [(0, True)], [(0, 1441)],
                        [(0, 1)] * 25, [(0, 1, 2)]):
            with self.subTest(windows=repr(windows)), self.assertRaises(ValueError):
                self.resolve({date(2026, 1, 5): [], date(2026, 1, 6): windows})
        for overrides in ([], (), "private-dates", True):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(overrides)
        with self.assertRaises(ValueError):
            self.resolve({date(2026, 1, 5): []}, hours={6: [(3, 2)]})
        with self.assertRaises(ValueError):
            self.resolve({date(2026, 1, 5): []}, open_target="")

    def test_date_count_and_mapping_iteration_are_bounded(self):
        first = date(2026, 1, 1)
        overrides = {first + timedelta(days=i): [] for i in range(366)}
        self.assertEqual(self.resolve(overrides), "voicemail")
        overrides[first + timedelta(days=366)] = []
        with self.assertRaises(ValueError):
            self.resolve(overrides)

        class EndlessDates(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                for index in range(368):
                    if index == 367:
                        raise AssertionError("too many dates read")
                    yield first + timedelta(days=index)

            def __getitem__(self, key):
                raise AssertionError("oversized mapping must not read values")

        with self.assertRaises(ValueError):
            self.resolve(EndlessDates())

    def test_snapshots_preserve_configuration_and_exact_targets_without_output(self):
        overrides = {date(2026, 1, 5): [[540, 720], [600, 780]]}
        before = copy.deepcopy(overrides)
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve(overrides, open_target=" équipe "), " équipe ")
            self.assertEqual(self.resolve({date(2026, 1, 5): []},
                                          closed_target=" 音声 "), " 音声 ")
            with self.assertRaises(ValueError) as caught:
                self.resolve({"private-date": []})
            self.assertNotIn("private-date", str(caught.exception))
            output.assert_not_called()
        self.assertEqual(overrides, before)


if __name__ == "__main__":
    unittest.main()
