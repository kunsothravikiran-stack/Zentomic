"""Synthetic weekly-hours routing, without clocks, timezone data, or I/O."""

import copy
import unittest
from collections.abc import Mapping
from unittest.mock import patch

from zentomic.business_hours import resolve_business_hours_target


class BusinessHoursTests(unittest.TestCase):
    def resolve(self, hours, **kwargs):
        config = dict(weekday=0, minute_of_day=600,
                      open_target="team", closed_target="voicemail")
        config.update(kwargs)
        return resolve_business_hours_target(hours, **config)

    def test_half_open_boundaries_and_split_shifts(self):
        hours = {0: [(540, 720), (780, 1020)]}
        for minute, expected in ((539, "voicemail"), (540, "team"),
                                 (719, "team"), (720, "voicemail"),
                                 (779, "voicemail"), (780, "team"),
                                 (1019, "team"), (1020, "voicemail")):
            with self.subTest(minute=minute):
                self.assertEqual(self.resolve(hours, minute_of_day=minute), expected)

    def test_missing_days_empty_schedule_and_full_days(self):
        self.assertEqual(self.resolve({}), "voicemail")
        self.assertEqual(self.resolve({0: []}), "voicemail")
        self.assertEqual(self.resolve({1: [(0, 1440)]}), "voicemail")
        for day in range(7):
            for minute in (0, 1439):
                self.assertEqual(self.resolve({day: ((0, 1440),)},
                                             weekday=day, minute_of_day=minute), "team")

    def test_overnight_windows_split_across_week_boundary(self):
        hours = {6: [(1320, 1440)], 0: [(0, 120)]}
        for day, minute, expected in ((6, 1319, "voicemail"), (6, 1320, "team"),
                                      (6, 1439, "team"), (0, 0, "team"),
                                      (0, 119, "team"), (0, 120, "voicemail")):
            with self.subTest(day=day, minute=minute):
                self.assertEqual(self.resolve(hours, weekday=day,
                                             minute_of_day=minute), expected)

    def test_union_preserves_input_and_exact_targets(self):
        hours = {0: [[900, 1080], [540, 960]]}
        before = copy.deepcopy(hours)
        self.assertEqual(self.resolve(hours, open_target=" équipe "), " équipe ")
        self.assertEqual(self.resolve(hours, minute_of_day=1100,
                                      closed_target=" 音声 "), " 音声 ")
        self.assertEqual(hours, before)
        self.assertEqual(self.resolve({0: [(i, i + 1) for i in range(24)]},
                                      minute_of_day=23), "team")

    def test_local_time_coordinates_are_strict(self):
        for day in (None, True, False, -1, 7, 1.0, "0", []):
            with self.subTest(day=day), self.assertRaises(ValueError):
                self.resolve({}, weekday=day)
        for minute in (None, True, False, -1, 1440, 600.0, "600", []):
            with self.subTest(minute=minute), self.assertRaises(ValueError):
                self.resolve({}, minute_of_day=minute)

    def test_invalid_schedule_shapes_and_windows_are_rejected(self):
        invalid = (None, [], (), "private-hours", {True: []}, {-1: []}, {7: []},
                   {"0": []}, {0: None}, {0: {}}, {0: iter(())},
                   {0: [()]}, {0: [(0,)]}, {0: [(0, 1, 2)]}, {0: ["0,1"]},
                   {0: [(True, 600)]}, {0: [(0, False)]}, {0: [(0.0, 600)]},
                   {0: [(0, "600")]}, {0: [(-1, 600)]}, {0: [(600, 600)]},
                   {0: [(600, 540)]}, {0: [(1320, 120)]}, {0: [(0, 1441)]},
                   {0: [(0, 1)] * 25})
        for hours in invalid:
            with self.subTest(hours=repr(hours)), self.assertRaises(ValueError):
                self.resolve(hours)

    def test_inactive_days_are_validated_even_after_an_open_match(self):
        for hours in ({0: [(0, 1440)], 6: [(1320, 120)]},
                      {0: [(0, 1440), (600, 600)]}):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                self.resolve(hours)

    def test_both_targets_are_validated_even_when_unused(self):
        for value in (None, True, [], "", " ", "x" * 257, "é" * 129, "\ud800"):
            for field in ("open_target", "closed_target"):
                with self.subTest(field=field, value=repr(value)), self.assertRaises(ValueError):
                    self.resolve({0: [(0, 1440)]}, **{field: value})

    def test_snapshot_iteration_is_bounded_at_every_level(self):
        class EndlessDays(Mapping):
            def __len__(self):
                return 1

            def __getitem__(self, key):
                raise AssertionError("oversized mapping must not read values")

            def __iter__(self):
                for number in range(9):
                    if number == 8:
                        raise AssertionError("too many days read")
                    yield number

        class EndlessList(list):
            def __init__(self, value, limit):
                self.value, self.limit, self.consumed = value, limit, 0

            def __iter__(self):
                while True:
                    self.consumed += 1
                    if self.consumed > self.limit:
                        raise AssertionError("too many list entries read")
                    yield self.value

        windows = EndlessList((0, 1), 25)
        endpoints = EndlessList(1, 3)
        for hours in (EndlessDays(), {0: windows}, {0: [endpoints]}):
            with self.subTest(hours=type(hours)), self.assertRaises(ValueError):
                self.resolve(hours)
        self.assertEqual(windows.consumed, 25)
        self.assertEqual(endpoints.consumed, 3)

    def test_errors_and_success_are_silent_and_do_not_echo_configuration(self):
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve({0: [(0, 1440)]}), "team")
            with self.assertRaises(ValueError) as caught:
                self.resolve({0: [("private-hours", 600)]},
                             open_target="private-open", closed_target="private-closed")
            for private in ("private-hours", "private-open", "private-closed"):
                self.assertNotIn(private, str(caught.exception))
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
