"""Offline IVR routing tests with synthetic, opaque target identifiers."""

import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import MappingProxyType
from unittest.mock import patch

from zentomic.routing import resolve_dtmf


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.routes = {"0": "reception", "1": "sales", "2": "support"}

    def resolve(self, digits, routes=None):
        return resolve_dtmf(
            digits, self.routes if routes is None else routes,
            fallback_target="reception",
        )

    def test_routes_each_configured_digit(self):
        for digit, target in self.routes.items():
            with self.subTest(digit=digit):
                self.assertEqual(self.resolve(digit), target)

    def test_supports_all_ten_digits(self):
        routes = {str(i): f"team-{i}" for i in range(10)}
        for digit, target in routes.items():
            with self.subTest(digit=digit):
                self.assertEqual(self.resolve(digit, routes), target)

    def test_unmapped_digit_and_empty_menu_use_fallback(self):
        self.assertEqual(self.resolve("9"), "reception")
        self.assertEqual(self.resolve("1", {}), "reception")

    def test_missing_or_malformed_input_uses_fallback(self):
        for digits in (None, "", "12", " 1", "1 ", "1\n", "*", "#", "x", "１", "١", 1, True, [], {}):
            with self.subTest(digits=digits):
                self.assertEqual(self.resolve(digits), "reception")

    def test_nonmapping_configuration_is_rejected(self):
        for routes in (None, [], "invalid", 1):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                resolve_dtmf("1", routes, fallback_target="reception")

    def test_invalid_keys_are_rejected_even_for_missing_input(self):
        for digit in (None, 1, True, "", "12", " 1", "*", "#", "１", "١"):
            with self.subTest(digit=digit), self.assertRaises(ValueError):
                self.resolve(None, {digit: "sales"})

    def test_invalid_targets_are_rejected_even_on_unused_routes(self):
        for target in (None, "", " \t\n", 1, [], {}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.resolve("0", {"0": "reception", "1": target})

    def test_fallback_must_be_nonblank_string(self):
        for target in (None, "", " \t\n", 1, [], {}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                resolve_dtmf("1", self.routes, fallback_target=target)

    def test_accepts_readonly_mapping_without_mutating_it(self):
        original = copy.deepcopy(self.routes)
        self.assertEqual(self.resolve("1", MappingProxyType(self.routes)), "sales")
        self.assertEqual(self.routes, original)

    def test_target_identifiers_are_preserved(self):
        self.assertEqual(self.resolve("1", {"1": "team/Sales"}), "team/Sales")
        self.assertEqual(resolve_dtmf(None, {}, fallback_target="team/Reception"), "team/Reception")

    def test_requires_no_network_environment_or_logging(self):
        output = StringIO()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ), redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(self.resolve("2"), "support")
        self.assertEqual(output.getvalue(), "")

    def test_configuration_errors_do_not_expose_target_contents(self):
        with self.assertRaises(ValueError) as error:
            self.resolve("1", {"invalid": "synthetic-private-marker"})
        self.assertNotIn("synthetic-private-marker", str(error.exception))


if __name__ == "__main__":
    unittest.main()
