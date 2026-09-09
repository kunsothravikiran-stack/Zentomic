"""Offline confirmed-intent routing tests using synthetic targets only."""

import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import MappingProxyType
from unittest.mock import patch

from zentomic.intent import resolve_intent


class IntentRoutingTests(unittest.TestCase):
    def setUp(self):
        self.routes = {"sales": "team/Sales", "technical_support": "team/Support"}

    def resolve(self, intent, *, routes=None, confirmed=True):
        return resolve_intent(
            intent, self.routes if routes is None else routes,
            fallback_target="team/Reception", confirmed=confirmed,
        )

    def test_routes_each_confirmed_intent(self):
        for label, target in self.routes.items():
            with self.subTest(label=label):
                self.assertEqual(self.resolve(label), target)

    def test_confirmation_defaults_to_fallback(self):
        self.assertEqual(
            resolve_intent("sales", self.routes, fallback_target="team/Reception"),
            "team/Reception",
        )

    def test_only_boolean_true_confirms(self):
        for value in (False, None, 0, 1, "true", "false", "yes", [], [True], {}):
            with self.subTest(confirmed=value):
                self.assertEqual(self.resolve("sales", confirmed=value), "team/Reception")

    def test_missing_malformed_and_unknown_intents_fall_back(self):
        for intent in (None, "", " \t", "unknown", 1, True, [], {}, ["sales"]):
            with self.subTest(intent=intent):
                self.assertEqual(self.resolve(intent), "team/Reception")

    def test_matching_is_exact_without_normalization(self):
        for intent in ("Sales", " sales", "sales ", "sales\n", "ｓａｌｅｓ"):
            with self.subTest(intent=intent):
                self.assertEqual(self.resolve(intent), "team/Reception")

    def test_intent_cannot_supply_an_arbitrary_target(self):
        self.assertEqual(self.resolve("team/Sales"), "team/Reception")

    def test_empty_menu_falls_back(self):
        self.assertEqual(self.resolve("sales", routes={}), "team/Reception")

    def test_invalid_mappings_are_rejected(self):
        for routes in (None, [], "sales", 1):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                resolve_intent(None, routes, fallback_target="team/Reception")

    def test_all_labels_validated_even_without_confirmation(self):
        for label in (None, 1, True, "", " \t\n"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.resolve(None, routes={label: "team/Sales"}, confirmed=False)

    def test_unused_targets_are_validated_before_routing(self):
        for target in (None, "", " \t\n", 1, [], {}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.resolve("sales", routes={"sales": "team/Sales", "other": target})

    def test_fallback_is_validated_even_when_intent_matches(self):
        for target in (None, "", " \t\n", 1, [], {}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                resolve_intent("sales", self.routes, fallback_target=target, confirmed=True)

    def test_readonly_mapping_is_not_mutated(self):
        original = copy.deepcopy(self.routes)
        self.assertEqual(
            self.resolve("sales", routes=MappingProxyType(self.routes)), "team/Sales",
        )
        self.assertEqual(self.routes, original)

    def test_requires_no_network_environment_or_logging(self):
        output = StringIO()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ), redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(self.resolve("sales"), "team/Sales")
            self.assertEqual(self.resolve("unknown"), "team/Reception")
        self.assertEqual(output.getvalue(), "")

    def test_configuration_errors_do_not_echo_configuration(self):
        with self.assertRaises(ValueError) as error:
            self.resolve(None, routes={None: "synthetic-private-marker"})
        self.assertNotIn("synthetic-private-marker", str(error.exception))


if __name__ == "__main__":
    unittest.main()
