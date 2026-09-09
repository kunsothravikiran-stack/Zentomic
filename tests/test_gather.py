"""Synthetic, offline tests for bounded IVR input collection."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from zentomic.gather import GatherDecision, resolve_gather


class GatherTests(unittest.TestCase):
    def decide(self, digits=None, **kwargs):
        config = {
            "routes": {"0": "reception", "1": "sales"},
            "fallback_target": "reception",
            "attempts": 0,
        }
        config.update(kwargs)
        return resolve_gather(digits, **config)

    def test_configured_digit_routes_immediately(self):
        self.assertEqual(self.decide("1"), GatherDecision("route", "sales", 1))

    def test_explicit_reception_selection_is_not_a_retry(self):
        self.assertEqual(self.decide("0"), GatherDecision("route", "reception", 1))

    def test_invalid_input_consumes_one_attempt_and_retries(self):
        for digits in (None, "", "9", "11", " 1", "１", "*", "#", 1, True, [], {}):
            with self.subTest(digits=digits):
                self.assertEqual(self.decide(digits), GatherDecision("retry", None, 1))

    def test_repeated_no_input_exhausts_default_budget(self):
        attempts = 0
        for action in ("retry", "retry", "fallback"):
            decision = self.decide(attempts=attempts)
            self.assertEqual(decision.action, action)
            self.assertEqual(decision.attempts, attempts + 1)
            attempts = decision.attempts
        self.assertEqual(decision.target, "reception")

    def test_valid_input_on_final_attempt_routes(self):
        self.assertEqual(self.decide("1", attempts=2), GatherDecision("route", "sales", 3))

    def test_exhausted_budget_never_routes_or_increments(self):
        for attempts in (3, 4, 100):
            for digits in (None, "1"):
                with self.subTest(attempts=attempts, digits=digits):
                    self.assertEqual(
                        self.decide(digits, attempts=attempts),
                        GatherDecision("fallback", "reception", attempts),
                    )

    def test_one_attempt_budget_never_retries(self):
        self.assertEqual(self.decide(max_attempts=1), GatherDecision("fallback", "reception", 1))
        self.assertEqual(self.decide("1", max_attempts=1), GatherDecision("route", "sales", 1))

    def test_custom_budget(self):
        self.assertEqual(self.decide(attempts=3, max_attempts=5), GatherDecision("retry", None, 4))
        self.assertEqual(self.decide(attempts=4, max_attempts=5), GatherDecision("fallback", "reception", 5))

    def test_invalid_counts_are_rejected_without_coercion(self):
        for field in ("attempts", "max_attempts"):
            for value in (-1, True, False, 1.5, "3", None, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.decide(**{field: value})
        with self.assertRaises(ValueError):
            self.decide(max_attempts=0)

    def test_invalid_configuration_rejected_even_after_exhaustion(self):
        for config in (
            {"routes": None}, {"routes": []}, {"routes": {"11": "sales"}},
            {"routes": {"1": "sales", "2": " "}}, {"fallback_target": ""},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.decide("1", attempts=3, **config)

    def test_empty_menu_obeys_same_budget(self):
        self.assertEqual(self.decide("1", routes={}), GatherDecision("retry", None, 1))
        self.assertEqual(self.decide("1", routes={}, attempts=2), GatherDecision("fallback", "reception", 3))

    def test_menu_is_not_mutated_and_identifiers_are_preserved(self):
        menu = {"1": " team/Sales "}
        self.assertEqual(self.decide("1", routes=menu).target, " team/Sales ")
        self.assertEqual(menu, {"1": " team/Sales "})

    def test_decision_is_immutable(self):
        decision = self.decide()
        with self.assertRaises(FrozenInstanceError):
            decision.attempts = 0

    def test_no_environment_or_network_required(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(self.decide("1").action, "route")


if __name__ == "__main__":
    unittest.main()
