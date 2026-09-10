"""Offline confirmation-step policy tests using only synthetic routing IDs."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.intent import resolve_intent_confirmation
from zentomic.twiml import render_dtmf_gather, render_hangup


class IntentConfirmationTests(unittest.TestCase):
    def resolve(self, digits, **overrides):
        options = dict(intent="support", routes={"support": "team/Support"},
                       fallback_target="team/Reception", attempts=0)
        options.update(overrides)
        return resolve_intent_confirmation(digits, **options)

    def test_confirmation_routes_and_rejection_falls_back(self):
        for digit, action, target in (("1", "route", "team/Support"),
                                      ("2", "fallback", "team/Reception")):
            with self.subTest(digit=digit):
                result = self.resolve(digit)
                self.assertEqual((result.action, result.target, result.attempts),
                                 (action, target, 1))

    def test_invalid_input_is_not_coerced_to_confirmation(self):
        for digits in (None, "", "0", "9", "*", "#", "11", " 1", "1 ", "１",
                       "true", True, 1, [], {}):
            with self.subTest(digits=digits):
                result = self.resolve(digits)
                self.assertEqual((result.action, result.target, result.attempts),
                                 ("retry", None, 1))

    def test_last_attempt_can_confirm_or_decline_but_cannot_retry(self):
        for digit, action, target in (("1", "route", "team/Support"),
                                      ("2", "fallback", "team/Reception"),
                                      (None, "fallback", "team/Reception")):
            with self.subTest(digit=digit):
                result = self.resolve(digit, attempts=2)
                self.assertEqual((result.action, result.target, result.attempts),
                                 (action, target, 3))

    def test_exhausted_budget_never_confirms_or_increments(self):
        for attempts in (3, 4):
            for digit in ("1", "2", None):
                result = self.resolve(digit, attempts=attempts)
                self.assertEqual((result.action, result.target, result.attempts),
                                 ("fallback", "team/Reception", attempts))

    def test_unknown_or_malformed_pending_intent_never_retries_or_routes(self):
        for intent in (None, "", "unknown", "Support", "support ", 1, [], {}):
            for digit in ("1", "2", None):
                with self.subTest(intent=intent, digit=digit):
                    result = self.resolve(digit, intent=intent)
                    self.assertEqual((result.action, result.target, result.attempts),
                                     ("fallback", "team/Reception", 1))
        self.assertEqual(self.resolve("1", routes={}).action, "fallback")

    def test_confirmation_can_select_same_target_as_fallback(self):
        result = self.resolve("1", routes={"support": "team/Reception"})
        self.assertEqual((result.action, result.target), ("route", "team/Reception"))

    def test_explicit_exit_consumes_one_attempt_without_a_destination(self):
        for digit in "03456789":
            for attempts in (0, 2):
                with self.subTest(digit=digit, attempts=attempts):
                    result = self.resolve(digit, hangup_digit=digit, attempts=attempts)
                    self.assertEqual((result.action, result.target, result.attempts),
                                     ("hangup", None, attempts + 1))

    def test_exit_does_not_override_missing_intent_or_exhausted_budget(self):
        for overrides, attempts in (({"intent": None}, 1), ({"routes": {}}, 1),
                                    ({"intent": "unknown"}, 1), ({"attempts": 3}, 3)):
            with self.subTest(overrides=overrides):
                result = self.resolve("9", hangup_digit="9", **overrides)
                self.assertEqual((result.action, result.target, result.attempts),
                                 ("fallback", "team/Reception", attempts))

    def test_exit_requires_an_exact_match_and_preserves_other_choices(self):
        for digits in (9, "９", " 9", "9 ", "99", None):
            with self.subTest(digits=digits):
                self.assertEqual(self.resolve(digits, hangup_digit="9").action, "retry")
        self.assertEqual(self.resolve("9").action, "retry")
        self.assertEqual(self.resolve("9", hangup_digit=None).action, "retry")
        self.assertEqual(self.resolve("1", hangup_digit="9").action, "route")
        self.assertEqual(self.resolve("2", hangup_digit="9").action, "fallback")

    def test_exit_configuration_always_rejects_reserved_or_invalid_keys(self):
        for base in ({}, {"intent": None}, {"routes": {}}, {"attempts": 3}):
            for digit in ("1", "2", "", "99", "９", "*", "#", " 9", 9, True, [], {}):
                with self.subTest(base=base, digit=digit), self.assertRaises(ValueError):
                    self.resolve("9", hangup_digit=digit, **base)

    def test_retry_then_exit_composes_with_terminal_renderer_offline(self):
        with patch.dict("os.environ", {}, clear=True), \
                patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            retry = self.resolve(None, hangup_digit="9")
            result = self.resolve("9", hangup_digit="9", attempts=retry.attempts)
            self.assertEqual((result.action, result.target, result.attempts),
                             ("hangup", None, 2))
            root = fromstring(render_hangup())
        self.assertEqual([child.tag for child in root], ["Hangup"])

    def test_custom_budget_and_unknown_intent_after_exhaustion(self):
        self.assertEqual(self.resolve(None, max_attempts=1).action, "fallback")
        self.assertEqual(self.resolve("1", max_attempts=1).action, "route")
        result = self.resolve("1", intent=None, attempts=5, max_attempts=1)
        self.assertEqual((result.action, result.target, result.attempts),
                         ("fallback", "team/Reception", 5))

    def test_configuration_is_validated_even_for_unknown_intent_or_exhaustion(self):
        invalid = ({"routes": []}, {"routes": {"unused": " "}},
                   {"routes": {"": "target"}}, {"fallback_target": ""},
                   {"attempts": True}, {"attempts": -1},
                   {"max_attempts": False}, {"max_attempts": 0})
        for base in ({}, {"intent": None}, {"attempts": 3}):
            for overrides in invalid:
                with self.subTest(base=base, overrides=overrides), self.assertRaises(ValueError):
                    self.resolve("1", **{**base, **overrides})

    def test_retry_then_confirmation_composes_with_existing_renderer(self):
        prompt = "For Support, press 1 to confirm or 2 for reception."
        root = fromstring(render_dtmf_gather(prompt, action_path="/voice/confirm"))
        self.assertEqual(root.find("Gather").attrib["numDigits"], "1")
        retry = self.resolve(None)
        confirmed = self.resolve("1", attempts=retry.attempts)
        self.assertEqual((confirmed.action, confirmed.attempts), ("route", 2))

    def test_results_are_immutable_and_policy_is_offline(self):
        routes = {"support": "team/Support"}
        with patch.dict("os.environ", {}, clear=True), \
                patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            result = self.resolve("1", routes=routes)
        self.assertEqual(routes, {"support": "team/Support"})
        with self.assertRaises(FrozenInstanceError):
            result.attempts = 10


if __name__ == "__main__":
    unittest.main()
