"""Classifier allowlists and routing menus must agree on label encoding."""

import json
import unittest

from zentomic.classification import parse_intent_response
from zentomic.intent import resolve_intent, resolve_intent_confirmation


class IntentLabelEncodingTests(unittest.TestCase):
    def test_invalid_labels_fail_even_when_not_selected(self):
        for label in ("private-\ud800", "private-\udfff", "private-\ud83d\ude00"):
            routes = {"sales": "team/sales", label: "team/other"}
            for intent, confirmed in ((label, True), ("sales", True), (None, False)):
                with self.subTest(label=ascii(label), intent=ascii(intent), confirmed=confirmed):
                    with self.assertRaises(ValueError) as caught:
                        resolve_intent(intent, routes, fallback_target="reception",
                                       confirmed=confirmed)
                    self.assertNotIn("private-", str(caught.exception))
                    self.assertNotIsInstance(caught.exception, UnicodeError)

    def test_confirmation_validates_labels_before_every_outcome(self):
        routes = {"sales": "team/sales", "private-\ud800": "team/other"}
        for digits in (None, "1", "2", "9"):
            for attempts in (0, 3):
                with self.subTest(digits=digits, attempts=attempts):
                    with self.assertRaises(ValueError):
                        resolve_intent_confirmation(
                            digits, "sales", routes, fallback_target="reception",
                            attempts=attempts, hangup_digit="9",
                        )

    def test_valid_unicode_labels_survive_classification_and_confirmation(self):
        for label in (" 受付 ", "😀", "é", "e\u0301"):
            with self.subTest(label=label):
                routes = {label: "team/support"}
                pending = parse_intent_response(
                    json.dumps({"intent": label}), allowed_intents=routes,
                )
                self.assertEqual(pending, label)
                decision = resolve_intent_confirmation(
                    "1", pending, routes, fallback_target="reception", attempts=0,
                )
                self.assertEqual((decision.action, decision.target, decision.attempts),
                                 ("route", "team/support", 1))
                self.assertEqual(routes, {label: "team/support"})

    def test_encoding_validation_does_not_normalize_labels(self):
        routes = {"é": "composed", "e\u0301": "decomposed"}
        for label, target in routes.items():
            with self.subTest(label=label):
                self.assertEqual(resolve_intent(label, routes, fallback_target="reception",
                                                confirmed=True), target)
        self.assertEqual(resolve_intent("\ud800", routes, fallback_target="reception",
                                        confirmed=True), "reception")


if __name__ == "__main__":
    unittest.main()
