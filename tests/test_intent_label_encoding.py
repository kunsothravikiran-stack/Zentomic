"""Classifier allowlists and routing menus must agree on label encoding."""

import json
import unittest

from zentomic.classification import parse_intent_response
from zentomic.intent import resolve_intent, resolve_intent_confirmation
from zentomic.labels import MAX_INTENT_LABEL_BYTES


class IntentLabelEncodingTests(unittest.TestCase):
    def test_byte_limit_accepts_exact_boundaries_across_admission_and_routing(self):
        for label in (
            "a" * MAX_INTENT_LABEL_BYTES,
            "é" * (MAX_INTENT_LABEL_BYTES // 2),
            "😀" * (MAX_INTENT_LABEL_BYTES // 4),
        ):
            routes = {label: "team/support"}
            with self.subTest(label=label[:4]):
                self.assertEqual(
                    parse_intent_response(
                        json.dumps({"intent": label}, ensure_ascii=False),
                        allowed_intents=routes,
                    ),
                    label,
                )
                self.assertEqual(
                    resolve_intent(
                        label, routes, fallback_target="reception", confirmed=True,
                    ),
                    "team/support",
                )

    def test_oversized_configured_labels_fail_closed_without_echoing_contents(self):
        for label in (
            "a" * (MAX_INTENT_LABEL_BYTES + 1),
            "é" * (MAX_INTENT_LABEL_BYTES // 2) + "a",
            "😀" * (MAX_INTENT_LABEL_BYTES // 4) + "a",
            "synthetic-private-marker-" + "x" * 10000,
        ):
            routes = {label: "team/support"}
            with self.subTest(label=label[:4]):
                for operation in (
                    lambda: parse_intent_response(None, allowed_intents=routes),
                    lambda: resolve_intent(
                        None, routes, fallback_target="reception", confirmed=False,
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "at most 256 bytes") as caught:
                        operation()
                    self.assertNotIn("synthetic-private-marker", str(caught.exception))

    def test_clearly_oversized_labels_are_rejected_before_content_work(self):
        class OversizedLabel(str):
            def strip(self, *args, **kwargs):
                raise AssertionError("oversized label must not be stripped")

            def encode(self, *args, **kwargs):
                raise AssertionError("oversized label must not be encoded")

        label = OversizedLabel("x" * (MAX_INTENT_LABEL_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "at most 256 bytes"):
            parse_intent_response(None, allowed_intents=[label])
        with self.assertRaisesRegex(ValueError, "at most 256 bytes"):
            resolve_intent(
                None, {label: "team/support"},
                fallback_target="reception", confirmed=False,
            )

    def test_oversized_incoming_label_falls_back_before_hashing(self):
        class OversizedLabel(str):
            def __hash__(self):
                raise AssertionError("oversized label must not be hashed")

        label = OversizedLabel("x" * (MAX_INTENT_LABEL_BYTES + 1))
        routes = {"sales": "team/sales"}
        self.assertEqual(
            resolve_intent(
                label, routes, fallback_target="reception", confirmed=True,
            ),
            "reception",
        )
        decision = resolve_intent_confirmation(
            "1", label, routes, fallback_target="reception", attempts=0,
        )
        self.assertEqual(
            (decision.action, decision.target, decision.attempts),
            ("fallback", "reception", 1),
        )

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
