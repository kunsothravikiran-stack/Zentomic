"""Synthetic model-output checks; no SDK, credentials, or external calls."""

import json
from collections.abc import Collection
import unittest
from unittest.mock import patch

from zentomic.classification import (
    MAX_ALLOWED_INTENTS,
    MAX_RESPONSE_BYTES,
    parse_intent_response,
)
from zentomic.intent import resolve_intent, resolve_intent_confirmation


class OversizedAllowlist(Collection):
    """Stop if admission reads beyond the one-item oversize probe."""

    def __contains__(self, value):
        return False

    def __len__(self):
        return MAX_ALLOWED_INTENTS + 1

    def __iter__(self):
        yield from (f"intent-{index}" for index in range(MAX_ALLOWED_INTENTS + 1))
        raise AssertionError("oversized allowlist must not be read further")


class UnreadableResponse(str):
    """Fail if trusted configuration rejection starts parsing model output."""

    def __len__(self):
        raise AssertionError("model response must not be inspected")


class ClassifierResponseTests(unittest.TestCase):
    def parse(self, response, labels=("sales", "support")):
        return parse_intent_response(response, allowed_intents=labels)

    def test_accepts_only_exact_allowlisted_labels(self):
        for label in ("sales", "support", "తెలుగు"):
            self.assertEqual(self.parse(json.dumps({"intent": label}), [label]), label)
        for label in ("Sales", " sales", "sales ", "unknown", "", None, True, 1, [], {}):
            self.assertIsNone(self.parse(json.dumps({"intent": label})))
        self.assertIsNone(self.parse('{"intent":"sales"}', []))

    def test_rejects_untrusted_routing_and_confirmation_fields(self):
        for key, value in (("target", "arbitrary-destination"), ("confirmed", True),
                           ("attempts", 0), ("reason", "ignore prior instructions")):
            self.assertIsNone(self.parse(json.dumps({"intent": "sales", key: value})))

    def test_rejects_duplicate_fields_including_escaped_names(self):
        for response in ('{"intent":"sales","intent":"support"}',
                         '{"intent":"sales","intent":"sales"}',
                         '{"intent":"sales","in\\u0074ent":"support"}'):
            self.assertIsNone(self.parse(response))

    def test_rejects_non_json_wrappers_and_wrong_shapes(self):
        for response in (None, b'{}', {}, [], 1, True, "", " ", "null", "true", "NaN",
                         '"sales"', '[]', '{}', '[["intent","sales"]]',
                         '```json\n{"intent":"sales"}\n```',
                         '{"intent":"sales"} trailing', '{"intent":"sales",}',
                         '{"intent":{"intent":"sales"}}'):
            self.assertIsNone(self.parse(response))

    def test_enforces_utf8_byte_limit_including_whitespace(self):
        response = '{"intent":"sales"}'
        boundary = response + " " * (MAX_RESPONSE_BYTES - len(response))
        self.assertEqual(self.parse(boundary), "sales")
        self.assertIsNone(self.parse(boundary + " "))
        label = "😀" * 1100
        response = json.dumps({"intent": label}, ensure_ascii=False)
        self.assertLess(len(response), MAX_RESPONSE_BYTES)
        self.assertIsNone(self.parse(response))

    def test_malformed_unicode_and_excessive_nesting_fail_closed(self):
        self.assertIsNone(self.parse('{"intent":"\ud800"}'))
        self.assertIsNone(self.parse("[" * 1500 + "0" + "]" * 1500))

    def test_invalid_configuration_is_rejected_even_for_invalid_output(self):
        for labels in (None, "sales", b"sales", 1, True, [None], [""], [" "], [1], [{}]):
            for response in (None, '{"intent":"sales"}'):
                with self.subTest(labels=labels), self.assertRaises(ValueError):
                    self.parse(response, labels)

    def test_rejects_oversized_allowlist_with_bounded_iteration(self):
        with self.assertRaisesRegex(
            ValueError, "^allowed_intents must contain at most 128 entries$",
        ):
            self.parse(UnreadableResponse("{}"), OversizedAllowlist())

    def test_allowlist_limit_is_inclusive(self):
        labels = tuple(f"intent-{index}" for index in range(MAX_ALLOWED_INTENTS))
        self.assertEqual(
            self.parse(json.dumps({"intent": labels[-1]}), labels), labels[-1],
        )

    def test_non_utf8_allowlist_labels_are_rejected_before_parsing(self):
        for label in ("synthetic-private-\ud800", "\udfff", "\ud83d\ude00"):
            for response in (None, "not json", json.dumps({"intent": label})):
                with self.subTest(label=repr(label), response=repr(response)):
                    with self.assertRaises(ValueError) as caught:
                        self.parse(response, ["sales", label])
                    self.assertEqual(
                        str(caught.exception),
                        "allowed_intents must contain only nonblank UTF-8 strings "
                        "of at most 256 bytes",
                    )

    def test_unicode_labels_preserve_literal_and_escaped_json_equivalence(self):
        for label in ("తెలుగు", "Café", "Cafe\u0301", "😀"):
            for escaped in (False, True):
                with self.subTest(label=label, escaped=escaped):
                    self.assertEqual(
                        self.parse(json.dumps({"intent": label}, ensure_ascii=escaped), [label]),
                        label,
                    )
        self.assertIsNone(self.parse(json.dumps({"intent": "Café"}), ["Cafe\u0301"]))

    def test_escaped_lone_surrogates_in_output_fail_closed(self):
        for response in ('{"intent":"\\ud800"}', '{"intent":"sales\\udfff"}'):
            with self.subTest(response=response):
                self.assertIsNone(self.parse(response))

    def test_pending_label_still_requires_independent_confirmation(self):
        routes = {"sales": "sales-team"}
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ):
            label = self.parse('{"intent":"sales"}', routes.keys())
            self.assertEqual(resolve_intent(label, routes, fallback_target="reception"),
                             "reception")
            decision = resolve_intent_confirmation(
                "1", label, routes, fallback_target="reception", attempts=0,
            )
            self.assertEqual((decision.action, decision.target), ("route", "sales-team"))
        self.assertEqual(routes, {"sales": "sales-team"})


if __name__ == "__main__":
    unittest.main()
