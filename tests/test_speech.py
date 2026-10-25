"""Synthetic speech admission checks, with no model or telephony calls."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from zentomic.intent import resolve_intent
from zentomic.speech import (
    MAX_CONFIDENCE_CHARACTERS,
    MAX_TRANSCRIPT_CHARS,
    MAX_TRANSCRIPT_LINES,
    MAX_TRANSCRIPT_UTF8_BYTES,
    MAX_TRANSCRIPT_WORDS,
    SpeechDecision,
    resolve_speech_gather,
)
from zentomic.webhook import parse_form_body


class SpeechDecisionTests(unittest.TestCase):
    def decide(self, speech, **kwargs):
        return resolve_speech_gather(
            speech, fallback_target="reception", **{"attempts": 0, **kwargs},
        )

    def test_valid_text_is_preserved_without_selecting_a_target(self):
        for speech in ("Sales please", "  Billing?\n", "తెలుగు café 😀"):
            with self.subTest(speech=speech):
                self.assertEqual(self.decide(speech), SpeechDecision("classify", speech, None, 1))

    def test_representations_do_not_expose_admitted_transcripts(self):
        speech = "Synthetic private caller statement"
        decision = self.decide(speech)
        for rendered in (repr(decision), str(decision), f"{decision!r}",
                         repr({"decision": decision}), repr([decision])):
            with self.subTest(rendered=rendered):
                self.assertNotIn(speech, rendered)
                self.assertNotIn("transcript=", rendered)
                self.assertIn("action='classify'", rendered)
                self.assertIn("attempts=1", rendered)

    def test_transcript_redaction_preserves_access_and_value_semantics(self):
        decision = self.decide("Synthetic caller A")
        self.assertEqual(decision.transcript, "Synthetic caller A")
        self.assertEqual(decision, self.decide("Synthetic caller A"))
        self.assertNotEqual(decision, self.decide("Synthetic caller B"))
        self.assertEqual(hash(decision), hash(self.decide("Synthetic caller A")))
        with self.assertRaises(FrozenInstanceError):
            decision.transcript = "replacement"

    def test_missing_blank_and_malformed_results_retry_then_fall_back(self):
        for speech in (None, "", " \t\n", "\u2003", 1, True, [], {}, b"sales"):
            with self.subTest(speech=speech):
                self.assertEqual(self.decide(speech), SpeechDecision("retry", None, None, 1))
                self.assertEqual(self.decide(speech, attempts=2),
                                 SpeechDecision("fallback", None, "reception", 3))

    def test_character_limit_includes_surrounding_whitespace(self):
        for speech in ("x" * MAX_TRANSCRIPT_CHARS, "😀" * MAX_TRANSCRIPT_CHARS):
            with self.subTest(first_character=speech[0]):
                self.assertEqual(self.decide(speech).transcript, speech)
                self.assertEqual(self.decide(speech + "x").action, "retry")
        self.assertEqual(self.decide(" " * MAX_TRANSCRIPT_CHARS + "x").action, "retry")

    def test_trusted_transcript_limit_can_reduce_classifier_input(self):
        for limit in (1, 80, MAX_TRANSCRIPT_CHARS):
            with self.subTest(limit=limit):
                accepted = "x" * limit
                self.assertEqual(
                    self.decide(
                        accepted, max_transcript_characters=limit,
                    ),
                    SpeechDecision("classify", accepted, None, 1),
                )
                self.assertEqual(
                    self.decide(
                        accepted + "x", max_transcript_characters=limit,
                    ),
                    SpeechDecision("retry", None, None, 1),
                )

    def test_trusted_minimum_counts_trimmed_characters(self):
        for speech, minimum, expected in (
            ("x", 1, "classify"),
            (" x ", 2, "retry"),
            (" xy ", 2, "classify"),
            ("\u2003xy\n", 3, "retry"),
            ("\u2003xyz\n", 3, "classify"),
        ):
            with self.subTest(speech=repr(speech), minimum=minimum):
                decision = self.decide(
                    speech, min_transcript_characters=minimum,
                )
                self.assertEqual(decision.action, expected)
                if expected == "classify":
                    self.assertEqual(decision.transcript, speech)

    def test_trusted_minimum_is_strict_and_cannot_exceed_maximum(self):
        for minimum in (0, -1, MAX_TRANSCRIPT_CHARS + 1, True, False,
                        2.0, "2", None, [], {}):
            with self.subTest(minimum=repr(minimum)), self.assertRaises(ValueError):
                self.decide("sales", min_transcript_characters=minimum)
        with self.assertRaisesRegex(
            ValueError,
            "^min_transcript_characters must not exceed max_transcript_characters$",
        ):
            self.decide(
                "sales", min_transcript_characters=6,
                max_transcript_characters=5,
            )

    def test_trusted_transcript_limit_is_strict(self):
        for limit in (0, -1, MAX_TRANSCRIPT_CHARS + 1, True, False,
                      80.0, "80", None, [], {}):
            with self.subTest(limit=repr(limit)), self.assertRaises(ValueError):
                self.decide("sales", max_transcript_characters=limit)

    def test_trusted_utf8_byte_limit_bounds_multibyte_classifier_input(self):
        for speech, limit, expected in (
            ("cafe", 4, "classify"),
            ("café", 5, "classify"),
            ("café", 4, "retry"),
            ("😀", 4, "classify"),
            ("😀", 3, "retry"),
        ):
            with self.subTest(speech=speech, limit=limit):
                self.assertEqual(
                    self.decide(speech, max_transcript_utf8_bytes=limit).action,
                    expected,
                )
        self.assertEqual(
            self.decide("😀" * MAX_TRANSCRIPT_CHARS).transcript,
            "😀" * MAX_TRANSCRIPT_CHARS,
        )

    def test_trusted_utf8_byte_minimum_counts_trimmed_content(self):
        for speech, minimum, expected in (
            ("e", 1, "classify"),
            ("e", 2, "retry"),
            ("é", 2, "classify"),
            ("  é  ", 3, "retry"),
            ("\u2003😀\n", 4, "classify"),
        ):
            with self.subTest(speech=repr(speech), minimum=minimum):
                decision = self.decide(
                    speech, min_transcript_utf8_bytes=minimum,
                )
                self.assertEqual(decision.action, expected)
                if expected == "classify":
                    self.assertEqual(decision.transcript, speech)

    def test_trusted_utf8_byte_minimum_is_strict_and_bounded(self):
        for minimum in (0, -1, MAX_TRANSCRIPT_UTF8_BYTES + 1, True, False,
                        2.0, "2", None, [], {}):
            with self.subTest(minimum=repr(minimum)), self.assertRaises(ValueError):
                self.decide("sales", min_transcript_utf8_bytes=minimum)
        with self.assertRaisesRegex(
            ValueError,
            "^min_transcript_utf8_bytes must not exceed max_transcript_utf8_bytes$",
        ):
            self.decide(
                "sales", min_transcript_utf8_bytes=6,
                max_transcript_utf8_bytes=5,
            )

    def test_trusted_utf8_byte_limit_is_strict(self):
        for limit in (0, -1, MAX_TRANSCRIPT_UTF8_BYTES + 1, True, False,
                      80.0, "80", None, [], {}):
            with self.subTest(limit=repr(limit)), self.assertRaises(ValueError):
                self.decide("sales", max_transcript_utf8_bytes=limit)

    def test_trusted_word_limit_bounds_whitespace_delimited_input(self):
        for speech, limit, expected in (
            ("one two", 2, "classify"),
            ("one two", 1, "retry"),
            ("  one\ttwo\n", 2, "classify"),
            ("one\u00a0two", 1, "retry"),
            ("没有空格", 1, "classify"),
        ):
            with self.subTest(speech=speech, limit=limit):
                self.assertEqual(
                    self.decide(speech, max_transcript_words=limit).action,
                    expected,
                )
        default_boundary = " ".join("x" for _ in range(MAX_TRANSCRIPT_WORDS))
        self.assertEqual(self.decide(default_boundary).transcript, default_boundary)

    def test_trusted_word_minimum_bounds_whitespace_delimited_input(self):
        for speech, minimum, expected in (
            ("one", 1, "classify"),
            ("one", 2, "retry"),
            ("  one\ttwo\n", 2, "classify"),
            ("one\u00a0two", 3, "retry"),
            ("没有空格", 2, "retry"),
        ):
            with self.subTest(speech=speech, minimum=minimum):
                decision = self.decide(speech, min_transcript_words=minimum)
                self.assertEqual(decision.action, expected)
                if expected == "classify":
                    self.assertEqual(decision.transcript, speech)

    def test_trusted_word_minimum_is_strict_and_cannot_exceed_maximum(self):
        for minimum in (0, -1, MAX_TRANSCRIPT_WORDS + 1, True, False,
                        2.0, "2", None, [], {}):
            with self.subTest(minimum=repr(minimum)), self.assertRaises(ValueError):
                self.decide("sales please", min_transcript_words=minimum)
        with self.assertRaisesRegex(
            ValueError,
            "^min_transcript_words must not exceed max_transcript_words$",
        ):
            self.decide(
                "sales please", min_transcript_words=3,
                max_transcript_words=2,
            )

    def test_trusted_minimums_must_fit_cross_dimension_maximums(self):
        for policy, message in (
            (
                {"min_transcript_characters": 5,
                 "max_transcript_utf8_bytes": 4},
                "min_transcript_characters must not exceed "
                "max_transcript_utf8_bytes",
            ),
            (
                {"min_transcript_words": 3,
                 "max_transcript_characters": 4},
                "min_transcript_words cannot fit within "
                "max_transcript_characters",
            ),
            (
                {"min_transcript_words": 3,
                 "max_transcript_utf8_bytes": 4},
                "min_transcript_words cannot fit within "
                "max_transcript_utf8_bytes",
            ),
            (
                {"min_transcript_utf8_bytes": 12,
                 "max_transcript_utf8_bytes": 12,
                 "min_transcript_words": 2,
                 "max_transcript_characters": 3},
                "min_transcript_utf8_bytes cannot fit within "
                "max_transcript_characters and min_transcript_words",
            ),
        ):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, f"^{message}$",
            ):
                self.decide("a b c", **policy)

        self.assertEqual(
            self.decide(
                "a b c", min_transcript_characters=5,
                max_transcript_characters=5, max_transcript_utf8_bytes=5,
                min_transcript_words=3,
            ),
            SpeechDecision("classify", "a b c", None, 1),
        )
        multibyte_boundary = "\U00010000\u3000\U00010000"
        self.assertEqual(
            self.decide(
                multibyte_boundary, min_transcript_utf8_bytes=11,
                max_transcript_utf8_bytes=11, min_transcript_words=2,
                max_transcript_characters=3,
            ),
            SpeechDecision("classify", multibyte_boundary, None, 1),
        )

    def test_trusted_word_limit_is_strict(self):
        for limit in (0, -1, MAX_TRANSCRIPT_WORDS + 1, True, False,
                      80.0, "80", None, [], {}):
            with self.subTest(limit=repr(limit)), self.assertRaises(ValueError):
                self.decide("sales", max_transcript_words=limit)

    def test_trusted_line_limit_bounds_unicode_multiline_input(self):
        for speech, limit, expected in (
            ("sales", 1, "classify"),
            ("sales\nsupport", 1, "retry"),
            ("sales\r\nsupport", 2, "classify"),
            ("sales\u2028support\u2029billing", 2, "retry"),
            ("sales\u2028support\u2029billing", 3, "classify"),
        ):
            with self.subTest(speech=repr(speech), limit=limit):
                decision = self.decide(speech, max_transcript_lines=limit)
                self.assertEqual(decision.action, expected)
                if expected == "classify":
                    self.assertEqual(decision.transcript, speech)

    def test_trusted_line_limit_is_strict(self):
        for limit in (0, -1, MAX_TRANSCRIPT_LINES + 1, True, False,
                      2.0, "2", None, [], {}):
            with self.subTest(limit=repr(limit)), self.assertRaises(ValueError):
                self.decide("sales", max_transcript_lines=limit)

    def test_invalid_unicode_never_reaches_classification(self):
        # Python strings (including decoded JSON) may contain surrogate code
        # points that cannot be sent as UTF-8 to a classifier dependency.
        for speech in ("\ud800", "\udfff", "sales \ud800 please", "\ud83d\ude00"):
            for attempts, expected in ((0, "retry"), (2, "fallback"), (3, "fallback")):
                with self.subTest(speech=repr(speech), attempts=attempts):
                    decision = self.decide(speech, attempts=attempts)
                    self.assertEqual(decision.action, expected)
                    self.assertEqual(decision.attempts, min(attempts + 1, 3))
                    self.assertIsNone(decision.transcript)
                    self.assertEqual(decision.target, "reception" if expected == "fallback" else None)

    def test_valid_unicode_is_not_normalized_or_replaced(self):
        for speech in ("cafe\u0301", "\ud7ff\ue000", "\U00010000\U0010ffff", "😀"):
            with self.subTest(speech=repr(speech)):
                self.assertEqual(self.decide(speech), SpeechDecision("classify", speech, None, 1))

    def test_valid_final_attempt_classifies_but_exhaustion_never_does(self):
        self.assertEqual(self.decide("sales", attempts=2), SpeechDecision("classify", "sales", None, 3))
        self.assertEqual(self.decide("sales", max_attempts=1).action, "classify")
        for attempts in (3, 10):
            with self.subTest(attempts=attempts):
                self.assertEqual(self.decide("sales", attempts=attempts),
                                 SpeechDecision("fallback", None, "reception", attempts))

    def test_optional_confidence_threshold_retries_only_explicit_low_scores(self):
        for confidence in (None, "0.80", "0.8000", "1", "1.0"):
            with self.subTest(confidence=confidence):
                self.assertEqual(
                    self.decide(
                        "sales", confidence=confidence,
                        minimum_confidence_percent=80,
                    ),
                    SpeechDecision("classify", "sales", None, 1),
                )
        for confidence in ("0", "0.799999", "0.01"):
            with self.subTest(confidence=confidence):
                self.assertEqual(
                    self.decide(
                        "sales", confidence=confidence,
                        minimum_confidence_percent=80,
                    ),
                    SpeechDecision("retry", None, None, 1),
                )

    def test_configured_threshold_rejects_malformed_present_scores_privately(self):
        for confidence in (
            "", " ", "00.8", ".8", "0.", "1.01", "-0.1", "+0.8", "8e-1",
            "nan", "inf", "private-confidence", "0.8\n", "0.8\ud800",
            "0." + "1" * MAX_CONFIDENCE_CHARACTERS, 0.8, True, [], {},
        ):
            with self.subTest(confidence=repr(confidence)[:40]):
                decision = self.decide(
                    "sales", confidence=confidence,
                    minimum_confidence_percent=80,
                )
                self.assertEqual(decision, SpeechDecision("retry", None, None, 1))
                self.assertNotIn("private-confidence", repr(decision))

    def test_confidence_is_ignored_without_an_explicit_threshold(self):
        for confidence in (None, "0", "private-confidence", 0.1, []):
            with self.subTest(confidence=repr(confidence)):
                self.assertEqual(
                    self.decide("sales", confidence=confidence),
                    SpeechDecision("classify", "sales", None, 1),
                )

    def test_confidence_presence_can_be_required_without_a_threshold(self):
        for confidence in ("0", "0.80", "1.0"):
            with self.subTest(confidence=confidence):
                self.assertEqual(
                    self.decide(
                        "sales", confidence=confidence, require_confidence=True,
                    ),
                    SpeechDecision("classify", "sales", None, 1),
                )
        for confidence in (None, "", "private-confidence", 0.8):
            with self.subTest(confidence=repr(confidence)):
                decision = self.decide(
                    "sales", confidence=confidence, require_confidence=True,
                )
                self.assertEqual(decision, SpeechDecision("retry", None, None, 1))
                self.assertNotIn("private-confidence", repr(decision))

    def test_confidence_threshold_configuration_is_strict(self):
        for threshold in (0, 1, 80, 100):
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    self.decide(
                        "sales", confidence="1",
                        minimum_confidence_percent=threshold,
                    ).action,
                    "classify",
                )
        for threshold in (-1, 101, True, False, 80.0, "80", [], {}):
            with self.subTest(threshold=repr(threshold)), self.assertRaises(ValueError):
                self.decide(
                    "sales", confidence="0.8",
                    minimum_confidence_percent=threshold,
                )
        for required in (None, 0, 1, "true", [], {}):
            with self.subTest(required=repr(required)), self.assertRaises(ValueError):
                self.decide("sales", confidence="0.8", require_confidence=required)

    def test_repeated_silence_terminates_without_resetting_the_budget(self):
        attempts = 0
        for expected in ("retry", "retry", "fallback", "fallback"):
            decision = self.decide(None, attempts=attempts)
            self.assertEqual(decision.action, expected)
            attempts = decision.attempts
        self.assertEqual(attempts, 3)

    def test_invalid_configuration_is_rejected_for_all_input_paths(self):
        for speech in (None, "sales", "x" * (MAX_TRANSCRIPT_CHARS + 1)):
            for key, values in (("attempts", (-1, True, 1.0, "1", None)),
                                ("max_attempts", (0, -1, False, 1.0, "3", None))):
                for value in values:
                    with self.subTest(speech=speech[:5] if speech else speech, key=key, value=value):
                        with self.assertRaises(ValueError):
                            self.decide(speech, **{key: value})
            for target in (None, "", " ", False, {}):
                for attempts in (0, 3):
                    with self.subTest(target=target, attempts=attempts), self.assertRaises(ValueError):
                        resolve_speech_gather(speech, fallback_target=target, attempts=attempts)

    def test_form_fields_do_not_supply_budget_or_confirm_a_route(self):
        # Transport composition only; a production adapter must authenticate first.
        fields = parse_form_body("SpeechResult=sales&attempts=0&confirmed=true")
        decision = self.decide(fields.get("SpeechResult"), attempts=2)
        self.assertEqual(decision.attempts, 3)
        self.assertIsNone(decision.target)
        self.assertEqual(resolve_intent(decision.transcript, {"sales": "sales-team"},
                                        fallback_target="reception"), "reception")

    def test_offline_immutable_and_no_transcript_on_retry_or_fallback(self):
        with patch.dict("os.environ", {}, clear=True), patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            decision = self.decide("sales")
            with self.assertRaises(FrozenInstanceError):
                decision.attempts = 0
            self.assertIsNone(self.decide("private synthetic text", attempts=3).transcript)
            self.assertIsNone(self.decide("x" * (MAX_TRANSCRIPT_CHARS + 1)).transcript)


if __name__ == "__main__":
    unittest.main()
