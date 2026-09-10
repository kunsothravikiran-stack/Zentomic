"""Synthetic speech admission checks, with no model or telephony calls."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from zentomic.intent import resolve_intent
from zentomic.speech import MAX_TRANSCRIPT_CHARS, SpeechDecision, resolve_speech_gather
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
