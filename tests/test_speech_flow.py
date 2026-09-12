"""Offline speech-to-confirmation contracts, not a deployable call adapter.

Synthetic fixtures, fake signature validation, and a mocked classifier only.
Local pending labels/budgets do not provide persistence or replay protection.
"""

import unittest
from unittest.mock import Mock, call, patch
from xml.etree.ElementTree import fromstring

from tests.test_voice_flow import ACCOUNT, CALL, DESTINATIONS, FALLBACK, ROUTES, SIGNATURE, callback
from zentomic.classification import parse_intent_response
from zentomic.confirmation_event import resolve_confirmation_event
from zentomic.intent import resolve_intent
from zentomic.speech import MAX_TRANSCRIPT_CHARS
from zentomic.speech_event import resolve_speech_event
from zentomic.twiml import render_dial


class SpeechFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.validator = Mock(return_value=True)
        self.classifier = Mock(return_value='{"intent":"support"}')

    def collect(self, fields, version="2.0", encoded=False, attempts=0):
        decision = resolve_speech_event(
            callback(fields, version, encoded),
            public_url="https://example.invalid/voice/speech-result",
            validator=self.validator, expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            fallback_target=FALLBACK, attempts=attempts,
        )
        pending = None
        if decision.action == "classify":
            pending = parse_intent_response(
                self.classifier(decision.transcript), allowed_intents=ROUTES.keys(),
            )
        return decision, pending

    def confirm(self, fields, pending, version="2.0", encoded=False, attempts=0, routes=ROUTES):
        return resolve_confirmation_event(
            callback(fields, version, encoded), pending_intent=pending, routes=routes,
            public_url="https://example.invalid/voice/confirm",
            validator=self.validator, expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            fallback_target=FALLBACK,
            attempts=attempts, hangup_digit="9",
        )

    def test_speech_reaches_allowlisted_destination_only_after_confirmation(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.classifier.reset_mock()
                    self.validator.reset_mock()
                    text = "  Help with café Wi-Fi, please 😀  "
                    speech, pending = self.collect({"SpeechResult": text}, version, encoded)
                    self.classifier.assert_called_once_with(text)
                    self.assertEqual((speech.action, speech.target, pending),
                                     ("classify", None, "support"))
                    self.assertEqual(resolve_intent(pending, ROUTES, fallback_target=FALLBACK),
                                     FALLBACK)
                    confirmed = self.confirm({"Digits": "1"}, pending, version, encoded)
                    self.assertEqual(self.validator.call_args_list, [
                        call(
                            "https://example.invalid/voice/speech-result",
                            {"AccountSid": ACCOUNT, "CallSid": CALL, "SpeechResult": text},
                            SIGNATURE,
                        ),
                        call(
                            "https://example.invalid/voice/confirm",
                            {"AccountSid": ACCOUNT, "CallSid": CALL, "Digits": "1"},
                            SIGNATURE,
                        ),
                    ])
                    self.assertEqual((confirmed.action, confirmed.target), ("route", ROUTES[pending]))
                    xml = fromstring(render_dial(
                        DESTINATIONS[confirmed.target], action_path="/voice/dial-result",
                    ))
                    self.assertEqual([node.text for node in xml.findall("Dial/Number")],
                                     list(DESTINATIONS[ROUTES[pending]]))

    def test_rejected_speech_callbacks_never_invoke_policy_or_classifier(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for overrides, valid in (({}, False), ({"AccountSid": "other-account"}, True),
                                         ({"CallSid": "other-call"}, True)):
                    with self.subTest(version=version, encoded=encoded, overrides=overrides):
                        self.validator.return_value = valid
                        with patch("zentomic.speech_event.resolve_speech_gather") as policy:
                            with self.assertRaises(ValueError):
                                self.collect({"SpeechResult": "support", **overrides}, version, encoded)
                            policy.assert_not_called()
                        self.classifier.assert_not_called()

    def test_silence_oversize_and_exhaustion_never_invoke_classifier(self):
        for fields in ({}, {"SpeechResult": " \t"},
                       {"SpeechResult": "x" * (MAX_TRANSCRIPT_CHARS + 1)}):
            attempts = 0
            for expected in ("retry", "retry", "fallback", "fallback"):
                decision, pending = self.collect(fields, attempts=attempts)
                self.assertEqual(decision.action, expected)
                self.assertIsNone(pending)
                attempts = decision.attempts
            self.assertEqual(attempts, 3)
        decision, pending = self.collect({"SpeechResult": "support", "attempts": "0"}, attempts=3)
        self.assertEqual((decision.action, decision.attempts, pending), ("fallback", 3, None))
        self.classifier.assert_not_called()

    def test_invalid_classifier_output_cannot_be_confirmed_into_a_route(self):
        for output in ('{"intent":"unknown"}', '{"intent":"support","confirmed":true}',
                       '{"intent":"support","target":"other-team"}',
                       '{"intent":"support","intent":"support"}', "not JSON", None):
            with self.subTest(output=output):
                self.classifier.return_value = output
                _, pending = self.collect({"SpeechResult": "support"})
                self.assertIsNone(pending)
                result = self.confirm({"Digits": "1", "intent": "support"}, pending)
                self.assertEqual((result.action, result.target), ("fallback", FALLBACK))

    def test_speech_fields_cannot_supply_confirmation_or_reset_its_budget(self):
        speech, pending = self.collect({
            "SpeechResult": "support", "Digits": "1", "confirmed": "true",
            "intent": "other-team", "attempts": "0",
        }, attempts=2)
        self.assertEqual((speech.attempts, speech.target, pending), (3, None, "support"))
        # Separate test-owned budgets: the last speech attempt does not exhaust
        # a newly started confirmation step, nor reset an exhausted one.
        retry = self.confirm({}, pending)
        self.assertEqual((retry.action, retry.attempts), ("retry", 1))
        confirmed = self.confirm({"Digits": "1"}, pending, attempts=retry.attempts)
        self.assertEqual((confirmed.action, confirmed.attempts), ("route", 2))
        exhausted = self.confirm({"Digits": "1", "attempts": "0"}, pending, attempts=3)
        self.assertEqual((exhausted.action, exhausted.target, exhausted.attempts),
                         ("fallback", FALLBACK, 3))

    def test_pending_intent_does_not_bypass_confirmation_authentication_or_choice(self):
        _, pending = self.collect({"SpeechResult": "support"})
        for overrides, valid in (({}, False), ({"AccountSid": "other-account"}, True),
                                 ({"CallSid": "other-call"}, True)):
            self.validator.return_value = valid
            with patch("zentomic.confirmation_event.resolve_intent_confirmation") as policy:
                with self.assertRaises(ValueError):
                    self.confirm({"Digits": "1", **overrides}, pending)
                policy.assert_not_called()
        self.validator.return_value = True
        declined = self.confirm({"Digits": "2"}, pending)
        self.assertEqual((declined.action, declined.target), ("fallback", FALLBACK))
        ended = self.confirm({"Digits": "9"}, pending)
        self.assertEqual((ended.action, ended.target), ("hangup", None))

    def test_removed_route_cannot_be_restored_by_confirmation_fields(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.classifier.reset_mock()
                    _, pending = self.collect({"SpeechResult": "support"}, version, encoded)
                    # Model a trusted configuration reload between steps. The
                    # old label is not authority to restore a removed route.
                    result = self.confirm(
                        {"Digits": "1", "intent": "support", "target": ROUTES["support"]},
                        pending, version, encoded, routes={},
                    )
                    self.assertEqual((result.action, result.target, result.attempts),
                                     ("fallback", FALLBACK, 1))
                    self.classifier.assert_called_once_with("support")
                    xml = fromstring(render_dial(
                        DESTINATIONS[result.target], action_path="/voice/dial-result",
                    ))
                    self.assertEqual([node.text for node in xml.findall("Dial/Number")],
                                     list(DESTINATIONS[FALLBACK]))


if __name__ == "__main__":
    unittest.main()
