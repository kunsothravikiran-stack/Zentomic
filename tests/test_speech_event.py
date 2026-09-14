"""Offline authenticated speech admission with synthetic signature fixtures."""

import copy
import unittest
from unittest.mock import Mock, patch

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.callback_claim_store import (
    CallbackReplayError,
    InMemoryCallbackClaimStore,
)
from zentomic.speech import MAX_TRANSCRIPT_CHARS, SpeechDecision
from zentomic.speech_event import resolve_claimed_speech_event, resolve_speech_event


class SpeechEventTests(unittest.TestCase):
    def resolve(self, event, validator, **overrides):
        config = dict(public_url="https://example.invalid/voice/speech",
                      expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                      fallback_target="demo-reception", attempts=0, max_attempts=3)
        config.update(overrides)
        return resolve_speech_event(event, validator=validator, **config)

    def test_authenticated_speech_and_silence_obey_trusted_budget(self):
        text = "  Synthetic support request \u2603  "
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, attempts, expected in (
                    ({"SpeechResult": text}, 0, SpeechDecision("classify", text, None, 1)),
                    ({"SpeechResult": text}, 2, SpeechDecision("classify", text, None, 3)),
                    ({}, 0, SpeechDecision("retry", None, None, 1)),
                    ({"SpeechResult": "  "}, 1, SpeechDecision("retry", None, None, 2)),
                    ({"SpeechResult": ""}, 2,
                     SpeechDecision("fallback", None, "demo-reception", 3)),
                    ({"SpeechResult": "x" * (MAX_TRANSCRIPT_CHARS + 1)}, 0,
                     SpeechDecision("retry", None, None, 1)),
                    ({"SpeechResult": text}, 3,
                     SpeechDecision("fallback", None, "demo-reception", 3)),
                ):
                    with self.subTest(version=version, encoded=encoded, attempts=attempts,
                                      action=expected.action):
                        event = callback(dict(fields, attempts="0", max_attempts="999",
                                              fallback_target="untrusted", Digits="1",
                                              transcript="untrusted"), version, encoded)
                        original = copy.deepcopy(event)
                        validator = Mock(return_value=True)
                        self.assertEqual(self.resolve(event, validator, attempts=attempts), expected)
                        validator.assert_called_once()
                        self.assertEqual(event, original)

    def test_rejected_signature_or_call_identity_never_reaches_speech_policy(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, valid in (({}, False), ({}, 1),
                                      ({"AccountSid": "other-account"}, True),
                                      ({"CallSid": "other-call"}, True)):
                    with self.subTest(version=version, encoded=encoded, fields=fields,
                                      valid=valid), patch(
                        "zentomic.speech_event.resolve_speech_gather",
                    ) as policy:
                        event = callback(dict(fields, SpeechResult="Synthetic request"),
                                         version, encoded)
                        with self.assertRaises(ValueError):
                            self.resolve(event, Mock(return_value=valid))
                        policy.assert_not_called()

    def test_validator_cannot_rewrite_or_invent_speech(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, expected in (
                    ({"SpeechResult": "Synthetic request"},
                     SpeechDecision("classify", "Synthetic request", None, 1)),
                    ({}, SpeechDecision("retry", None, None, 1)),
                ):
                    with self.subTest(version=version, encoded=encoded, fields=fields):
                        def mutate(url, values, signature):
                            values["SpeechResult"] = "Injected request"
                            return True

                        event = callback(fields, version, encoded)
                        original = copy.deepcopy(event)
                        self.assertEqual(self.resolve(event, mutate), expected)
                        self.assertEqual(event, original)

    def test_invalid_transport_fails_before_validator_and_policy(self):
        event = callback({}, "2.0", False)
        event["body"] += "&SpeechResult=first&SpeechResult=second"
        for invalid in ({}, event):
            validator = Mock(return_value=True)
            with self.subTest(event=invalid), patch(
                "zentomic.speech_event.resolve_speech_gather",
            ) as policy:
                with self.assertRaises(ValueError):
                    self.resolve(invalid, validator)
                validator.assert_not_called()
                policy.assert_not_called()

    def test_invalid_trusted_policy_configuration_is_not_hidden(self):
        for overrides in ({"attempts": True}, {"attempts": -1}, {"max_attempts": 0},
                          {"fallback_target": ""}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(callback({"SpeechResult": "Synthetic request"}, "2.0", False),
                             Mock(return_value=True), **overrides)


class ClaimedSpeechEventTests(unittest.TestCase):
    def resolve(self, event, claimer, validator=None, **overrides):
        config = dict(
            public_url="https://example.invalid/voice/speech",
            validator=validator or Mock(return_value=True),
            expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
            workspace_id="synthetic-workspace",
            step_id="speech-attempt-1",
            claimer=claimer,
            fallback_target="demo-reception",
            attempts=0,
            max_attempts=3,
        )
        config.update(overrides)
        return resolve_claimed_speech_event(event, **config)

    def test_authenticated_step_admits_speech_once_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    store = InMemoryCallbackClaimStore()
                    event = callback(
                        {"SpeechResult": "Synthetic support request",
                         "workspace_id": "untrusted", "step_id": "untrusted",
                         "attempts": "99"},
                        version, encoded,
                    )
                    original = copy.deepcopy(event)
                    self.assertEqual(
                        self.resolve(event, store),
                        SpeechDecision("classify", "Synthetic support request", None, 1),
                    )
                    self.assertEqual(event, original)
                    with self.assertRaisesRegex(
                        CallbackReplayError, "^callback step already claimed$",
                    ), patch("zentomic.speech_event._resolve_validated_speech") as policy:
                        self.resolve(event, store)
                    policy.assert_not_called()
                    self.assertFalse(store.claim(
                        "synthetic-workspace", CALL, "speech-attempt-1",
                    ))
                    self.assertTrue(store.claim("untrusted", CALL, "untrusted"))

    def test_rejected_callback_does_not_consume_the_step(self):
        for fields, valid in (
            ({}, False),
            ({"AccountSid": "other-account"}, True),
            ({"CallSid": "other-call"}, True),
        ):
            store = InMemoryCallbackClaimStore()
            with self.subTest(fields=fields, valid=valid), self.assertRaises(ValueError):
                self.resolve(callback(fields, "2.0", False), store, Mock(return_value=valid))
            self.assertTrue(store.claim(
                "synthetic-workspace", CALL, "speech-attempt-1",
            ))

    def test_invalid_policy_does_not_authenticate_or_claim(self):
        for overrides in (
            {"fallback_target": ""},
            {"attempts": True},
            {"max_attempts": 0},
        ):
            validator = Mock(return_value=True)
            store = InMemoryCallbackClaimStore()
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(callback({"SpeechResult": "request"}, "2.0", False),
                             store, validator, **overrides)
            validator.assert_not_called()
            self.assertTrue(store.claim(
                "synthetic-workspace", CALL, "speech-attempt-1",
            ))

    def test_fallback_is_validated_once_before_the_claim(self):
        class SingleValidationTarget(str):
            checks = 0

            def encode(self, *args, **kwargs):
                self.checks += 1
                if self.checks > 1:
                    raise AssertionError("fallback was revalidated after the claim")
                return super().encode(*args, **kwargs)

        target = SingleValidationTarget("demo-reception")
        claimer = Mock()
        claimer.claim.return_value = True
        self.assertEqual(
            self.resolve(callback({}, "2.0", False), claimer,
                         fallback_target=target, attempts=2),
            SpeechDecision("fallback", None, target, 3),
        )
        self.assertEqual(target.checks, 1)
        claimer.claim.assert_called_once_with(
            "synthetic-workspace", CALL, "speech-attempt-1",
        )


if __name__ == "__main__":
    unittest.main()
