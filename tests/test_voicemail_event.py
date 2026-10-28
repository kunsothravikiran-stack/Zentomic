"""Offline voicemail action policy and authentication boundary tests."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.callback_claim_store import (
    CallbackReplayError,
    InMemoryCallbackClaimStore,
)
from zentomic.voicemail import VoicemailDecision, resolve_voicemail_result
from zentomic.voicemail_event import (
    resolve_claimed_voicemail_result_event,
    resolve_voicemail_result_event,
)


class VoicemailPolicyTests(unittest.TestCase):
    def test_supported_completion_reasons_are_sanitized(self):
        cases = (
            (None, VoicemailDecision("continue", 12, "silence-or-limit")),
            ("", VoicemailDecision("continue", 12, "silence-or-limit")),
            ("#", VoicemailDecision("continue", 12, "finish-key")),
            ("hangup", VoicemailDecision("hangup", 12, "hangup")),
        )
        for digits, expected in cases:
            with self.subTest(digits=digits):
                self.assertEqual(resolve_voicemail_result("12", digits), expected)

    def test_duration_must_be_canonical_and_within_trusted_bound(self):
        self.assertEqual(resolve_voicemail_result("0", None).duration_seconds, 0)
        self.assertEqual(
            resolve_voicemail_result(
                "600", "*", max_length=600, finish_on_key="*",
            ).duration_seconds,
            600,
        )
        for duration in (
            None, 1, "", "-1", "+1", "01", "1.0", " 1", "١", "601",
            "9" * 10000,
        ):
            with self.subTest(duration=repr(duration)[:30]), self.assertRaises(ValueError):
                resolve_voicemail_result(duration, None, max_length=600)
        with self.assertRaisesRegex(ValueError, "exceeds the configured maximum"):
            resolve_voicemail_result("121", None, max_length=120)

    def test_only_configured_finish_key_or_hangup_is_accepted(self):
        for digits in (True, 1, "*", "##", "Hangup", "silence", "x" * 10000):
            with self.subTest(digits=repr(digits)[:30]), self.assertRaises(ValueError):
                resolve_voicemail_result("10", digits)

    def test_trusted_policy_is_strictly_bounded(self):
        for max_length in (1, 601, True, 120.0, "120", None):
            with self.subTest(max_length=max_length), self.assertRaises(ValueError):
                resolve_voicemail_result("10", None, max_length=max_length)
        for key in (None, True, "", "12", "a", "hangup"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_voicemail_result("10", None, finish_on_key=key)


class VoicemailEventTests(unittest.TestCase):
    def event(self, version="2.0", encoded=False, **overrides):
        fields = {
            "AccountSid": "synthetic-account",
            "CallSid": "synthetic-call",
            "RecordingDuration": "12",
            "Digits": "#",
            "RecordingUrl": "https://api.example.invalid/private-recording",
            **overrides,
        }
        body = urlencode({key: value for key, value in fields.items() if value is not None})
        event = {
            "version": version,
            "body": body,
            "isBase64Encoded": encoded,
            "headers": {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": "A" * 27 + "=",
            },
        }
        if encoded:
            event["body"] = base64.b64encode(body.encode()).decode()
        if version == "1.0":
            event["httpMethod"] = "POST"
        else:
            event["requestContext"] = {"http": {"method": "POST"}}
        return event

    def resolve(self, event, validator=None, **overrides):
        config = {
            "public_url": "https://example.com/voice/voicemail-result",
            "validator": validator or Mock(return_value=True),
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": "synthetic-call",
            "max_length": 120,
            "finish_on_key": "#",
        }
        config.update(overrides)
        return resolve_voicemail_result_event(event, **config)

    def test_authenticated_result_works_in_both_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    event = self.event(
                        version, encoded, max_length="600", finish_on_key="*",
                    )
                    original = copy.deepcopy(event)
                    result = self.resolve(event)
                    self.assertEqual(
                        result, VoicemailDecision("continue", 12, "finish-key"),
                    )
                    self.assertEqual(event, original)
                    self.assertNotIn("private-recording", repr(result))

    def test_callback_cannot_replace_trusted_policy(self):
        event = self.event(
            RecordingDuration="121", max_length="600", finish_on_key="*",
        )
        with self.assertRaisesRegex(ValueError, "exceeds the configured maximum"):
            self.resolve(event)
        event = self.event(Digits="*", finish_on_key="*")
        with self.assertRaisesRegex(ValueError, "unsupported voicemail completion input"):
            self.resolve(event)

    def test_invalid_trusted_policy_fails_before_authentication(self):
        for overrides in ({"max_length": 601}, {"finish_on_key": "x"}):
            validator = Mock(return_value=True)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(self.event(), validator, **overrides)
            validator.assert_not_called()

    def test_rejected_transport_signature_or_identity_never_reaches_policy(self):
        cases = (
            (self.event(AccountSid="other"), Mock(return_value=True)),
            (self.event(CallSid="other"), Mock(return_value=True)),
            (self.event(), Mock(return_value=False)),
        )
        invalid_method = self.event()
        invalid_method["requestContext"]["http"]["method"] = "GET"
        cases += ((invalid_method, Mock(return_value=True)),)
        for event, validator in cases:
            with self.subTest(body=event["body"]), patch(
                "zentomic.voicemail_event._resolve_validated_voicemail_result",
            ) as policy, self.assertRaises(ValueError):
                self.resolve(event, validator)
            policy.assert_not_called()

    def test_validator_cannot_rewrite_duration_or_completion(self):
        def mutate(url, fields, signature):
            fields.update(RecordingDuration="1", Digits="hangup")
            return True

        self.assertEqual(
            self.resolve(self.event(), mutate),
            VoicemailDecision("continue", 12, "finish-key"),
        )


class ClaimedVoicemailEventTests(unittest.TestCase):
    event = VoicemailEventTests.event

    def resolve(self, event, claimer, validator=None, **overrides):
        config = {
            "public_url": "https://example.com/voice/voicemail-result",
            "validator": validator or Mock(return_value=True),
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": "synthetic-call",
            "workspace_id": "synthetic-workspace",
            "step_id": "voicemail-1",
            "claimer": claimer,
            "max_length": 120,
            "finish_on_key": "#",
        }
        config.update(overrides)
        return resolve_claimed_voicemail_result_event(event, **config)

    def test_authenticated_result_is_claimed_once(self):
        store = InMemoryCallbackClaimStore()
        event = self.event(workspace_id="untrusted", step_id="untrusted")
        self.assertEqual(
            self.resolve(event, store),
            VoicemailDecision("continue", 12, "finish-key"),
        )
        with self.assertRaisesRegex(CallbackReplayError, "callback step already claimed"):
            self.resolve(event, store)
        self.assertTrue(store.claim("untrusted", "synthetic-call", "untrusted"))

    def test_rejected_or_malformed_callback_does_not_consume_step(self):
        cases = (
            (self.event(RecordingDuration="121"), Mock(return_value=True)),
            (self.event(Digits="*"), Mock(return_value=True)),
            (self.event(CallSid="other"), Mock(return_value=True)),
            (self.event(), Mock(return_value=False)),
        )
        for event, validator in cases:
            store = InMemoryCallbackClaimStore()
            with self.subTest(body=event["body"]), self.assertRaises(ValueError):
                self.resolve(event, store, validator)
            self.assertTrue(store.claim(
                "synthetic-workspace", "synthetic-call", "voicemail-1",
            ))

    def test_invalid_claim_configuration_does_not_authenticate_or_claim(self):
        for overrides in (
            {"workspace_id": ""}, {"step_id": " "}, {"claimer": object()},
            {"max_length": 601}, {"finish_on_key": "x"},
        ):
            validator = Mock(return_value=True)
            claimer = Mock()
            call_overrides = dict(overrides)
            supplied_claimer = call_overrides.pop("claimer", claimer)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(self.event(), supplied_claimer, validator, **call_overrides)
            validator.assert_not_called()
            claimer.claim.assert_not_called()

    def test_claim_finishes_before_decision_is_returned(self):
        claimer = Mock()
        claimer.claim.return_value = True
        result = self.resolve(self.event(), claimer)
        self.assertEqual(result.duration_seconds, 12)
        claimer.claim.assert_called_once_with(
            "synthetic-workspace", "synthetic-call", "voicemail-1",
        )


if __name__ == "__main__":
    unittest.main()
