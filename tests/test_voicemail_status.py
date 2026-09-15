"""Offline voicemail recording-status admission and callback boundary tests."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.callback_claim_store import (
    CallbackReplayError,
    InMemoryCallbackClaimStore,
)
from zentomic.voicemail_status import (
    VoicemailRecordingStatus,
    resolve_voicemail_recording_status,
)
from zentomic.voicemail_status_event import (
    resolve_claimed_voicemail_recording_status_event,
    resolve_voicemail_recording_status_event,
)


RECORDING_SID = "RE" + "a1" * 16


class VoicemailRecordingStatusPolicyTests(unittest.TestCase):
    def resolve(self, **overrides):
        fields = {
            "recording_sid": RECORDING_SID,
            "recording_status": "completed",
            "recording_duration": "12",
            "recording_channels": "1",
            "recording_source": "RecordVerb",
            "max_length": 120,
        }
        fields.update(overrides)
        return resolve_voicemail_recording_status(**fields)

    def test_completed_recording_is_available_with_final_duration(self):
        self.assertEqual(
            self.resolve(),
            VoicemailRecordingStatus("available", RECORDING_SID, 12),
        )

    def test_failed_recording_is_unavailable_without_usable_duration(self):
        self.assertEqual(
            self.resolve(recording_status="failed", recording_duration="0"),
            VoicemailRecordingStatus("unavailable", RECORDING_SID, None),
        )

    def test_only_documented_bounded_record_fields_are_admitted(self):
        cases = (
            {"recording_sid": "CA" + "1" * 32},
            {"recording_sid": "RE" + "g" * 32},
            {"recording_status": "absent"},
            {"recording_status": "in-progress"},
            {"recording_duration": None},
            {"recording_duration": "01"},
            {"recording_duration": "121"},
            {"recording_channels": "2"},
            {"recording_source": "DialVerb"},
            {"max_length": 601},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(**overrides)


class VoicemailRecordingStatusEventTests(unittest.TestCase):
    def event(self, version="2.0", encoded=False, **overrides):
        fields = {
            "AccountSid": "synthetic-account",
            "CallSid": "synthetic-call",
            "RecordingSid": RECORDING_SID,
            "RecordingStatus": "completed",
            "RecordingDuration": "12",
            "RecordingChannels": "1",
            "RecordingSource": "RecordVerb",
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
            "public_url": "https://example.com/voice/voicemail-status",
            "validator": validator or Mock(return_value=True),
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": "synthetic-call",
            "max_length": 120,
        }
        config.update(overrides)
        return resolve_voicemail_recording_status_event(event, **config)

    def test_authenticated_status_works_in_both_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    event = self.event(version, encoded, max_length="600")
                    original = copy.deepcopy(event)
                    result = self.resolve(event)
                    self.assertEqual(result.duration_seconds, 12)
                    self.assertEqual(event, original)
                    self.assertNotIn("private-recording", repr(result))

    def test_rejected_callback_never_reaches_recording_policy(self):
        cases = (
            (self.event(AccountSid="other"), Mock(return_value=True)),
            (self.event(CallSid="other"), Mock(return_value=True)),
            (self.event(), Mock(return_value=False)),
        )
        for event, validator in cases:
            with self.subTest(body=event["body"]), patch(
                "zentomic.voicemail_status_event._resolve_fields",
            ) as policy, self.assertRaises(ValueError):
                self.resolve(event, validator)
            policy.assert_not_called()

    def test_trusted_duration_bound_is_not_callback_configurable(self):
        event = self.event(RecordingDuration="121", max_length="600")
        with self.assertRaisesRegex(ValueError, "exceeds the configured maximum"):
            self.resolve(event)


class ClaimedVoicemailRecordingStatusEventTests(unittest.TestCase):
    event = VoicemailRecordingStatusEventTests.event

    def resolve(self, event, claimer, validator=None, **overrides):
        config = {
            "public_url": "https://example.com/voice/voicemail-status",
            "validator": validator or Mock(return_value=True),
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": "synthetic-call",
            "workspace_id": "synthetic-workspace",
            "status_step_id": "voicemail-status-1",
            "claimer": claimer,
            "max_length": 120,
        }
        config.update(overrides)
        return resolve_claimed_voicemail_recording_status_event(event, **config)

    def test_authenticated_status_is_claimed_once(self):
        store = InMemoryCallbackClaimStore()
        event = self.event(status_step_id="untrusted")
        self.assertEqual(self.resolve(event, store).availability, "available")
        with self.assertRaisesRegex(CallbackReplayError, "callback step already claimed"):
            self.resolve(event, store)
        self.assertTrue(store.claim(
            "synthetic-workspace", "synthetic-call", "untrusted",
        ))

    def test_malformed_callback_does_not_consume_status_step(self):
        store = InMemoryCallbackClaimStore()
        with self.assertRaises(ValueError):
            self.resolve(self.event(RecordingChannels="2"), store)
        self.assertTrue(store.claim(
            "synthetic-workspace", "synthetic-call", "voicemail-status-1",
        ))

    def test_invalid_claim_configuration_fails_before_authentication(self):
        validator = Mock(return_value=True)
        claimer = Mock()
        with self.assertRaises(ValueError):
            self.resolve(self.event(), claimer, validator, status_step_id="")
        validator.assert_not_called()
        claimer.claim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
