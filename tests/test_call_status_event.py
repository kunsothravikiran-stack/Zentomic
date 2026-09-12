"""Reusable authenticated lifecycle boundary with synthetic callbacks only."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.call_status_event import advance_call_status_event


class CallStatusEventTests(unittest.TestCase):
    def event(self, version="2.0", encoded=False, **fields):
        body = urlencode({"AccountSid": "synthetic-account", "CallSid": "synthetic-leg",
                          "CallStatus": "completed", **fields})
        event = {"version": version, "body": body, "isBase64Encoded": encoded,
                 "headers": {"Content-Type": "application/x-www-form-urlencoded",
                             "X-Twilio-Signature": "A" * 27 + "="}}
        if encoded:
            event["body"] = base64.b64encode(body.encode()).decode()
        if version == "1.0":
            event["httpMethod"] = "POST"
        else:
            event["requestContext"] = {"http": {"method": "POST"}}
        return event

    def advance(self, current, event, validator):
        return advance_call_status_event(
            current, event, public_url="https://example.com/voice/status",
            validator=validator, expected_account_sid="synthetic-account",
            expected_call_sid="synthetic-leg",
        )

    def test_replays_authenticated_observations_in_both_transport_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                current = None
                validator = Mock(return_value=True)
                for incoming, expected in (("ringing", "ringing"), ("queued", "ringing"),
                                           ("in-progress", "in-progress"),
                                           ("completed", "completed"), ("failed", "completed")):
                    with self.subTest(version=version, encoded=encoded, incoming=incoming):
                        event = self.event(version, encoded, CallStatus=incoming,
                                           DialCallStatus="busy", SpeechResult="synthetic-text")
                        original = copy.deepcopy(event)
                        current = self.advance(current, event, validator)
                        self.assertEqual(current, expected)
                        self.assertEqual(event, original)
                self.assertEqual(validator.call_count, 5)

    def test_invalid_stored_state_is_rejected_before_authentication(self):
        for current in ("unknown", "", [], True):
            validator = Mock(return_value=True)
            with self.subTest(current=current), self.assertRaises(ValueError):
                self.advance(current, self.event(), validator)
            validator.assert_not_called()

    def test_signature_and_identity_failures_never_reach_transition_policy(self):
        cases = (({}, Mock(return_value=False)), ({}, Mock(side_effect=RuntimeError)),
                 ({"AccountSid": "other-account"}, Mock(return_value=True)),
                 ({"CallSid": "other-leg"}, Mock(return_value=True)))
        for fields, validator in cases:
            with self.subTest(fields=fields), patch(
                "zentomic.call_status_event.advance_call_status",
            ) as policy:
                with self.assertRaises(ValueError):
                    self.advance("completed", self.event(**fields), validator)
                policy.assert_not_called()
                validator.assert_called_once()

    def test_terminal_state_does_not_skip_status_validation(self):
        for status in ("", "unknown", "COMPLETED"):
            validator = Mock(return_value=True)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "unsupported call status"):
                self.advance("completed", self.event(CallStatus=status), validator)
            validator.assert_called_once()

    def test_dial_status_cannot_substitute_for_missing_call_status(self):
        event = self.event()
        event["body"] = urlencode({"AccountSid": "synthetic-account", "CallSid": "synthetic-leg",
                                   "DialCallStatus": "completed"})
        validator = Mock(return_value=True)
        with self.assertRaisesRegex(ValueError, "unsupported call status"):
            self.advance(None, event, validator)
        validator.assert_called_once()

    def test_invalid_transport_is_rejected_before_validator(self):
        event = self.event()
        event["requestContext"]["http"]["method"] = "GET"
        validator = Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.advance(None, event, validator)
        validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
