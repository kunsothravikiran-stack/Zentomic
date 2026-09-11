"""Authenticated lifecycle composition, without a provider or persistence SDK."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.authentication import validate_call_event
from zentomic.call_status import advance_call_status


class CallStatusFlowTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def event(self, version, encoded, **overrides):
        fields = {"AccountSid": "synthetic-account", "CallSid": "synthetic-parent",
                  "CallStatus": "completed", "DialCallStatus": "failed"}
        fields.update(overrides)
        fields = {key: value for key, value in fields.items() if value is not None}
        body = urlencode(fields)
        if encoded:
            body = base64.b64encode(body.encode()).decode()
        event = {
            "version": version, "body": body, "isBase64Encoded": encoded,
            "headers": {"Content-Type": "application/x-www-form-urlencoded",
                        "X-Twilio-Signature": "A" * 27 + "="},
        }
        if version == "1.0":
            event["httpMethod"] = "POST"
        else:
            event["requestContext"] = {"http": {"method": "POST"}}
        return event

    def advance(self, current, event, validator):
        # Example ordering only. This local assignment is not atomic storage,
        # replay prevention, cleanup authorization, or a production endpoint.
        fields = validate_call_event(
            event, public_url="https://example.com/voice/status",
            validator=validator, expected_account_sid="synthetic-account",
            expected_call_sid="synthetic-parent",
        )
        return advance_call_status(current, fields.get("CallStatus"))

    def test_authenticated_delays_and_replays_preserve_terminal_outcome(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    current = None
                    history = []
                    validator = Mock(return_value=True)
                    for status in ("ringing", "queued", "in-progress", "completed",
                                   "ringing", "completed", "failed"):
                        event = self.event(version, encoded, CallStatus=status)
                        original = copy.deepcopy(event)
                        current = self.advance(current, event, validator)
                        history.append(current)
                        self.assertEqual(event, original)
                    self.assertEqual(history, ["ringing", "ringing", "in-progress"]
                                     + ["completed"] * 4)
                    self.assertEqual(validator.call_count, 7)

    def test_rejected_callbacks_cannot_change_existing_state(self):
        cases = (
            ({"CallSid": "synthetic-child"}, True),
            ({"AccountSid": "synthetic-other-account"}, True),
            ({"CallStatus": None}, True),
            ({"CallStatus": "future-status"}, True),
            ({}, False),
        )
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for starting in (None, "in-progress", "completed"):
                    for overrides, valid in cases:
                        with self.subTest(version=version, encoded=encoded,
                                          starting=starting, overrides=overrides, valid=valid):
                            current = starting
                            event = self.event(version, encoded, **overrides)
                            original = copy.deepcopy(event)
                            validator = Mock(return_value=valid)
                            with self.assertRaises(ValueError):
                                current = self.advance(current, event, validator)
                            self.assertEqual(current, starting)
                            self.assertEqual(event, original)
                            validator.assert_called_once()

    def test_authentication_failure_precedes_status_policy(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for validator in (Mock(return_value=False), Mock(side_effect=RuntimeError)):
                    with self.subTest(version=version, encoded=encoded), patch(
                        __name__ + ".advance_call_status",
                    ) as policy:
                        with self.assertRaisesRegex(ValueError, "signature validation failed"):
                            self.advance("in-progress", self.event(version, encoded), validator)
                        policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
