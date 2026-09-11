"""Offline call lifecycle classification with synthetic callback fields."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.authentication import validate_call_event
from zentomic.call_status import is_terminal_call_status


class CallStatusTests(unittest.TestCase):
    def test_all_terminal_outcomes_are_terminal_not_only_completed(self):
        for status in ("completed", "busy", "failed", "no-answer", "canceled"):
            with self.subTest(status=status):
                self.assertIs(is_terminal_call_status(status), True)

    def test_known_nonterminal_statuses_are_not_terminal(self):
        for status in ("queued", "ringing", "in-progress"):
            with self.subTest(status=status):
                self.assertIs(is_terminal_call_status(status), False)

    def test_unknown_missing_and_wrong_type_values_raise_generic_errors(self):
        for status in (None, "", "future-private-status", "initiated", "answered",
                       "cancelled", "absent", True, 0, 1, [], {}, b"completed"):
            with self.subTest(status=status), self.assertRaises(ValueError) as caught:
                is_terminal_call_status(status)
            self.assertEqual(str(caught.exception), "unsupported call status")

    def test_status_is_not_trimmed_case_folded_or_coerced(self):
        for status in ("COMPLETED", " completed", "completed\n", "in_progress",
                       "ｃｏｍｐｌｅｔｅｄ", "completed\x00", "\ud800"):
            with self.subTest(status=repr(status)), self.assertRaises(ValueError):
                is_terminal_call_status(status)

    def test_authenticated_composition_uses_call_status_not_dial_status(self):
        fields = {"AccountSid": "synthetic-account", "CallSid": "synthetic-call",
                  "CallStatus": "in-progress", "DialCallStatus": "completed"}
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
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
                    original = copy.deepcopy(event)
                    validator = Mock(return_value=True)
                    authenticated = validate_call_event(
                        event, public_url="https://example.com/voice/status",
                        validator=validator, expected_account_sid="synthetic-account",
                        expected_call_sid="synthetic-call",
                    )
                    self.assertIs(is_terminal_call_status(authenticated["CallStatus"]), False)
                    validator.assert_called_once()
                    self.assertEqual(event, original)
                    self.assertEqual(authenticated, fields)

    def test_classification_does_not_remember_or_order_previous_callbacks(self):
        # Persistence must not reopen a terminal call on a delayed ringing event.
        self.assertEqual([is_terminal_call_status(s) for s in
                          ("completed", "ringing", "completed")], [True, False, True])

    def test_policy_is_offline_and_silent(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("builtins.print") as output:
            self.assertIs(is_terminal_call_status("completed"), True)
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
