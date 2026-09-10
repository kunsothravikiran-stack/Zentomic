"""Offline binding of authenticated callbacks to synthetic call sessions."""

import base64
import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.authentication import validate_call_event
from zentomic.gather import resolve_gather


URL = "https://example.invalid/voice/menu"
ACCOUNT = "AC" + "0" * 32
CALL = "CA" + "0" * 32
SIGNATURE = "A" * 27 + "="  # Shape-only fixture, not a real signature.


def event(fields=None, version="1.0", encoded=False):
    body = urlencode(fields if fields is not None else {
        "AccountSid": ACCOUNT, "CallSid": CALL, "Digits": "1", "Optional": "",
    })
    result = {
        "version": version,
        "headers": {"Content-Type": "application/x-www-form-urlencoded",
                    "X-Twilio-Signature": SIGNATURE},
        "body": base64.b64encode(body.encode()).decode() if encoded else body,
        "isBase64Encoded": encoded,
    }
    if version == "2.0":
        result["requestContext"] = {"http": {"method": "POST"}}
    else:
        result["httpMethod"] = "POST"
    return result


def validate(candidate, validator, **overrides):
    options = {"public_url": URL, "validator": validator,
               "expected_account_sid": ACCOUNT, "expected_call_sid": CALL}
    return validate_call_event(candidate, **{**options, **overrides})


class CallAuthenticationTests(unittest.TestCase):
    def test_matching_session_preserves_all_authenticated_fields(self):
        expected = {"AccountSid": ACCOUNT, "CallSid": CALL, "Digits": "1", "Optional": ""}
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    validator = Mock(return_value=True)
                    self.assertEqual(validate(event(version=version, encoded=encoded), validator), expected)
                    validator.assert_called_once_with(URL, expected, SIGNATURE)

    def test_missing_blank_and_mismatched_identifiers_fail_after_authentication(self):
        for field in ("AccountSid", "CallSid"):
            for value in (None, "", "different-session", " " + (ACCOUNT if field == "AccountSid" else CALL)):
                fields = {"AccountSid": ACCOUNT, "CallSid": CALL, "Digits": "1"}
                if value is None:
                    del fields[field]
                else:
                    fields[field] = value
                validator = Mock(return_value=True)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError) as caught:
                    validate(event(fields), validator)
                self.assertEqual(str(caught.exception), "webhook does not match the expected call session")
                validator.assert_called_once_with(URL, fields, SIGNATURE)

    def test_field_names_are_case_sensitive(self):
        for field in ("AccountSid", "CallSid"):
            fields = {"AccountSid": ACCOUNT, "CallSid": CALL}
            fields[field.lower()] = fields.pop(field)
            with self.assertRaises(ValueError):
                validate(event(fields), Mock(return_value=True))

    def test_invalid_trusted_configuration_never_invokes_validator(self):
        for key in ("expected_account_sid", "expected_call_sid"):
            for value in (None, False, 1, [], {}, "", " \t\n"):
                validator = Mock(return_value=True)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate(event(), validator, **{key: value})
                validator.assert_not_called()

    def test_matching_identifiers_cannot_bypass_signature_or_transport_gate(self):
        for validator in (Mock(return_value=False), Mock(return_value=1),
                          Mock(side_effect=RuntimeError("synthetic-private-marker"))):
            with self.assertRaisesRegex(ValueError, "webhook signature validation failed"):
                validate(event(), validator)
        candidate = event()
        candidate["httpMethod"] = "GET"
        validator = Mock(return_value=True)
        with self.assertRaises(ValueError):
            validate(candidate, validator)
        validator.assert_not_called()

    def test_injected_validator_cannot_rewrite_session_identifiers(self):
        candidate = event({"AccountSid": ACCOUNT, "CallSid": "another-call"})
        original = copy.deepcopy(candidate)

        def validator(url, fields, signature):
            fields["CallSid"] = CALL
            return True

        with self.assertRaisesRegex(ValueError, "expected call session"):
            validate(candidate, validator)
        self.assertEqual(candidate, original)

    def test_rejected_session_never_reaches_routing(self):
        with patch("zentomic.gather.resolve_gather") as resolver:
            with self.assertRaises(ValueError):
                fields = validate(event(), Mock(return_value=True), expected_call_sid="another-call")
                resolver(fields.get("Digits"), {"1": "sales"},
                         fallback_target="reception", attempts=0)
            resolver.assert_not_called()
        fields = validate(event(), Mock(return_value=True))
        decision = resolve_gather(fields.get("Digits"), {"1": "sales"},
                                  fallback_target="reception", attempts=0)
        self.assertEqual((decision.action, decision.target, decision.attempts), ("route", "sales", 1))

    def test_no_network_environment_or_output(self):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.dict("os.environ", {}, clear=True), \
                patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            validate(event(), Mock(return_value=True))
            with self.assertRaises(ValueError) as caught:
                validate(event(), Mock(return_value=True), expected_call_sid="synthetic-private-marker")
        self.assertNotIn("synthetic-private-marker", str(caught.exception))
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
