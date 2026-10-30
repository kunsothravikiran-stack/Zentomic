"""Offline checks for the standard-library Twilio signature validator."""

import functools
import unittest
from unittest.mock import patch

from zentomic.authentication import validate_form_event
from zentomic.twilio_signature import (
    MAX_AUTH_TOKEN_BYTES,
    validate_twilio_signature,
)
from zentomic.webhook import MAX_FORM_FIELDS


URL = "https://example.com/myapp.php?foo=1&bar=2"
TOKEN = "12345"
FIELDS = {
    "CallSid": "CA1234567890ABCDE",
    "Caller": "+14158675310",
    "Digits": "1234",
    "From": "+14158675310",
    "To": "+18005551212",
}
SIGNATURE = "L/OH5YylLD5NRKLltdqwSvS0BnU="


class TwilioSignatureTests(unittest.TestCase):
    def validate(self, url=URL, fields=FIELDS, signature=SIGNATURE, token=TOKEN):
        return validate_twilio_signature(
            url, fields, signature, auth_token=token,
        )

    def test_matches_twilio_documented_form_signature_vector(self):
        self.assertTrue(self.validate())
        self.assertTrue(self.validate(fields=dict(reversed(tuple(FIELDS.items())))))

    def test_url_field_signature_and_token_changes_fail(self):
        changes = (
            {"url": URL + "&extra=1"},
            {"fields": {**FIELDS, "Digits": "1235"}},
            {"signature": "A" * 27 + "="},
            {"token": TOKEN + "0"},
        )
        for change in changes:
            with self.subTest(change=next(iter(change))):
                self.assertFalse(self.validate(**change))

    def test_integrates_with_authenticated_form_event_gate(self):
        body = "&".join(f"{name}={value.replace('+', '%2B')}" for name, value in FIELDS.items())
        event = {
            "version": "1.0",
            "httpMethod": "POST",
            "headers": {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": SIGNATURE,
            },
            "body": body,
        }
        validator = functools.partial(validate_twilio_signature, auth_token=TOKEN)
        self.assertEqual(
            validate_form_event(event, public_url=URL, validator=validator),
            FIELDS,
        )

    def test_invalid_request_shapes_return_false(self):
        cases = (
            {"url": None}, {"fields": None}, {"fields": {"A": 1}},
            {"fields": {str(index): "" for index in range(MAX_FORM_FIELDS + 1)}},
            {"signature": None}, {"signature": "short"},
            {"url": "https://example.invalid/\ud800"},
            {"fields": {"A": "\ud800"}},
        )
        for change in cases:
            with self.subTest(change=repr(change)[:80]):
                self.assertFalse(self.validate(**change))

    def test_invalid_trusted_token_configuration_raises_privately(self):
        for token in (None, "", "\ud800", "x" * (MAX_AUTH_TOKEN_BYTES + 1),
                      "é" * (MAX_AUTH_TOKEN_BYTES // 2 + 1)):
            with self.subTest(token=repr(token)[:20]), self.assertRaises(ValueError) as caught:
                self.validate(token=token)
            self.assertNotIn(repr(token), str(caught.exception))

    def test_validation_is_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertTrue(self.validate())


if __name__ == "__main__":
    unittest.main()
