"""Offline checks for the standard-library Twilio signature validator."""

import base64
import functools
import hashlib
import hmac
import unittest
from unittest.mock import patch

from zentomic import twilio_signature as twilio_signature_module
from zentomic.authentication import MAX_PUBLIC_URL_BYTES, validate_form_event
from zentomic.twilio_signature import (
    MAX_AUTH_TOKEN_BYTES,
    MAX_AUTH_TOKENS,
    MAX_SIGNED_PAYLOAD_BYTES,
    create_twilio_signature_validator,
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


def _signature(url, fields):
    signed = url + "".join(
        name + value for name, value in sorted(fields.items())
    )
    return base64.b64encode(
        hmac.new(TOKEN.encode(), signed.encode(), hashlib.sha1).digest()
    ).decode()


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

    def test_malformed_base64_signatures_return_false(self):
        for signature in ("é" * 28, "!" * 27 + "=", "A" * 28):
            with self.subTest(signature=ascii(signature)):
                self.assertFalse(self.validate(signature=signature))

    def test_rejects_payload_over_limit(self):
        fields = {"Body": "x" * (MAX_SIGNED_PAYLOAD_BYTES + 1)}
        self.assertFalse(self.validate(
            fields=fields, signature=_signature(URL, fields),
        ))

    def test_rejects_public_url_over_utf8_byte_limit(self):
        url = "https://example.com/" + "é" * (MAX_PUBLIC_URL_BYTES // 2)
        self.assertLessEqual(len(url), MAX_PUBLIC_URL_BYTES)
        self.assertGreater(len(url.encode()), MAX_PUBLIC_URL_BYTES)
        self.assertFalse(self.validate(url=url, fields={}, signature=_signature(url, {})))

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
        validators = (
            functools.partial(validate_twilio_signature, auth_token=TOKEN),
            create_twilio_signature_validator(TOKEN),
        )
        for validator in validators:
            with self.subTest(factory=validator.__class__.__name__):
                self.assertEqual(
                    validate_form_event(event, public_url=URL, validator=validator),
                    FIELDS,
                )

    def test_factory_validates_configuration_once(self):
        with patch(
            "zentomic.twilio_signature._auth_token_bytes",
            wraps=twilio_signature_module._auth_token_bytes,
        ) as encode_token:
            validator = create_twilio_signature_validator(TOKEN)
            self.assertTrue(validator(URL, FIELDS, _signature(URL, FIELDS)))
            self.assertTrue(validator(URL, FIELDS, _signature(URL, FIELDS)))
        encode_token.assert_called_once_with(TOKEN)

    def test_factory_supports_bounded_token_rotation(self):
        previous = "previous-token"
        current_signature = _signature(URL, FIELDS)
        previous_signature = base64.b64encode(
            hmac.new(
                previous.encode(),
                (URL + "".join(name + value for name, value in sorted(FIELDS.items()))).encode(),
                hashlib.sha1,
            ).digest()
        ).decode()
        validator = create_twilio_signature_validator(
            TOKEN, previous_auth_tokens=(previous,),
        )

        self.assertTrue(validator(URL, FIELDS, current_signature))
        self.assertTrue(validator(URL, FIELDS, previous_signature))
        self.assertFalse(validator(URL, FIELDS, "A" * 27 + "="))

    def test_factory_checks_every_rotation_token(self):
        validator = create_twilio_signature_validator(
            TOKEN, previous_auth_tokens=("previous-one", "previous-two"),
        )
        with patch(
            "zentomic.twilio_signature._validate_twilio_signature",
            return_value=True,
        ) as validate:
            self.assertTrue(validator(URL, FIELDS, SIGNATURE))
        self.assertEqual(validate.call_count, MAX_AUTH_TOKENS)

    def test_factory_rejects_invalid_rotation_configuration(self):
        class UnderreportedTokens(list):
            def __len__(self):
                return 0

        cases = (
            None,
            TOKEN,
            ("",),
            ("previous", None),
            tuple(f"token-{index}" for index in range(MAX_AUTH_TOKENS)),
            UnderreportedTokens(
                f"underreported-{index}" for index in range(MAX_AUTH_TOKENS)
            ),
        )
        for previous in cases:
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                create_twilio_signature_validator(
                    TOKEN, previous_auth_tokens=previous,
                )

    def test_factory_bounds_rotation_collection_before_encoding_secrets(self):
        class UnderreportedTokens(list):
            def __len__(self):
                return 0

        previous = UnderreportedTokens(
            f"underreported-{index}" for index in range(MAX_AUTH_TOKENS + 10)
        )
        with patch(
            "zentomic.twilio_signature._auth_token_bytes",
            side_effect=AssertionError("oversized configuration must not be encoded"),
        ) as encode_token, self.assertRaisesRegex(
            ValueError, "^previous_auth_tokens must contain at most 2 tokens$",
        ):
            create_twilio_signature_validator(
                TOKEN, previous_auth_tokens=previous,
            )
        encode_token.assert_not_called()

    def test_factory_rejects_duplicate_rotation_tokens_privately(self):
        cases = (
            (TOKEN,),
            ("retiring-secret", "retiring-secret"),
        )
        for previous in cases:
            with self.subTest(previous_count=len(previous)), self.assertRaises(
                ValueError,
            ) as caught:
                create_twilio_signature_validator(
                    TOKEN, previous_auth_tokens=previous,
                )
            self.assertEqual(
                str(caught.exception),
                "auth_token and previous_auth_tokens must be unique",
            )
            for token in (TOKEN, *previous):
                self.assertNotIn(token, str(caught.exception))

    def test_factory_rejects_invalid_configuration_before_requests(self):
        for token in (None, "", "\ud800", "x" * (MAX_AUTH_TOKEN_BYTES + 1)):
            with self.subTest(token=repr(token)[:20]), self.assertRaises(ValueError):
                create_twilio_signature_validator(token)

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

    def test_failing_mapping_returns_false(self):
        class FailingItems(dict):
            def items(self):
                raise RuntimeError("synthetic-private-marker")

        self.assertFalse(self.validate(fields=FailingItems(FIELDS)))

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
