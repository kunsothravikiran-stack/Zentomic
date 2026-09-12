"""Synthetic signature-gate tests; no SDK, credentials, or network required."""

import base64
import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from zentomic.authentication import validate_form_event


URL = "https://example.invalid/voice/menu?first=%2F&second=2"
SIGNATURE = "A" * 27 + "="  # Shape-only fixture, not a valid Twilio signature.


def event(version="1.0"):
    result = {
        "version": version,
        "headers": {"Content-Type": "application/x-www-form-urlencoded",
                    "X-Twilio-Signature": SIGNATURE},
        "body": "Digits=1&Unknown=+value+&Empty=&Unicode=%C3%A9",
    }
    if version == "2.0":
        result["requestContext"] = {"http": {"method": "POST"}}
    else:
        result["httpMethod"] = "POST"
    return result


class AuthenticationTests(unittest.TestCase):
    def test_all_fields_and_exact_url_reach_validator(self):
        expected = {"Digits": "1", "Unknown": " value ", "Empty": "", "Unicode": "é"}
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                candidate = event(version)
                candidate["isBase64Encoded"] = encoded
                if encoded:
                    candidate["body"] = base64.b64encode(candidate["body"].encode()).decode()
                validator = Mock(return_value=True)
                with self.subTest(version=version, encoded=encoded):
                    self.assertEqual(validate_form_event(candidate, public_url=URL, validator=validator), expected)
                    validator.assert_called_once_with(URL, expected, SIGNATURE)

    def test_v1_signature_mirror_multivalue_only_and_omitted_version(self):
        for mirrored in (False, True):
            candidate = event()
            del candidate["version"]
            candidate["multiValueHeaders"] = {"x-twilio-signature": [SIGNATURE]}
            if not mirrored:
                del candidate["headers"]["X-Twilio-Signature"]
            validator = Mock(return_value=True)
            self.assertEqual(validate_form_event(candidate, public_url=URL, validator=validator)["Digits"], "1")

    def test_header_names_are_case_insensitive(self):
        candidate = event()
        candidate["headers"]["x-TwIlIo-SiGnAtUrE"] = candidate["headers"].pop("X-Twilio-Signature")
        validate_form_event(candidate, public_url=URL, validator=Mock(return_value=True))

    def test_invalid_or_missing_signature_never_reaches_validator(self):
        for value in (None, [], True, "", "synthetic", "é" * 28, SIGNATURE + " ",
                      SIGNATURE + "," + SIGNATURE, "A" * 10000):
            candidate = event()
            candidate["headers"]["X-Twilio-Signature"] = value
            self.assert_rejected_before_validator(candidate)
        candidate = event()
        del candidate["headers"]["X-Twilio-Signature"]
        self.assert_rejected_before_validator(candidate)

    def test_duplicate_conflicting_and_v2_multivalue_signatures_fail(self):
        candidate = event()
        candidate["headers"]["x-twilio-signature"] = SIGNATURE
        self.assert_rejected_before_validator(candidate)
        for value in ([], SIGNATURE, [SIGNATURE, SIGNATURE], ["B" * 27 + "="]):
            candidate = event()
            candidate["multiValueHeaders"] = {"X-Twilio-Signature": value}
            self.assert_rejected_before_validator(candidate)
        candidate = event("2.0")
        candidate["multiValueHeaders"] = {"X-Twilio-Signature": [SIGNATURE]}
        self.assert_rejected_before_validator(candidate)

    def assert_rejected_before_validator(self, candidate, public_url=URL):
        validator = Mock(return_value=True)
        with self.assertRaises(ValueError):
            validate_form_event(candidate, public_url=public_url, validator=validator)
        validator.assert_not_called()

    def test_transport_rejected_before_signature_validation(self):
        for overrides in ({"httpMethod": "GET"}, {"body": "Digits=1&Digits=2"},
                          {"headers": []}, {"multiValueHeaders": []}, {"body": None}):
            self.assert_rejected_before_validator({**event(), **overrides})

    def test_only_explicit_true_is_accepted(self):
        for result in (False, None, 1, "true", {"valid": True}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_form_event(event(), public_url=URL, validator=Mock(return_value=result))

    def test_configuration_validation(self):
        for url in (None, [], "", "/voice", "http://example.invalid/voice",
                    "https:///voice", "https://user:pass@example.invalid/voice",
                    URL + "#fragment", URL + "#", "https://example.invalid:bad/voice",
                    "https://example.invalid:99999/voice", "https://[invalid/voice",
                    " https://example.invalid/voice", URL + "\n", URL + "\x00"):
            with self.subTest(url=url):
                self.assert_rejected_before_validator(event(), public_url=url)
        with self.assertRaises(ValueError):
            validate_form_event(event(), public_url=URL, validator=None)

    def test_request_host_headers_do_not_select_validation_url(self):
        candidate = event()
        candidate["headers"].update({"Host": "untrusted.invalid", "X-Forwarded-Proto": "http"})
        validator = Mock(return_value=True)
        validate_form_event(candidate, public_url=URL, validator=validator)
        self.assertEqual(validator.call_args.args[0], URL)

    def test_non_utf8_callback_urls_are_rejected_before_validator(self):
        for character in ("\ud800", "\udfff"):
            for url in (f"https://exam{character}ple.invalid/voice",
                        f"https://example.invalid/{character}",
                        f"https://example.invalid/voice?value={character}"):
                with self.subTest(character=ascii(character), url=ascii(url)):
                    validator = Mock(return_value=True)
                    with self.assertRaises(ValueError) as caught:
                        validate_form_event(event(), public_url=url, validator=validator)
                    validator.assert_not_called()
                    self.assertEqual(str(caught.exception),
                                     "public_url must be a configured HTTPS callback URL")

    def test_valid_unicode_callback_urls_reach_validator_unchanged(self):
        for url in ("https://example.invalid/café?value=\U0001f600",
                    "https://example.invalid/caf%C3%A9?value=%F0%9F%98%80"):
            with self.subTest(url=url):
                validator = Mock(return_value=True)
                validate_form_event(event(), public_url=url, validator=validator)
                self.assertEqual(validator.call_args.args[0], url)

    def test_malformed_callback_url_escapes_fail_before_transport_or_validator(self):
        for escape in ("%", "%2", "%GG", "%2G", "%G2", "%２Ｆ"):
            for url in (f"https://example.invalid/voice/{escape}",
                        f"https://example.invalid/voice?value={escape}"):
                with self.subTest(url=url):
                    validator = Mock(return_value=True)
                    with patch("zentomic.authentication.parse_form_event") as parse:
                        with self.assertRaisesRegex(
                            ValueError, "^public_url must be a configured HTTPS callback URL$",
                        ):
                            validate_form_event(event(), public_url=url, validator=validator)
                    parse.assert_not_called()
                    validator.assert_not_called()

    def test_valid_callback_url_escapes_are_not_decoded_or_normalized(self):
        for suffix in ("/path%2fpart", "/path%2Fpart", "/literal%25",
                       "/literal%252G", "/voice?x=%26%3D&x=%2b&blank="):
            url = "https://example.invalid" + suffix
            with self.subTest(url=url):
                validator = Mock(return_value=True)
                validate_form_event(event(), public_url=url, validator=validator)
                self.assertEqual(validator.call_args.args[0], url)

    def test_validator_cannot_mutate_returned_fields_or_event(self):
        candidate = event()
        original = copy.deepcopy(candidate)

        def validator(url, fields, signature):
            fields["Digits"] = "9"
            return True

        self.assertEqual(validate_form_event(candidate, public_url=URL, validator=validator)["Digits"], "1")
        self.assertEqual(candidate, original)

    def test_validator_failure_is_private_and_offline(self):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.dict("os.environ", {}, clear=True), \
                patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            with self.assertRaises(ValueError) as caught:
                validate_form_event(event(), public_url=URL,
                                    validator=Mock(side_effect=RuntimeError("synthetic-private-marker")))
            validate_form_event(event(), public_url=URL, validator=Mock(return_value=True))
        self.assertEqual(str(caught.exception), "webhook signature validation failed")
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
