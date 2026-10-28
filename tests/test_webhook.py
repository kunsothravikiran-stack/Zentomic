"""Synthetic form parsing checks without live webhook traffic."""

import base64
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from zentomic.gather import resolve_gather
from zentomic.webhook import MAX_BODY_BYTES, MAX_FORM_FIELDS, parse_form_body


class FormBodyTests(unittest.TestCase):
    def assert_invalid(self, body, **kwargs):
        with self.assertRaises(ValueError):
            parse_form_body(body, **kwargs)

    def test_plain_and_base64_forms_match(self):
        fields = {"Digits": "1", "Label": "Café తెలుగు 😀", "Optional": ""}
        body = urlencode(fields)
        encoded = base64.b64encode(body.encode()).decode()
        self.assertEqual(parse_form_body(body), fields)
        self.assertEqual(parse_form_body(encoded, is_base64_encoded=True), fields)

    def test_literal_utf8_is_supported(self):
        self.assertEqual(parse_form_body("Label=Café"), {"Label": "Café"})

    def test_blank_and_missing_digits_remain_distinct(self):
        self.assertEqual(parse_form_body("Digits="), {"Digits": ""})
        self.assertEqual(parse_form_body(""), {})
        self.assertEqual(parse_form_body("", is_base64_encoded=True), {})

    def test_decodes_once_and_preserves_values(self):
        self.assertEqual(parse_form_body("A=a+b%2Bc%26d%3De&B=%2531&C=x;y"), {
            "A": "a b+c&d=e", "B": "%31", "C": "x;y",
        })

    def test_unknown_fields_and_name_case_are_preserved(self):
        self.assertEqual(parse_form_body("Digits=1&digits=2&FutureField=new"), {
            "Digits": "1", "digits": "2", "FutureField": "new",
        })

    def test_duplicate_decoded_names_are_rejected(self):
        for body in ("Digits=1&Digits=2", "Digits=1&%44igits=1", "A=&A=", "a+b=1&a%20b=2"):
            with self.subTest(body=body):
                self.assert_invalid(body)

    def test_malformed_fields_and_empty_names_are_rejected(self):
        for body in ("Digits", "=1", "&Digits=1", "Digits=1&", "A=1&&B=2"):
            with self.subTest(body=body):
                self.assert_invalid(body)

    def test_malformed_percent_encodings_and_utf8_are_rejected(self):
        for body in ("A=%", "A=%1", "A=%GG", "%XY=1", "A=%FF", "A=%C3", "A=%ED%A0%80", "A=\ud800"):
            with self.subTest(body=repr(body)):
                self.assert_invalid(body)

    def test_invalid_base64_is_rejected(self):
        for body in ("%%%", "YQ", "QQ==\n", "é", "/w=="):
            with self.subTest(body=body):
                self.assert_invalid(body, is_base64_encoded=True)

    def test_body_and_flag_types_are_strict(self):
        for body in (None, False, 0, [], {}, b"Digits=1"):
            with self.subTest(body=body):
                self.assert_invalid(body)
        for flag in (None, 0, 1, "false", "true", [], {}):
            with self.subTest(flag=flag):
                self.assert_invalid("Digits=1", is_base64_encoded=flag)

    def test_byte_limit_applies_to_plain_and_decoded_bodies(self):
        boundary = "A=" + "x" * (MAX_BODY_BYTES - 2)
        for body in (boundary, boundary + "x", "A=" + "é" * (MAX_BODY_BYTES // 2)):
            for encoded in (False, True):
                candidate = base64.b64encode(body.encode()).decode() if encoded else body
                with self.subTest(size=len(body.encode()), encoded=encoded):
                    if body == boundary:
                        self.assertEqual(parse_form_body(candidate, is_base64_encoded=encoded), {"A": body[2:]})
                    else:
                        self.assert_invalid(candidate, is_base64_encoded=encoded)

    def test_field_count_limit(self):
        boundary = "&".join(f"Field{i}=" for i in range(MAX_FORM_FIELDS))
        self.assertEqual(len(parse_form_body(boundary)), MAX_FORM_FIELDS)
        self.assert_invalid(boundary + "&Extra=")

    def test_error_messages_do_not_include_request_data(self):
        for body in ("SyntheticPrivateValue", "Secret=%FF", "Secret=%", "Secret=1&Secret=2"):
            with self.subTest(body=body), self.assertRaises(ValueError) as caught:
                parse_form_body(body)
            self.assertNotIn("Secret", str(caught.exception))
            self.assertNotIn("SyntheticPrivateValue", str(caught.exception))

    def test_decoded_input_uses_existing_gather_policy(self):
        for body, action in (("Digits=1", "route"), ("Digits=", "retry"), ("", "retry"), ("Digits=%2531", "retry")):
            with self.subTest(body=body):
                fields = parse_form_body(body)
                result = resolve_gather(fields.get("Digits"), {"1": "sales"}, fallback_target="reception", attempts=0)
                self.assertEqual(result.action, action)

    def test_parser_is_offline_and_results_are_independent(self):
        with patch.dict("os.environ", {}, clear=True), patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            first = parse_form_body("Digits=1")
            first["Digits"] = "2"
            self.assertEqual(parse_form_body("Digits=1"), {"Digits": "1"})


if __name__ == "__main__":
    unittest.main()
