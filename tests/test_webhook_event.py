"""Synthetic proxy transport tests, without authentication or service calls."""

import base64
import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from zentomic.webhook_event import parse_form_event


FORM = "application/x-www-form-urlencoded"


def event(version="1.0"):
    result = {"version": version, "headers": {"Content-Type": FORM}, "body": "Digits=1"}
    if version == "2.0":
        result["requestContext"] = {"http": {"method": "POST"}}
    else:
        result["httpMethod"] = "POST"
    return result


class FormEventTests(unittest.TestCase):
    def test_supported_proxy_formats_and_base64(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                candidate = event(version)
                candidate["isBase64Encoded"] = encoded
                if encoded:
                    candidate["body"] = base64.b64encode(b"Digits=1").decode()
                with self.subTest(version=version, encoded=encoded):
                    self.assertEqual(parse_form_event(candidate), {"Digits": "1"})
        candidate = event()
        del candidate["version"]
        self.assertEqual(parse_form_event(candidate), {"Digits": "1"})

    def test_utf8_media_type_variants(self):
        for value in (FORM, FORM.upper(), FORM + "; charset=UTF-8", FORM + ';charset="utf-8"', "\t" + FORM + " "):
            with self.subTest(value=value):
                self.assertEqual(parse_form_event({**event(), "headers": {"cOnTeNt-TyPe": value}}), {"Digits": "1"})

    def test_v1_multivalue_header_and_mirror(self):
        for headers in (None, {}, {"Content-Type": FORM}):
            self.assertEqual(parse_form_event({**event(), "headers": headers,
                             "multiValueHeaders": {"content-type": [FORM]}}), {"Digits": "1"})

    def test_invalid_media_types(self):
        for value in (None, [], "", "application/json", FORM + ";charset=latin-1", FORM + "," + FORM,
                      FORM + ";charset=utf-8;charset=utf-8", FORM + "\r\n", FORM + ";boundary=x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_form_event({**event(), "headers": {"Content-Type": value}})

    def test_missing_malformed_and_ambiguous_headers(self):
        cases = [
            {"headers": None}, {"headers": []}, {"headers": {}},
            {"headers": {"Content-Type": FORM, "content-type": FORM}},
            {"multiValueHeaders": []},
            {"multiValueHeaders": {"Content-Type": []}},
            {"multiValueHeaders": {"Content-Type": FORM}},
            {"multiValueHeaders": {"Content-Type": [FORM, FORM]}},
            {"multiValueHeaders": {"Content-Type": [FORM + ";charset=utf-8"]}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                parse_form_event({**event(), **overrides})
        with self.assertRaises(ValueError):
            parse_form_event({**event("2.0"), "multiValueHeaders": {"Content-Type": [FORM]}})

    def test_post_required_before_body_decoding(self):
        for version in ("1.0", "2.0"):
            for method in (None, "GET", "post", "", [], True):
                candidate = event(version)
                if version == "1.0":
                    candidate["httpMethod"] = method
                else:
                    candidate["requestContext"]["http"]["method"] = method
                    candidate["httpMethod"] = "POST"
                with self.subTest(version=version, method=method):
                    with patch("zentomic.webhook_event.parse_form_body") as decoder:
                        with self.assertRaises(ValueError):
                            parse_form_event(candidate)
                        decoder.assert_not_called()

    def test_malformed_event_and_v2_context(self):
        cases = [None, [], "invalid", {}, {**event(), "version": "3.0"}]
        cases.extend({**event("2.0"), "requestContext": context}
                     for context in (None, [], {}, {"http": None}, {"http": []}))
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                parse_form_event(candidate)

    def test_missing_body_is_not_silence(self):
        candidate = event()
        del candidate["body"]
        with self.assertRaises(ValueError):
            parse_form_event(candidate)
        self.assertEqual(parse_form_event({**event(), "body": ""}), {})

    def test_decoder_rejections_are_preserved(self):
        for overrides in ({"body": None}, {"body": "Digits=1&Digits=2"}, {"body": "X=%FF"},
                          {"body": "X=" + "x" * 16384}, {"isBase64Encoded": "false"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                parse_form_event({**event(), **overrides})

    def test_offline_private_and_nonmutating(self):
        candidate = {**event(), "body": "Digits=1&FutureField=synthetic-private-marker"}
        original = copy.deepcopy(candidate)
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output), patch.dict("os.environ", {}, clear=True), \
                patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            self.assertEqual(parse_form_event(candidate)["FutureField"], "synthetic-private-marker")
            with self.assertRaises(ValueError) as caught:
                parse_form_event({**candidate, "headers": {"Content-Type": "synthetic-private-marker"}})
        self.assertNotIn("synthetic-private-marker", str(caught.exception))
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(candidate, original)


if __name__ == "__main__":
    unittest.main()
