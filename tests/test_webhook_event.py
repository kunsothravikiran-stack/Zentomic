"""Synthetic proxy transport tests, without authentication or service calls."""

import base64
import copy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from zentomic.webhook_event import (
    MAX_CONTENT_TYPE_CHARACTERS,
    MAX_HEADER_FIELDS,
    MAX_HEADER_NAME_CHARACTERS,
    parse_form_event,
)


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

    def test_content_type_length_limit_is_inclusive(self):
        boundary = FORM + " " * (MAX_CONTENT_TYPE_CHARACTERS - len(FORM))
        self.assertEqual(
            parse_form_event({**event(), "headers": {"Content-Type": boundary}}),
            {"Digits": "1"},
        )
        for value in (
            FORM + " " * (MAX_CONTENT_TYPE_CHARACTERS - len(FORM) + 1),
            "x" * (MAX_CONTENT_TYPE_CHARACTERS + 1),
        ):
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                parse_form_event({**event(), "headers": {"Content-Type": value}})

    def test_oversized_content_type_fails_before_trimming_or_matching(self):
        class OversizedContentType(str):
            def strip(self, *args, **kwargs):
                raise AssertionError("oversized content type must not be trimmed")

        value = OversizedContentType("x" * (MAX_CONTENT_TYPE_CHARACTERS + 1))
        for headers in (
            {"headers": {"Content-Type": value}},
            {"headers": None, "multiValueHeaders": {"Content-Type": [value]}},
        ):
            with self.subTest(headers=headers), self.assertRaises(ValueError) as caught:
                parse_form_event({**event(), **headers})
            self.assertNotIn("x", str(caught.exception))

    def test_header_collection_limit_is_inclusive(self):
        candidate = event()
        candidate["headers"].update(
            {f"X-Synthetic-{index}": "value"
             for index in range(MAX_HEADER_FIELDS - 1)}
        )
        self.assertEqual(len(candidate["headers"]), MAX_HEADER_FIELDS)
        self.assertEqual(parse_form_event(candidate), {"Digits": "1"})

        candidate["headers"]["X-One-Too-Many"] = "value"
        with patch("zentomic.webhook_event.parse_form_body") as decoder:
            with self.assertRaisesRegex(ValueError, "^header collection exceeds size limit$"):
                parse_form_event(candidate)
        decoder.assert_not_called()

    def test_active_header_mapping_is_rejected_before_hooks_run(self):
        class ActiveHeaders(dict):
            touched = False

            def __len__(self):
                self.touched = True
                raise AssertionError("active header length must not run")

            def items(self):
                self.touched = True
                raise AssertionError("active header iteration must not run")

        candidates = (
            {**event(), "headers": ActiveHeaders({"Content-Type": FORM})},
            {**event(), "headers": None,
             "multiValueHeaders": ActiveHeaders({"Content-Type": [FORM]})},
        )
        for field, candidate in zip(("headers", "multiValueHeaders"), candidates):
            with self.subTest(field=field), \
                    patch("zentomic.webhook_event.parse_form_body") as decoder:
                with self.assertRaisesRegex(
                    ValueError, "^headers must be a plain dictionary$",
                ):
                    parse_form_event(candidate)
                decoder.assert_not_called()

        self.assertFalse(candidates[0]["headers"].touched)
        self.assertFalse(candidates[1]["multiValueHeaders"].touched)

    def test_oversized_header_name_fails_before_case_folding(self):
        class OversizedHeaderName(str):
            def lower(self):
                raise AssertionError("oversized header name must not be case folded")

        name = OversizedHeaderName("X" * (MAX_HEADER_NAME_CHARACTERS + 1))
        candidate = event()
        candidate["headers"][name] = "synthetic-private-marker"
        with patch("zentomic.webhook_event.parse_form_body") as decoder:
            with self.assertRaisesRegex(ValueError, "^header name must be a plain string$") as caught:
                parse_form_event(candidate)
        decoder.assert_not_called()
        self.assertNotIn("synthetic-private-marker", str(caught.exception))

    def test_active_header_primitives_are_rejected_without_running_overrides(self):
        class ActiveName(str):
            def __len__(self):
                raise AssertionError("active header name length must not run")

            def lower(self):
                raise AssertionError("active header name case folding must not run")

        class ActiveValue(str):
            def __len__(self):
                raise AssertionError("active header value length must not run")

            def strip(self, *args, **kwargs):
                raise AssertionError("active header value trimming must not run")

        class ActiveValues(list):
            def __len__(self):
                raise AssertionError("active header list length must not run")

            def __getitem__(self, index):
                raise AssertionError("active header list indexing must not run")

        cases = (
            ({"headers": {1: "synthetic-private-marker", "Content-Type": FORM}},
             "^header name must be a plain string$"),
            ({"headers": {ActiveName("Content-Type"): FORM}},
             "^header name must be a plain string$"),
            ({"headers": {"Content-Type": ActiveValue(FORM)}},
             "^content type must be a UTF-8 URL-encoded form$"),
            ({"headers": None,
              "multiValueHeaders": {"Content-Type": ActiveValues([FORM])}},
             "^content type must have exactly one value$"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    parse_form_event({**event(), **overrides})

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
