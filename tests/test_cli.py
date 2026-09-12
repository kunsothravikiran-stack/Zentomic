"""Synthetic event replay through the offline Lambda command line."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.__main__ import offline_guard
from zentomic.__main__ import MAX_EVENT_BYTES, main


class EventCliTests(unittest.TestCase):
    def invoke(self, raw, argv=None):
        output, errors = io.StringIO(), io.StringIO()
        stream = io.BytesIO(raw)
        with offline_guard(), patch("sys.stdin", SimpleNamespace(buffer=stream)), \
                redirect_stdout(output), redirect_stderr(errors):
            code = main(["--stdin"] if argv is None else argv)
        return code, output.getvalue(), errors.getvalue(), stream.tell()

    def test_default_smoke_does_not_read_stdin(self):
        code, output, errors, consumed = self.invoke(b"invalid", [])
        self.assertEqual((code, errors, consumed), (0, "", 0))
        self.assertEqual(json.loads(output)["statusCode"], 200)

    def test_replays_both_formats_and_preserves_http_error_responses(self):
        for version in ("1.0", "2.0"):
            for method, path, status in (("GET", "/health", 200),
                                         ("HEAD", "/health", 200),
                                         ("POST", "/health", 405),
                                         ("GET", "/missing", 404)):
                event = {"version": version, "body": "synthetic-private-text"}
                if version == "1.0":
                    event.update(httpMethod=method, path=path)
                else:
                    event.update(rawPath=path, requestContext={"http": {"method": method}})
                with self.subTest(version=version, method=method, path=path):
                    code, output, errors, _ = self.invoke(json.dumps(event).encode())
                    self.assertEqual((code, errors), (0, ""))
                    self.assertEqual(json.loads(output)["statusCode"], status)
                    self.assertNotIn("synthetic-private", output)
                    if method == "HEAD":
                        self.assertEqual(json.loads(output)["body"], "")

    def test_invalid_json_is_rejected_without_invoking_handler_or_echoing(self):
        for raw in (b"", b"synthetic-private", b"{} {}", b"[]", b"null",
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}',
                    b'{"x":1,"x":2}', b'{"nested":{"x":1,"x":2}}',
                    b'{"x":"\xff"}', b"[" * 2000 + b"]" * 2000):
            with self.subTest(raw=raw[:40]), patch("zentomic.__main__.lambda_handler") as handler:
                code, output, errors, _ = self.invoke(raw)
                self.assertEqual((code, output), (2, ""))
                self.assertEqual(errors, "Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.\n")
                handler.assert_not_called()

    def test_byte_limit_is_inclusive_and_read_is_bounded(self):
        exact = b"{}" + b" " * (MAX_EVENT_BYTES - 2)
        self.assertEqual(self.invoke(exact)[:1], (0,))
        code, output, _, consumed = self.invoke(exact + b" " * MAX_EVENT_BYTES)
        self.assertEqual((code, output, consumed), (2, "", MAX_EVENT_BYTES + 1))
        # A character-count check would incorrectly admit this UTF-8 payload.
        multibyte = json.dumps({"body": "é" * (MAX_EVENT_BYTES // 2)},
                               ensure_ascii=False).encode("utf-8")
        self.assertGreater(len(multibyte), MAX_EVENT_BYTES)
        self.assertEqual(self.invoke(multibyte)[:2], (2, ""))

    def test_overflowing_json_numbers_never_reach_handler(self):
        for number in (b"1e400", b"-1e400", b"1.8e308", b"9" * 400 + b".0"):
            raw = b'{"private-number":[{"nested":' + number + b'}]}'
            with self.subTest(number=number[:20]), \
                    patch("zentomic.__main__.lambda_handler", return_value={}) as handler:
                code, output, errors, _ = self.invoke(raw)
                self.assertEqual((code, output), (2, ""))
                self.assertEqual(errors, "Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.\n")
                handler.assert_not_called()

    def test_finite_numbers_keep_standard_json_types(self):
        raw = b'{"values":[0,1.25,-2.5e2,1.7976931348623157e308,5e-324,' + b"9" * 400 + b']}'
        with patch("zentomic.__main__.lambda_handler", return_value={}) as handler:
            code, output, errors, _ = self.invoke(raw)
        self.assertEqual((code, output, errors), (0, "{}\n", ""))
        values = handler.call_args.args[0]["values"]
        self.assertEqual(values, [0, 1.25, -250.0, float.fromhex("0x1.fffffffffffffp+1023"),
                                  5e-324, int("9" * 400)])
        self.assertIs(type(values[0]), int)
        self.assertIs(type(values[1]), float)
        self.assertIs(type(values[-1]), int)

    def test_valid_utf8_is_accepted_without_reflecting_input(self):
        raw = json.dumps({"httpMethod": "GET", "path": "/health", "body": "తెలుగు"},
                         ensure_ascii=False).encode("utf-8")
        code, output, errors, _ = self.invoke(raw)
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output)["statusCode"], 200)
        self.assertNotIn("తెలుగు", output)

    def test_escaped_lone_surrogates_never_reach_handler(self):
        for raw in (b'{"body":"\\ud800"}', b'{"\\udfff":"value"}',
                    b'{"nested":[{"value":["private-\\ud800-text"]}]}',
                    b'{"body":"\\udc00\\ud800"}'):
            for argv in (["--stdin"], ["--event-file", "synthetic.json"]):
                with self.subTest(raw=raw, argv=argv), \
                        patch("builtins.open", return_value=io.BytesIO(raw)), \
                        patch("zentomic.__main__.lambda_handler", return_value={}) as handler:
                    code, output, errors, _ = self.invoke(raw, argv)
                    self.assertEqual((code, output), (2, ""))
                    self.assertEqual(errors, "Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.\n")
                    handler.assert_not_called()

    def test_escaped_unicode_pairs_and_literal_escape_text_are_preserved(self):
        raw = (b'{"\\ud83d\\ude00":["\\ud83d\\ude00",'
               b'"\\u0c24\\u0c46", "\\\\ud800", null, true, 42]}')
        with patch("zentomic.__main__.lambda_handler", return_value={}) as handler:
            code, output, errors, _ = self.invoke(raw)
        self.assertEqual((code, output, errors), (0, "{}\n", ""))
        self.assertEqual(handler.call_args.args[0],
                         {"😀": ["😀", "తె", "\\ud800", None, True, 42]})

    def test_leading_utf8_bom_replays_the_same_event(self):
        for event in (
            {"httpMethod": "HEAD", "path": "/health"},
            {"version": "2.0", "rawPath": "/health",
             "requestContext": {"http": {"method": "GET"}}, "body": "తెలుగు"},
        ):
            raw = json.dumps(event, ensure_ascii=False).encode("utf-8")
            with self.subTest(event=event):
                self.assertEqual(self.invoke(b"\xef\xbb\xbf" + raw)[:3], self.invoke(raw)[:3])

    def test_bom_bytes_count_toward_input_limit(self):
        exact = b"\xef\xbb\xbf{}" + b" " * (MAX_EVENT_BYTES - 5)
        self.assertEqual(self.invoke(exact)[:1], (0,))
        code, output, _, consumed = self.invoke(exact + b" ")
        self.assertEqual((code, output, consumed), (2, "", MAX_EVENT_BYTES + 1))

    def test_only_one_leading_utf8_bom_is_accepted(self):
        for raw in (b"\xef\xbb\xbf\xef\xbb\xbf{}", b" \xef\xbb\xbf{}",
                    "{}".encode("utf-16"), "{}".encode("utf-32")):
            with self.subTest(raw=raw), patch("zentomic.__main__.lambda_handler") as handler:
                code, output, errors, _ = self.invoke(raw)
                self.assertEqual((code, output), (2, ""))
                self.assertEqual(errors, "Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.\n")
                handler.assert_not_called()

    def test_read_failure_has_generic_diagnostics(self):
        stream = Mock()
        stream.read.side_effect = OSError("synthetic-private-device-name")
        errors, output = io.StringIO(), io.StringIO()
        with offline_guard(), patch("sys.stdin", SimpleNamespace(buffer=stream)), \
                redirect_stderr(errors), redirect_stdout(output):
            self.assertEqual(main(["--stdin"]), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("synthetic-private", errors.getvalue())

    def test_event_file_matches_stdin_and_closes_file_without_reading_stdin(self):
        for raw in (b'{"httpMethod":"HEAD","path":"/health"}',
                    b'\xef\xbb\xbf{"version":"2.0","rawPath":"/health",'
                    b'"requestContext":{"http":{"method":"GET"}}}',
                    b'{"httpMethod":"GET","path":"/missing"}',
                    b'{"duplicate":1,"duplicate":2}',
                    b'{"value":1e400}',
                    b'{"body":"\xff"}',
                    b'{}' + b' ' * (MAX_EVENT_BYTES - 2),
                    b'{}' + b' ' * MAX_EVENT_BYTES):
            with self.subTest(raw=raw[:40]):
                expected = self.invoke(raw)[:3]
                stream = io.BytesIO(raw)
                with patch("builtins.open", return_value=stream) as opened, \
                        patch.object(stream, "read", wraps=stream.read) as read:
                    actual = self.invoke(b"do not read", ["--event-file", "synthetic.json"])
                    read.assert_called_once_with(MAX_EVENT_BYTES + 1)
                self.assertEqual(actual[:3], expected)
                self.assertEqual(actual[3], 0)
                opened.assert_called_once_with("synthetic.json", "rb")
                self.assertTrue(stream.closed)

    def test_event_file_errors_do_not_reflect_path_or_invoke_handler(self):
        for failure in (FileNotFoundError("synthetic-private-path"),
                        PermissionError("synthetic-private-path"),
                        IsADirectoryError("synthetic-private-path")):
            with self.subTest(failure=type(failure)), \
                    patch("builtins.open", side_effect=failure), \
                    patch("zentomic.__main__.lambda_handler") as handler:
                code, output, errors, consumed = self.invoke(
                    b"do not read", ["--event-file", "synthetic-private-path"],
                )
                self.assertEqual((code, output, consumed), (2, "", 0))
                self.assertEqual(errors, "Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.\n")
                handler.assert_not_called()

    def test_event_file_is_closed_after_read_failure(self):
        stream = io.BytesIO(b"{}")
        with patch("builtins.open", return_value=stream), \
                patch.object(stream, "read", side_effect=OSError("private-device")), \
                patch("zentomic.__main__.lambda_handler") as handler:
            code, output, errors, consumed = self.invoke(
                b"do not read", ["--event-file", "synthetic.json"],
            )
        self.assertEqual((code, output, consumed), (2, "", 0))
        self.assertNotIn("private-device", errors)
        self.assertTrue(stream.closed)
        handler.assert_not_called()

    def test_event_file_and_stdin_are_mutually_exclusive(self):
        with patch("builtins.open") as opened:
            with self.assertRaises(SystemExit) as error:
                self.invoke(b"{}", ["--stdin", "--event-file", "synthetic.json"])
        self.assertEqual(error.exception.code, 2)
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
