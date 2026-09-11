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

    def test_valid_utf8_is_accepted_without_reflecting_input(self):
        raw = json.dumps({"httpMethod": "GET", "path": "/health", "body": "తెలుగు"},
                         ensure_ascii=False).encode("utf-8")
        code, output, errors, _ = self.invoke(raw)
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output)["statusCode"], 200)
        self.assertNotIn("తెలుగు", output)

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


if __name__ == "__main__":
    unittest.main()
