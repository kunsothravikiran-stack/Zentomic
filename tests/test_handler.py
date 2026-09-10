"""Offline contract tests using only synthetic API Gateway events."""

import copy
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from zentomic.handler import lambda_handler


def rest_event(method="GET", path="/health"):
    return {"httpMethod": method, "path": path}


def http_event(method="GET", path="/health"):
    return {
        "version": "2.0",
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
    }


class HandlerTests(unittest.TestCase):
    def test_health_supports_both_proxy_formats(self):
        for event in (rest_event(), http_event(), {**rest_event(), "version": "1.0"}):
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(result["statusCode"], 200)
                self.assertEqual(json.loads(result["body"]), {"status": "ok", "service": "zentomic"})

    def test_proxy_response_contract(self):
        for event in (http_event(), rest_event(path="/missing"), rest_event("POST"), {}):
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(result["headers"]["Content-Type"], "application/json")
                self.assertEqual(result["headers"]["Cache-Control"], "no-store")
                self.assertIs(result["isBase64Encoded"], False)
                self.assertIsInstance(json.loads(result["body"]), dict)

    def test_unknown_paths_do_not_match_health(self):
        for factory in (rest_event, http_event):
            for path in ("/", "/health/", "/health/private", "/HEALTH"):
                with self.subTest(format=factory.__name__, path=path):
                    result = lambda_handler(factory(path=path), None)
                    self.assertEqual(result["statusCode"], 404)
                    self.assertEqual(json.loads(result["body"]), {"error": "not_found"})

    def test_health_rejects_other_methods(self):
        for factory in (rest_event, http_event):
            for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS", "head", "get"):
                with self.subTest(format=factory.__name__, method=method):
                    result = lambda_handler(factory(method=method), None)
                    self.assertEqual(result["statusCode"], 405)
                    self.assertEqual(result["headers"]["Allow"], "GET, HEAD")

    def test_head_matches_get_without_content_for_both_proxy_formats(self):
        for factory in (rest_event, http_event):
            for path, status in (("/health", 200), ("/missing", 404), (None, 400), ("", 400)):
                with self.subTest(format=factory.__name__, path=path):
                    expected = lambda_handler(factory(path=path), None)
                    expected["body"] = ""
                    result = lambda_handler(factory("HEAD", path), None)
                    self.assertEqual(result, expected)
                    self.assertEqual(result["statusCode"], status)

    def test_explicit_v1_head_and_v2_method_precedence(self):
        events = (
            {**rest_event("HEAD"), "version": "1.0"},
            {**http_event("HEAD"), "httpMethod": "GET"},
        )
        for event in events:
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(result["statusCode"], 200)
                self.assertEqual(result["body"], "")
        event = {**http_event("GET"), "httpMethod": "HEAD"}
        self.assertEqual(json.loads(lambda_handler(event, None)["body"])["status"], "ok")

    def test_head_is_offline_and_preserves_request_privacy(self):
        event = {**http_event("HEAD"), "body": "synthetic-private-marker"}
        original = copy.deepcopy(event)
        output = StringIO()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ), redirect_stdout(output), redirect_stderr(output):
            result = lambda_handler(event, None)
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"], "")
        self.assertEqual(event, original)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("synthetic", json.dumps(result))

    def test_invalid_events_return_400(self):
        events = (
            None, [], "invalid", {}, rest_event(method=None), rest_event(path=None),
            rest_event(method=""), rest_event(path=""), rest_event(path=[]),
            {**http_event(), "requestContext": None},
            {**http_event(), "requestContext": {"http": []}},
            {**http_event(), "requestContext": {"http": {}}},
        )
        for event in events:
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(result["statusCode"], 400)
                self.assertEqual(json.loads(result["body"]), {"error": "invalid_request"})

    def test_unsupported_version_is_rejected(self):
        for version in (None, "3.0", 2, []):
            with self.subTest(version=version):
                result = lambda_handler({**rest_event(), "version": version}, None)
                self.assertEqual(result["statusCode"], 400)

    def test_v2_uses_raw_path_and_http_method(self):
        event = {**http_event(), "path": "/missing", "httpMethod": "POST"}
        self.assertEqual(lambda_handler(event, None)["statusCode"], 200)

    def test_request_is_not_mutated(self):
        event = http_event()
        original = copy.deepcopy(event)
        lambda_handler(event, None)
        self.assertEqual(event, original)

    def test_request_data_is_not_logged_or_reflected(self):
        event = {**http_event(), "body": "synthetic-private-marker", "headers": {"Authorization": "synthetic-token"}}
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = lambda_handler(event, None)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("synthetic", json.dumps(result))

    def test_health_requires_no_network_or_environment(self):
        with patch.dict("os.environ", {}, clear=True), patch("socket.socket", side_effect=AssertionError("Network forbidden")):
            self.assertEqual(lambda_handler(http_event(), None)["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
