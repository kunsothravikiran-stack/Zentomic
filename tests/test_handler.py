"""Offline contract tests using only synthetic API Gateway events."""

import copy
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from zentomic.handler import (
    MAX_EVENT_FIELDS,
    MAX_EVENT_KEY_CHARACTERS,
    lambda_handler,
)


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
                self.assertEqual(
                    result["headers"]["X-Zentomic-Version"], "0.1.0",
                )
                self.assertEqual(
                    result["headers"]["Access-Control-Expose-Headers"],
                    "X-Zentomic-Version",
                )
                self.assertEqual(
                    json.loads(result["body"]),
                    {"status": "ok", "service": "zentomic", "version": "0.1.0"},
                )

    def test_proxy_response_contract(self):
        for event in (http_event(), rest_event(path="/missing"), rest_event("POST"), {}):
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(
                    result["headers"]["Content-Type"],
                    "application/json; charset=utf-8",
                )
                self.assertEqual(
                    result["headers"]["Content-Length"],
                    str(len(result["body"].encode("utf-8"))),
                )
                self.assertEqual(result["headers"]["Cache-Control"], "no-store")
                self.assertEqual(result["headers"]["Access-Control-Allow-Origin"], "*")
                self.assertEqual(result["headers"]["X-Content-Type-Options"], "nosniff")
                self.assertIs(result["isBase64Encoded"], False)
                self.assertIsInstance(json.loads(result["body"]), dict)

    def test_lambda_request_id_correlates_every_response_path(self):
        context = SimpleNamespace(aws_request_id="offline-request-123")
        events = (
            http_event(),
            rest_event("HEAD"),
            rest_event("OPTIONS"),
            rest_event(path="/missing"),
            rest_event("POST"),
            None,
        )
        for event in events:
            with self.subTest(event=event):
                result = lambda_handler(event, context)
                self.assertEqual(
                    result["headers"]["X-Zentomic-Request-Id"],
                    "offline-request-123",
                )
                self.assertIn(
                    "X-Zentomic-Request-Id",
                    result["headers"]["Access-Control-Expose-Headers"].split(", "),
                )

    def test_missing_or_invalid_lambda_request_id_is_not_exposed(self):
        class ActiveRequestId(str):
            def __len__(self):
                raise AssertionError("active request ID length must not run")

            def encode(self, *args, **kwargs):
                raise AssertionError("active request ID encoding must not run")

        contexts = (
            None,
            SimpleNamespace(),
            SimpleNamespace(aws_request_id=None),
            SimpleNamespace(aws_request_id=""),
            SimpleNamespace(aws_request_id="with space"),
            SimpleNamespace(aws_request_id="తెలుగు"),
            SimpleNamespace(aws_request_id="x" * 129),
            SimpleNamespace(aws_request_id=ActiveRequestId("active")),
        )
        for context in contexts:
            with self.subTest(context=context):
                result = lambda_handler(http_event(), context)
                self.assertNotIn("X-Zentomic-Request-Id", result["headers"])
                self.assertEqual(
                    result["headers"]["Access-Control-Expose-Headers"],
                    "X-Zentomic-Version",
                )

    def test_failing_context_request_id_does_not_break_health(self):
        class FailingContext:
            @property
            def aws_request_id(self):
                raise RuntimeError("synthetic-private-marker")

        result = lambda_handler(http_event(), FailingContext())

        self.assertEqual(result["statusCode"], 200)
        self.assertNotIn("X-Zentomic-Request-Id", result["headers"])
        self.assertNotIn("synthetic-private-marker", json.dumps(result))

    def test_unknown_paths_do_not_match_health(self):
        for factory in (rest_event, http_event):
            for path in ("/", "/health/", "/health/private", "/HEALTH"):
                with self.subTest(format=factory.__name__, path=path):
                    result = lambda_handler(factory(path=path), None)
                    self.assertEqual(result["statusCode"], 404)
                    self.assertEqual(json.loads(result["body"]), {"error": "not_found"})

    def test_health_rejects_other_methods(self):
        for factory in (rest_event, http_event):
            for method in ("POST", "PUT", "DELETE", "PATCH", "head", "get"):
                with self.subTest(format=factory.__name__, method=method):
                    result = lambda_handler(factory(method=method), None)
                    self.assertEqual(result["statusCode"], 405)
                    self.assertEqual(
                        result["headers"]["Allow"], "GET, HEAD, OPTIONS",
                    )

    def test_head_matches_get_without_content_for_both_proxy_formats(self):
        for factory in (rest_event, http_event):
            for path, status in (("/health", 200), ("/missing", 404), (None, 400), ("", 400)):
                with self.subTest(format=factory.__name__, path=path):
                    get_result = lambda_handler(factory(path=path), None)
                    expected = copy.deepcopy(get_result)
                    expected["body"] = ""
                    result = lambda_handler(factory("HEAD", path), None)
                    self.assertEqual(result, expected)
                    self.assertEqual(result["statusCode"], status)
                    self.assertEqual(
                        result["headers"]["Content-Length"],
                        str(len(get_result["body"].encode("utf-8"))),
                    )

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

    def test_proxy_structure_field_limits_are_inclusive(self):
        cases = (
            ("event", http_event()),
            ("v2 request context", http_event()),
            ("v2 HTTP context", http_event()),
        )
        for location, candidate in cases:
            container = candidate
            if location == "v2 request context":
                container = candidate["requestContext"]
            elif location == "v2 HTTP context":
                container = candidate["requestContext"]["http"]
            container.update({
                f"X-Synthetic-{index}": None
                for index in range(MAX_EVENT_FIELDS - len(container))
            })
            with self.subTest(location=location, boundary=True):
                self.assertEqual(lambda_handler(candidate, None)["statusCode"], 200)

            container["X-One-Too-Many"] = None
            with self.subTest(location=location, boundary=False):
                result = lambda_handler(candidate, None)
                self.assertEqual(result["statusCode"], 400)
                self.assertEqual(
                    json.loads(result["body"]), {"error": "invalid_request"},
                )

    def test_proxy_structure_key_length_limits_are_inclusive(self):
        cases = (
            ("event", http_event()),
            ("v2 request context", http_event()),
            ("v2 HTTP context", http_event()),
        )
        for location, candidate in cases:
            container = candidate
            if location == "v2 request context":
                container = candidate["requestContext"]
            elif location == "v2 HTTP context":
                container = candidate["requestContext"]["http"]
            container["X" * MAX_EVENT_KEY_CHARACTERS] = None
            with self.subTest(location=location, boundary=True):
                self.assertEqual(lambda_handler(candidate, None)["statusCode"], 200)

            container["X" * (MAX_EVENT_KEY_CHARACTERS + 1)] = None
            with self.subTest(location=location, boundary=False):
                self.assertEqual(lambda_handler(candidate, None)["statusCode"], 400)

    def test_unsupported_version_is_rejected(self):
        for version in (None, "3.0", 2, []):
            with self.subTest(version=version):
                result = lambda_handler({**rest_event(), "version": version}, None)
                self.assertEqual(result["statusCode"], 400)

    def test_active_route_strings_are_rejected_without_invoking_overrides(self):
        class ActiveString(str):
            touched = False

            def _activate(self, *args, **kwargs):
                self.touched = True
                raise AssertionError("active string override executed")

            __bool__ = _activate
            __eq__ = _activate
            __hash__ = _activate
            __len__ = _activate
            __ne__ = _activate

        cases = (
            ({**rest_event(), "version": ActiveString("1.0")}, "version"),
            (rest_event(method=ActiveString("GET")), "httpMethod"),
            (rest_event(path=ActiveString("/health")), "path"),
            (http_event(method=ActiveString("GET")), "method"),
            (http_event(path=ActiveString("/health")), "rawPath"),
        )
        for event, field in cases:
            with self.subTest(field=field):
                result = lambda_handler(event, None)
                self.assertEqual(result["statusCode"], 400)
                self.assertEqual(json.loads(result["body"]), {"error": "invalid_request"})
                value = (
                    event["requestContext"]["http"][field]
                    if field == "method"
                    else event[field]
                )
                self.assertFalse(value.touched)

    def test_active_event_mappings_are_rejected_before_lookup(self):
        class ActiveMapping(dict):
            touched = False

            def get(self, *args, **kwargs):
                self.touched = True
                raise AssertionError("active mapping lookup executed")

        candidates = (
            ActiveMapping(rest_event()),
            {**http_event(), "requestContext": ActiveMapping({"http": {"method": "GET"}})},
            {**http_event(), "requestContext": {"http": ActiveMapping({"method": "GET"})}},
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                result = lambda_handler(candidate, None)
                self.assertEqual(result["statusCode"], 400)
                self.assertEqual(json.loads(result["body"]), {"error": "invalid_request"})

        self.assertFalse(candidates[0].touched)
        self.assertFalse(candidates[1]["requestContext"].touched)
        self.assertFalse(candidates[2]["requestContext"]["http"].touched)

    def test_active_event_keys_are_rejected_before_comparison(self):
        class ActiveKey(str):
            touched = False

            __hash__ = str.__hash__

            def __eq__(self, other):
                self.touched = True
                raise AssertionError("active key comparison executed")

        cases = (
            ({ActiveKey("version"): "1.0", **rest_event()}, "event"),
            ({**http_event(), "requestContext": {ActiveKey("http"): {"method": "GET"}}}, "context"),
            ({**http_event(), "requestContext": {"http": {ActiveKey("method"): "GET"}}}, "http"),
        )
        for candidate, location in cases:
            with self.subTest(location=location):
                result = lambda_handler(candidate, None)
                self.assertEqual(result["statusCode"], 400)
                self.assertEqual(json.loads(result["body"]), {"error": "invalid_request"})

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
