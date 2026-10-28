"""Focused offline contract tests for API Gateway health security headers."""

import json
import unittest

from zentomic.handler import lambda_handler


def rest_event(method="GET", path="/health"):
    return {"httpMethod": method, "path": path}


def http_event(method="GET", path="/health"):
    return {
        "version": "2.0",
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
    }


class HandlerSecurityHeaderTests(unittest.TestCase):
    def test_csp_denies_document_capabilities_for_every_response(self):
        expected = "default-src 'none'; frame-ancestors 'none'; sandbox"
        events = (
            rest_event(),
            http_event("HEAD"),
            rest_event(path="/missing"),
            rest_event("POST"),
            {},
            {**rest_event(), "version": "3.0"},
        )
        for event in events:
            with self.subTest(event=event):
                response = lambda_handler(event, None)
                self.assertEqual(response["headers"]["Content-Security-Policy"], expected)

    def test_permissions_policy_disables_device_features_for_every_response(self):
        expected = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "microphone=(), payment=(), usb=()"
        )
        events = (
            rest_event(),
            http_event("HEAD"),
            rest_event(path="/missing"),
            rest_event("POST"),
            {},
            {**rest_event(), "version": "3.0"},
        )
        for event in events:
            with self.subTest(event=event):
                response = lambda_handler(event, None)
                self.assertEqual(response["headers"]["Permissions-Policy"], expected)

    def test_referrer_policy_suppresses_endpoint_disclosure_for_every_response(self):
        events = (
            rest_event(),
            http_event("HEAD"),
            rest_event(path="/missing"),
            rest_event("POST"),
            {},
            {**rest_event(), "version": "3.0"},
        )
        for event in events:
            with self.subTest(event=event):
                response = lambda_handler(event, None)
                self.assertEqual(response["headers"]["Referrer-Policy"], "no-referrer")

    def test_security_headers_do_not_change_payload_or_head_semantics(self):
        get_response = lambda_handler(http_event(), None)
        head_response = lambda_handler(http_event("HEAD"), None)

        self.assertEqual(
            json.loads(get_response["body"]),
            {"status": "ok", "service": "zentomic"},
        )
        self.assertEqual(head_response["body"], "")
        self.assertEqual(
            head_response["headers"]["Content-Length"],
            get_response["headers"]["Content-Length"],
        )


if __name__ == "__main__":
    unittest.main()
