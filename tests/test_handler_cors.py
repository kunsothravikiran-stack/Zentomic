"""Focused offline CORS contracts for the public health route."""

import unittest

from zentomic.handler import lambda_handler


def rest_event(method="GET"):
    return {"httpMethod": method, "path": "/health"}


def http_event(method="GET"):
    return {
        "version": "2.0",
        "rawPath": "/health",
        "requestContext": {"http": {"method": method}},
    }


class HandlerCorsTests(unittest.TestCase):
    def test_health_options_supports_cors_preflight(self):
        for factory in (rest_event, http_event):
            with self.subTest(format=factory.__name__):
                result = lambda_handler(factory("OPTIONS"), None)
                self.assertEqual(result["statusCode"], 204)
                self.assertEqual(result["body"], "")
                self.assertEqual(result["headers"]["Content-Length"], "0")
                self.assertEqual(result["headers"]["Allow"], "GET, HEAD, OPTIONS")
                self.assertEqual(
                    result["headers"]["Access-Control-Allow-Methods"],
                    "GET, HEAD, OPTIONS",
                )
                self.assertEqual(
                    result["headers"]["Access-Control-Allow-Headers"],
                    "Content-Type",
                )
                self.assertEqual(result["headers"]["Access-Control-Max-Age"], "600")

    def test_every_health_response_advertises_supported_cors_methods(self):
        for event in (http_event(), rest_event("HEAD"), rest_event("POST")):
            with self.subTest(event=event):
                result = lambda_handler(event, None)
                self.assertEqual(
                    result["headers"]["Access-Control-Allow-Methods"],
                    "GET, HEAD, OPTIONS",
                )

    def test_method_rejection_lists_options_as_supported(self):
        result = lambda_handler(rest_event("POST"), None)
        self.assertEqual(result["statusCode"], 405)
        self.assertEqual(result["headers"]["Allow"], "GET, HEAD, OPTIONS")


if __name__ == "__main__":
    unittest.main()
