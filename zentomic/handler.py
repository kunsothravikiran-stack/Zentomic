"""Dependency-free liveness route for API Gateway proxy integrations."""

import json
from collections.abc import Mapping
from typing import Any


def _response(status: int, payload: dict, *, head: bool = False, **headers: str) -> dict:
    body = json.dumps(payload, separators=(",", ":"))
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body.encode("utf-8"))),
            "Cache-Control": "no-store",
            # The health payload is fixed public metadata, so simple browser
            # checks may read success and error responses across origins.
            "Access-Control-Allow-Origin": "*",
            # If a JSON response is opened as a document, deny executable,
            # embedded, and navigational capabilities in modern browsers.
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
            # Isolate a top-level health response from a cross-origin opener.
            "Cross-Origin-Opener-Policy": "same-origin",
            # Preserve frame denial for clients that do not enforce CSP.
            "X-Frame-Options": "DENY",
            # The endpoint never needs browser-controlled device capabilities.
            "Permissions-Policy": (
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "microphone=(), payment=(), usb=()"
            ),
            # Do not disclose the health endpoint's URL through a Referer
            # header if a browser follows or embeds content from this response.
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            **headers,
        },
        "body": "" if head else body,
        "isBase64Encoded": False,
    }


def lambda_handler(event: Mapping[str, Any], context: Any) -> dict:
    """Handle REST v1 or HTTP v2 events without contacting external services."""
    if not isinstance(event, Mapping):
        return _response(400, {"error": "invalid_request"})

    version = event.get("version", "1.0")
    if version == "2.0":
        request_context = event.get("requestContext")
        http = request_context.get("http") if isinstance(request_context, Mapping) else None
        method = http.get("method") if isinstance(http, Mapping) else None
        path = event.get("rawPath")
    elif version == "1.0":
        method = event.get("httpMethod")
        path = event.get("path")
    else:
        return _response(400, {"error": "invalid_request"})

    head = method == "HEAD"
    if not isinstance(method, str) or not method or not isinstance(path, str) or not path:
        return _response(400, {"error": "invalid_request"}, head=head)
    if path != "/health":
        return _response(404, {"error": "not_found"}, head=head)
    if method not in ("GET", "HEAD"):
        return _response(405, {"error": "method_not_allowed"}, Allow="GET, HEAD")
    return _response(200, {"status": "ok", "service": "zentomic"}, head=head)
