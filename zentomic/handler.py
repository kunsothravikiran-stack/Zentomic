"""Dependency-free liveness route for API Gateway proxy integrations."""

import json
from collections.abc import Mapping
from typing import Any

from zentomic import __version__


MAX_EVENT_FIELDS = 64
MAX_EVENT_KEY_CHARACTERS = 128


def _plain_bounded_proxy_container(candidate: Any) -> bool:
    """Return whether a decoded proxy container has a bounded inert shape."""
    if type(candidate) is not dict or len(candidate) > MAX_EVENT_FIELDS:
        return False
    return all(
        type(key) is str and len(key) <= MAX_EVENT_KEY_CHARACTERS
        for key in candidate
    )


def _context_request_id(context: Any) -> str | None:
    """Return a bounded Lambda invocation ID when the runtime supplies one."""
    request_id = getattr(context, "aws_request_id", None)
    if type(request_id) is not str or not request_id or len(request_id) > 128:
        return None
    try:
        request_id.encode("ascii")
    except UnicodeEncodeError:
        return None
    return request_id if all(character.isalnum() or character in "-_"
                             for character in request_id) else None


def _response(
    status: int, payload: dict, *, head: bool = False, empty: bool = False,
    request_id: str | None = None,
    **headers: str,
) -> dict:
    body = json.dumps(payload, separators=(",", ":"))
    response_headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(body.encode("utf-8"))),
        "Cache-Control": "no-store",
        # The health payload is fixed public metadata, so simple browser
        # checks may read success and error responses across origins.
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        # If a JSON response is opened as a document, deny executable,
        # embedded, and navigational capabilities in modern browsers.
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
        # Isolate a top-level health response from a cross-origin opener.
        "Cross-Origin-Opener-Policy": "same-origin",
        # The public health resource is intentionally readable across origins.
        # State that policy explicitly so cross-origin isolated dashboards can
        # load it without weakening isolation for their other resources.
        "Cross-Origin-Resource-Policy": "cross-origin",
        # API Gateway serves the endpoint over HTTPS. Remember that choice
        # for a year without extending policy to unrelated subdomains.
        # Let cross-origin monitoring dashboards include this public endpoint
        # in Resource Timing without exposing any request or customer data.
        "Timing-Allow-Origin": "*",
        "Strict-Transport-Security": "max-age=31536000",
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
        # Keep the operational endpoint out of search results and prevent
        # crawlers from following any content they might infer from it.
        "X-Robots-Tag": "noindex, nofollow",
        "X-Content-Type-Options": "nosniff",
        **headers,
    }
    if request_id is not None:
        response_headers["X-Zentomic-Request-Id"] = request_id
        exposed = response_headers.get("Access-Control-Expose-Headers")
        response_headers["Access-Control-Expose-Headers"] = (
            "X-Zentomic-Request-Id" if exposed is None
            else f"{exposed}, X-Zentomic-Request-Id"
        )
    if empty:
        # No-content responses cannot carry representation metadata. Enforce
        # this in the shared envelope so future callers cannot accidentally
        # advertise a body or override its zero length.
        response_headers.pop("Content-Type", None)
        response_headers["Content-Length"] = "0"
    return {
        "statusCode": status,
        "headers": response_headers,
        "body": "" if head or empty else body,
        "isBase64Encoded": False,
    }


def lambda_handler(event: Mapping[str, Any], context: Any) -> dict:
    """Handle REST v1 or HTTP v2 events without contacting external services."""
    request_id = _context_request_id(context)
    # API Gateway JSON decoding produces plain dictionaries with plain string
    # keys. Reject programmatic Mapping implementations and active key
    # subclasses before invoking ``get`` or dictionary key comparisons.
    if not _plain_bounded_proxy_container(event):
        return _response(400, {"error": "invalid_request"}, request_id=request_id)

    version = event.get("version", "1.0")
    # API Gateway JSON decoding produces exact built-in strings. Reject active
    # subclasses before equality, truthiness, or length operations so a
    # programmatic adapter cannot execute overrides inside route admission.
    if type(version) is not str:
        return _response(400, {"error": "invalid_request"}, request_id=request_id)
    if version == "2.0":
        request_context = event.get("requestContext")
        if not _plain_bounded_proxy_container(request_context):
            return _response(400, {"error": "invalid_request"}, request_id=request_id)
        http = request_context.get("http")
        if not _plain_bounded_proxy_container(http):
            return _response(400, {"error": "invalid_request"}, request_id=request_id)
        method = http.get("method")
        path = event.get("rawPath")
    elif version == "1.0":
        method = event.get("httpMethod")
        path = event.get("path")
    else:
        return _response(400, {"error": "invalid_request"}, request_id=request_id)

    if type(method) is not str or not method:
        return _response(400, {"error": "invalid_request"}, request_id=request_id)
    head = method == "HEAD"
    if type(path) is not str or not path:
        return _response(
            400, {"error": "invalid_request"}, head=head, request_id=request_id,
        )
    if path != "/health":
        return _response(404, {"error": "not_found"}, head=head, request_id=request_id)
    if method == "OPTIONS":
        return _response(
            204, {}, empty=True, request_id=request_id, Allow="GET, HEAD, OPTIONS",
            **{
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "600",
            },
        )
    if method not in ("GET", "HEAD"):
        return _response(
            405, {"error": "method_not_allowed"}, request_id=request_id,
            Allow="GET, HEAD, OPTIONS",
        )
    return _response(
        200,
        {"status": "ok", "service": "zentomic", "version": __version__},
        head=head,
        request_id=request_id,
        **{
            "Access-Control-Expose-Headers": "X-Zentomic-Version",
            "X-Zentomic-Version": __version__,
        },
    )
