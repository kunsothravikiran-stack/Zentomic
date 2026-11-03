"""Offline transport validation for API Gateway form webhook events."""

import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

from zentomic.webhook import parse_form_body


_FORM_CONTENT_TYPE = re.compile(
    r'application/x-www-form-urlencoded(?:[ \t]*;[ \t]*charset=(?:utf-8|"utf-8"))?',
    re.IGNORECASE | re.ASCII,
)
MAX_CONTENT_TYPE_CHARACTERS = 128
MAX_HEADER_FIELDS = 128
MAX_HEADER_NAME_CHARACTERS = 128


def _content_type(headers: Any, *, multiple: bool = False) -> str | None:
    if headers is None:
        return None
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a mapping")
    # API Gateway bounds the wire request, but callers can invoke this offline
    # helper with arbitrary mappings. Bound both collection size and each name
    # before scanning or allocating a lowercase copy.
    entries = tuple(islice(headers.items(), MAX_HEADER_FIELDS + 1))
    if len(entries) > MAX_HEADER_FIELDS:
        raise ValueError("header collection exceeds size limit")
    values = []
    for name, value in entries:
        if not isinstance(name, str):
            continue
        # Only inert built-in primitives may cross this unauthenticated
        # boundary. Subclasses can override length, case folding, trimming,
        # indexing, or other operations performed below.
        if type(name) is not str:
            raise ValueError("header name must be a plain string")
        if len(name) > MAX_HEADER_NAME_CHARACTERS:
            raise ValueError("header name exceeds size limit")
        if name.lower() == "content-type":
            values.append(value)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("content type must be unambiguous")
    value = values[0]
    if multiple:
        if type(value) is not list or len(value) != 1:
            raise ValueError("content type must have exactly one value")
        value = value[0]
    # Bound work before trimming or matching an untrusted header value. Every
    # accepted media type is ASCII, so a character limit also bounds its bytes.
    if (type(value) is not str or len(value) > MAX_CONTENT_TYPE_CHARACTERS
            or not _FORM_CONTENT_TYPE.fullmatch(value.strip(" \t"))):
        raise ValueError("content type must be a UTF-8 URL-encoded form")
    return value


def _require_plain_string_keys(candidate: dict[Any, Any], label: str) -> None:
    """Reject programmatic dictionaries that JSON decoding cannot produce."""
    if any(type(key) is not str for key in candidate):
        raise ValueError(f"{label} keys must be plain strings")


def parse_form_event(event: Mapping[str, Any]) -> dict[str, str]:
    """Validate POST/form transport and decode a v1 or v2 proxy body.

    This does not authenticate input, select a route, or consume an attempt.
    Retain the unchanged original event for subsequent signature validation.
    Invalid transport raises ValueError without reflecting request contents.
    """
    # JSON decoding produces plain dictionaries and strings. Requiring those
    # inert primitives keeps unauthenticated request objects from executing
    # overridden ``get`` or comparison methods inside this trust boundary.
    if type(event) is not dict:
        raise ValueError("event must be a plain dictionary")
    _require_plain_string_keys(event, "event")
    version = event.get("version", "1.0")
    if type(version) is not str:
        raise ValueError("unsupported proxy event version")
    if version == "1.0":
        method = event.get("httpMethod")
    elif version == "2.0":
        context = event.get("requestContext")
        if type(context) is not dict:
            raise ValueError("v2 request context must be a plain dictionary")
        _require_plain_string_keys(context, "v2 request context")
        http = context.get("http")
        if type(http) is not dict:
            raise ValueError("v2 HTTP context must be a plain dictionary")
        _require_plain_string_keys(http, "v2 HTTP context")
        method = http.get("method")
    else:
        raise ValueError("unsupported proxy event version")
    if type(method) is not str or method != "POST":
        raise ValueError("form webhook requires POST")

    single = _content_type(event.get("headers"))
    multiple = _content_type(event.get("multiValueHeaders"), multiple=True)
    # REST v1 mirrors a header in both maps. Accept that exact mirror, not
    # conflicting values or duplicate Content-Type fields. V2 joins duplicates
    # with commas, which the media-type allowlist rejects.
    if single is None and multiple is None:
        raise ValueError("content type is required")
    if single is not None and multiple is not None and single != multiple:
        raise ValueError("content type headers disagree")
    if version == "2.0" and multiple is not None:
        raise ValueError("v2 content type must use headers")

    return parse_form_body(
        event.get("body"), is_base64_encoded=event.get("isBase64Encoded", False),
    )
