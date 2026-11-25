"""Offline transport validation for API Gateway form webhook events."""

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from zentomic.webhook import parse_form_body


_FORM_CONTENT_TYPE = re.compile(
    r'application/x-www-form-urlencoded(?:[ \t]*;[ \t]*charset=(?:utf-8|"utf-8"))?',
    re.IGNORECASE | re.ASCII,
)
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", re.ASCII)
MAX_CONTENT_TYPE_CHARACTERS = 128
MAX_HEADER_FIELDS = 128
MAX_HEADER_NAME_CHARACTERS = 128
MAX_EVENT_FIELDS = 64
MAX_EVENT_KEY_CHARACTERS = 128
MAX_METHOD_CHARACTERS = 32
MAX_PROXY_VERSION_CHARACTERS = 16


def _content_type(headers: Any, *, multiple: bool = False) -> str | None:
    if headers is None:
        return None
    # API Gateway JSON decoding produces an inert built-in dictionary. Reject
    # programmatic mappings before invoking their length or iteration hooks at
    # this unauthenticated boundary.
    if type(headers) is not dict:
        raise ValueError("headers must be a plain dictionary")
    if len(headers) > MAX_HEADER_FIELDS:
        raise ValueError("header collection exceeds size limit")
    values = []
    for name, value in headers.items():
        # Only inert built-in primitives may cross this unauthenticated
        # boundary. Subclasses can override length, case folding, trimming,
        # indexing, or other operations performed below.
        if type(name) is not str:
            raise ValueError("header name must be a plain string")
        # HTTP field names use the ASCII token grammar. Reject whitespace,
        # controls, separators, and Unicode before case folding or comparing
        # names so malformed direct Lambda events cannot introduce ambiguous
        # structural keys at this unauthenticated boundary.
        if (len(name) > MAX_HEADER_NAME_CHARACTERS
                or not _HEADER_NAME.fullmatch(name)):
            raise ValueError("header name must be a bounded HTTP token")
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
    """Reject event containers that exceed the bounded proxy shape."""
    if len(candidate) > MAX_EVENT_FIELDS:
        raise ValueError(f"{label} exceeds field limit")
    for key in candidate:
        if type(key) is not str:
            raise ValueError(f"{label} keys must be plain strings")
        if not key:
            raise ValueError(f"{label} keys must not be empty")
        if len(key) > MAX_EVENT_KEY_CHARACTERS:
            raise ValueError(f"{label} key exceeds size limit")
        # Invisible controls and Unicode line separators can make unknown proxy
        # fields appear identical to trusted structural keys or split one field
        # across log lines. Preserve visible Unicode extension keys, but reject
        # controls, formatting marks, surrogates, and line separators before
        # any downstream lookup.
        if any(
            unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            for char in key
        ):
            raise ValueError(f"{label} key contains an invisible control")


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
    if (type(version) is not str
            or len(version) > MAX_PROXY_VERSION_CHARACTERS):
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
    if (type(method) is not str or len(method) > MAX_METHOD_CHARACTERS
            or method != "POST"):
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
