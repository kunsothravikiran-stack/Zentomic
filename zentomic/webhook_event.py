"""Offline transport validation for API Gateway form webhook events."""

import re
from collections.abc import Mapping
from typing import Any

from zentomic.webhook import parse_form_body


_FORM_CONTENT_TYPE = re.compile(
    r'application/x-www-form-urlencoded(?:[ \t]*;[ \t]*charset=(?:utf-8|"utf-8"))?',
    re.IGNORECASE | re.ASCII,
)


def _content_type(headers: Any, *, multiple: bool = False) -> str | None:
    if headers is None:
        return None
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a mapping")
    values = [value for name, value in headers.items()
              if isinstance(name, str) and name.lower() == "content-type"]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("content type must be unambiguous")
    value = values[0]
    if multiple:
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError("content type must have exactly one value")
        value = value[0]
    if not isinstance(value, str) or not _FORM_CONTENT_TYPE.fullmatch(value.strip(" \t")):
        raise ValueError("content type must be a UTF-8 URL-encoded form")
    return value


def parse_form_event(event: Mapping[str, Any]) -> dict[str, str]:
    """Validate POST/form transport and decode a v1 or v2 proxy body.

    This does not authenticate input, select a route, or consume an attempt.
    Retain the unchanged original event for subsequent signature validation.
    Invalid transport raises ValueError without reflecting request contents.
    """
    if not isinstance(event, Mapping):
        raise ValueError("event must be a mapping")
    version = event.get("version", "1.0")
    if version == "1.0":
        method = event.get("httpMethod")
    elif version == "2.0":
        context = event.get("requestContext")
        http = context.get("http") if isinstance(context, Mapping) else None
        method = http.get("method") if isinstance(http, Mapping) else None
    else:
        raise ValueError("unsupported proxy event version")
    if method != "POST":
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
