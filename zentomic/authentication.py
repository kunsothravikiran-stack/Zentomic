"""Dependency-injected signature gate for future form webhook adapters."""

import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from zentomic.webhook_event import parse_form_event


SignatureValidator = Callable[[str, dict[str, str], str], bool]
_SIGNATURE = re.compile(r"[A-Za-z0-9+/]{27}=")


def _signature_header(event: Mapping[str, Any]) -> str:
    values = []
    for key in ("headers", "multiValueHeaders"):
        headers = event.get(key) or {}
        matches = [value for name, value in headers.items()
                   if isinstance(name, str) and name.lower() == "x-twilio-signature"]
        if len(matches) > 1:
            raise ValueError("webhook signature must be unambiguous")
        if not matches:
            continue
        value = matches[0]
        if key == "multiValueHeaders":
            if event.get("version") == "2.0" or not isinstance(value, list) or len(value) != 1:
                raise ValueError("webhook signature must have exactly one value")
            value = value[0]
        if not isinstance(value, str) or not _SIGNATURE.fullmatch(value):
            raise ValueError("invalid webhook signature header")
        values.append(value)
    if not values or any(value != values[0] for value in values):
        raise ValueError("webhook signature is missing or conflicting")
    return values[0]


def validate_form_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
) -> dict[str, str]:
    """Return decoded fields only after a trusted validator explicitly succeeds.

    Inject a bound Twilio SDK RequestValidator.validate method configured with
    the expected account's auth token, never a callback supplied by the caller.
    public_url must be the exact externally configured HTTPS callback URL,
    including its original query string, not a URL built from request headers.
    No cryptography, SDK setup, token lookup, or routing is implemented here.
    This gate does not prevent replay or authorize workspace/session access.
    """
    if not isinstance(public_url, str) or any(char.isspace() for char in public_url):
        raise ValueError("public_url must be a configured HTTPS callback URL")
    try:
        url = urlsplit(public_url)
        valid_url = (url.scheme == "https" and bool(url.hostname)
                     and url.username is None and url.password is None
                     and not url.fragment and "#" not in public_url)
        url.port  # Reject malformed/out-of-range ports without rewriting the URL.
    except ValueError:
        valid_url = False
    if not valid_url or any(ord(char) < 32 or ord(char) == 127 for char in public_url):
        raise ValueError("public_url must be a configured HTTPS callback URL")
    if not callable(validator):
        raise ValueError("validator must be a trusted callable")

    fields = parse_form_event(event)
    signature = _signature_header(event)
    try:
        # A defensive copy prevents the injected dependency from changing the
        # fields subsequently consumed by routing or workspace authorization.
        valid = validator(public_url, dict(fields), signature)
    except Exception:
        raise ValueError("webhook signature validation failed") from None
    if valid is not True:
        raise ValueError("webhook signature validation failed")
    return fields
