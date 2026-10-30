"""Offline validation of Twilio form-webhook signatures."""

import base64
import binascii
import hashlib
import hmac
from collections.abc import Callable, Mapping
from itertools import islice

from zentomic.authentication import MAX_PUBLIC_URL_BYTES, MAX_SIGNATURE_CHARACTERS
from zentomic.webhook import MAX_BODY_BYTES, MAX_FORM_FIELDS


MAX_AUTH_TOKEN_BYTES = 256
MAX_SIGNED_PAYLOAD_BYTES = MAX_PUBLIC_URL_BYTES + MAX_BODY_BYTES


def _auth_token_bytes(auth_token: str) -> bytes:
    """Validate trusted secret configuration without disclosing its value."""
    if (not isinstance(auth_token, str)
            or not auth_token
            or len(auth_token) > MAX_AUTH_TOKEN_BYTES):
        raise ValueError("auth_token must be a nonempty UTF-8 string of at most 256 bytes")
    try:
        encoded = auth_token.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            "auth_token must be a nonempty UTF-8 string of at most 256 bytes"
        ) from None
    if len(encoded) > MAX_AUTH_TOKEN_BYTES:
        raise ValueError("auth_token must be a nonempty UTF-8 string of at most 256 bytes")
    return encoded


def _validate_twilio_signature(
    public_url: str,
    fields: Mapping[str, str],
    signature: str,
    *,
    token: bytes,
) -> bool:
    """Validate request material with an already checked Auth Token."""
    if (not isinstance(public_url, str)
            or len(public_url) > MAX_PUBLIC_URL_BYTES
            or not isinstance(fields, Mapping)
            or not isinstance(signature, str)
            or len(signature) != MAX_SIGNATURE_CHARACTERS):
        return False

    try:
        supplied = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(supplied) != hashlib.sha1().digest_size:
        return False

    entries = tuple(islice(fields.items(), MAX_FORM_FIELDS + 1))
    if len(entries) > MAX_FORM_FIELDS:
        return False
    if any(type(name) is not str or type(value) is not str for name, value in entries):
        return False

    try:
        url_bytes = public_url.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(url_bytes) > MAX_PUBLIC_URL_BYTES:
        return False

    digest = hmac.new(token, digestmod=hashlib.sha1)
    digest.update(url_bytes)
    remaining = MAX_SIGNED_PAYLOAD_BYTES - len(url_bytes)
    for name, value in sorted(entries, key=lambda item: item[0]):
        for part in (name, value):
            # UTF-8 uses at least one byte per character. Check the character
            # count first so an oversized untrusted field cannot make the
            # validator allocate an equally oversized encoded copy.
            if len(part) > remaining:
                return False
            try:
                encoded = part.encode("utf-8")
            except UnicodeEncodeError:
                return False
            if len(encoded) > remaining:
                return False
            digest.update(encoded)
            remaining -= len(encoded)

    return hmac.compare_digest(digest.digest(), supplied)


def create_twilio_signature_validator(
    auth_token: str,
) -> Callable[[str, Mapping[str, str], str], bool]:
    """Create a reusable validator from trusted Auth Token configuration.

    Validate and encode the token once during adapter initialization so an
    invalid secret fails as configuration, before any webhook is processed.
    The returned callable has the signature expected by
    ``authentication.validate_form_event`` and performs no I/O.
    """
    token = _auth_token_bytes(auth_token)

    def validator(
        public_url: str, fields: Mapping[str, str], signature: str,
    ) -> bool:
        return _validate_twilio_signature(
            public_url, fields, signature, token=token,
        )

    return validator


def validate_twilio_signature(
    public_url: str,
    fields: Mapping[str, str],
    signature: str,
    *,
    auth_token: str,
) -> bool:
    """Validate one URL-encoded Twilio webhook with constant-time comparison.

    ``public_url`` must be the exact externally configured URL, including its
    original query string. ``fields`` must include every decoded POST field.
    Pass the Auth Token from trusted secret configuration, never request data.

    Invalid request material returns ``False``. Invalid trusted Auth Token
    configuration raises ``ValueError``. This helper performs no I/O, secret
    lookup, replay protection, workspace authorization, or callback routing.
    For repeated validation, prefer ``create_twilio_signature_validator`` so
    trusted configuration is checked and encoded once during initialization.
    """
    return _validate_twilio_signature(
        public_url, fields, signature, token=_auth_token_bytes(auth_token),
    )
