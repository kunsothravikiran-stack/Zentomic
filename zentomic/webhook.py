"""Bounded, offline decoding for form-encoded webhook bodies."""

import base64
import binascii
import re
import unicodedata
from urllib.parse import parse_qsl


MAX_BODY_BYTES = 16 * 1024
MAX_FORM_FIELDS = 128
MAX_FORM_FIELD_NAME_BYTES = 256
_MAX_BASE64_CHARS = 4 * ((MAX_BODY_BYTES + 2) // 3)
_INVALID_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def parse_form_body(body: str, *, is_base64_encoded: bool = False) -> dict[str, str]:
    """Decode a UTF-8 form body, rejecting ambiguous or oversized input.

    Pass API Gateway's body and isBase64Encoded flag, not caller form fields.
    Blank values and unknown fields are retained; duplicate decoded names are
    rejected rather than silently selecting a value. Empty bodies return {}.
    All invalid inputs raise ValueError without including body contents.

    This is only a transport decoder, not authentication or a webhook endpoint.
    The adapter must enforce method/content type, verify the provider signature,
    authorize workspace state, and deduplicate callbacks before acting on input.
    """
    # Request material must be inert built-in values. A ``str`` subclass can
    # override ``__len__`` or ``encode`` and execute adapter code before this
    # boundary has authenticated the webhook.
    if type(body) is not str or type(is_base64_encoded) is not bool:
        raise ValueError("body must be a string and encoding flag must be a boolean")

    limit = _MAX_BASE64_CHARS if is_base64_encoded else MAX_BODY_BYTES
    if len(body) > limit:
        raise ValueError("form body exceeds size limit")
    try:
        raw = base64.b64decode(body, validate=True) if is_base64_encoded else body.encode("utf-8")
        # API Gateway emits canonical Base64. Reject alternate encodings with
        # nonzero padding bits so one decoded request body has one transport
        # representation for replay controls and request diagnostics.
        if (is_base64_encoded
                and base64.b64encode(raw).decode("ascii") != body):
            raise ValueError("noncanonical base64 body")
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("form body exceeds size limit")
        form = raw.decode("utf-8")
    except (UnicodeError, binascii.Error, ValueError):
        raise ValueError("invalid form body encoding or size") from None

    # strict_parsing checks field structure, not malformed percent escapes.
    if _INVALID_ESCAPE.search(form):
        raise ValueError("invalid form percent encoding")
    try:
        pairs = parse_qsl(
            form, keep_blank_values=True, strict_parsing=True,
            encoding="utf-8", errors="strict", max_num_fields=MAX_FORM_FIELDS,
            separator="&",
        )
    except ValueError:
        raise ValueError("invalid form fields") from None

    fields = {}
    for name, value in pairs:
        # Bound and sanitize names before dictionary membership hashes them.
        # Values remain opaque and may legitimately contain whitespace or
        # controls such as newlines in transcription provider payloads.
        if (not name or len(name) > MAX_FORM_FIELD_NAME_BYTES
                or any(
                    char.isspace()
                    or unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                    or 0xFDD0 <= ord(char) <= 0xFDEF
                    or (ord(char) & 0xFFFF) in {0xFFFE, 0xFFFF}
                    for char in name
                )
                or len(name.encode("utf-8")) > MAX_FORM_FIELD_NAME_BYTES):
            raise ValueError(
                "form field names must be nonempty, bounded, and contain no whitespace or controls"
            )
        if name in fields:
            raise ValueError("form field names must be unique")
        fields[name] = value
    return fields
