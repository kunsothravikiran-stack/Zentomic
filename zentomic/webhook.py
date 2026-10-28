"""Bounded, offline decoding for form-encoded webhook bodies."""

import base64
import binascii
import re
from urllib.parse import parse_qsl


MAX_BODY_BYTES = 16 * 1024
MAX_FORM_FIELDS = 128
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
    if not isinstance(body, str) or type(is_base64_encoded) is not bool:
        raise ValueError("body must be a string and encoding flag must be a boolean")

    limit = _MAX_BASE64_CHARS if is_base64_encoded else MAX_BODY_BYTES
    if len(body) > limit:
        raise ValueError("form body exceeds size limit")
    try:
        raw = base64.b64decode(body, validate=True) if is_base64_encoded else body.encode("utf-8")
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
        if not name or name in fields:
            raise ValueError("form field names must be nonempty and unique")
        fields[name] = value
    return fields
