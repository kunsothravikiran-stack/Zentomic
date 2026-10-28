"""Shared validation for opaque intent labels."""


MAX_INTENT_LABEL_BYTES = 256


def _valid_intent_label(value: object) -> bool:
    """Return whether value is a bounded, nonblank UTF-8 intent label."""
    # UTF-8 uses at least one byte per character. Reject clearly oversized
    # values before scanning or encoding so direct policy use remains bounded.
    if (not isinstance(value, str) or len(value) > MAX_INTENT_LABEL_BYTES
            or not value.strip()):
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(encoded) <= MAX_INTENT_LABEL_BYTES
