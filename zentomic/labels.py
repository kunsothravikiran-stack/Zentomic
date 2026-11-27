"""Shared validation for opaque intent labels."""

import unicodedata


MAX_INTENT_LABEL_BYTES = 256


def _valid_intent_label(value: object) -> bool:
    """Return whether value is a bounded, visibly representable intent label."""
    # UTF-8 uses at least one byte per character. Reject clearly oversized
    # values before scanning or encoding so direct policy use remains bounded.
    # Labels cross classifier and policy boundaries before routing. Require an
    # inert built-in string so programmatic adapters cannot execute overridden
    # length, whitespace, or encoding behavior during admission.
    if (type(value) is not str or len(value) > MAX_INTENT_LABEL_BYTES
            or not value.strip()):
        return False
    # Labels enter prompts, persistence keys, logs, and routing policy. Preserve
    # visible Unicode exactly, but reject characters that can hide or reorder a
    # label, split operational output, or cannot be safely interchanged.
    for character in value:
        codepoint = ord(character)
        if (unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                or 0xFDD0 <= codepoint <= 0xFDEF
                or (codepoint & 0xFFFF) in {0xFFFE, 0xFFFF}):
            return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(encoded) <= MAX_INTENT_LABEL_BYTES
