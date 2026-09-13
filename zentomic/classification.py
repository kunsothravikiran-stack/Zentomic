"""Strict, offline admission of untrusted classifier output as a pending label."""

import json
from collections.abc import Collection

from zentomic.labels import _valid_intent_label


MAX_RESPONSE_BYTES = 4096


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate classifier field")
        result[key] = value
    return result


def parse_intent_response(
    response: str | None, *, allowed_intents: Collection[str],
) -> str | None:
    """Accept exactly {\"intent\": <allowlisted string>} or return None.

    Labels must be nonblank UTF-8 strings of at most 256 bytes and match
    exactly, without normalization.
    Invalid trusted configuration raises ValueError;
    malformed, oversized, ambiguous, or unknown model output returns None.
    An empty allowlist admits nothing. This neither invokes a model nor
    establishes confirmation: persist a pending label and confirm separately.
    """
    if (not isinstance(allowed_intents, Collection)
            or isinstance(allowed_intents, (str, bytes))):
        raise ValueError("allowed_intents must be a collection of nonblank strings")
    labels = tuple(allowed_intents)
    if any(not _valid_intent_label(label) for label in labels):
        # JSON escapes can decode to lone surrogates even in an ASCII response.
        # Reject unusable trusted labels before they can become pending state.
        raise ValueError(
            "allowed_intents must contain only nonblank UTF-8 strings "
            "of at most 256 bytes"
        )
    if not isinstance(response, str) or len(response) > MAX_RESPONSE_BYTES:
        return None
    try:
        if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
            return None
        payload = json.loads(response, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"intent"}:
        return None
    intent = payload["intent"]
    return intent if _valid_intent_label(intent) and intent in labels else None
