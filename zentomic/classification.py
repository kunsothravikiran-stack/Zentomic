"""Strict, offline admission of untrusted classifier output as a pending label."""

import json
from collections.abc import Collection


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

    Labels must be UTF-8 encodable and match exactly, without normalization.
    Invalid trusted configuration raises ValueError;
    malformed, oversized, ambiguous, or unknown model output returns None.
    An empty allowlist admits nothing. This neither invokes a model nor
    establishes confirmation: persist a pending label and confirm separately.
    """
    if (not isinstance(allowed_intents, Collection)
            or isinstance(allowed_intents, (str, bytes))):
        raise ValueError("allowed_intents must be a collection of nonblank strings")
    labels = tuple(allowed_intents)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("allowed_intents must contain only nonblank strings")
    try:
        for label in labels:
            label.encode("utf-8")
    except UnicodeEncodeError:
        # JSON escapes can decode to lone surrogates even in an ASCII response.
        # Reject unusable trusted labels before they can become pending state.
        raise ValueError("allowed_intents must contain only UTF-8 encodable strings") from None
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
    return intent if isinstance(intent, str) and intent in labels else None
