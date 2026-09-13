"""Pure single-digit IVR menu routing, without telephony or persistence."""

from collections.abc import Mapping
from itertools import islice


MAX_DTMF_ROUTES = 10
MAX_TARGET_IDENTIFIER_BYTES = 256


def _valid_target(value: object) -> bool:
    """Reject oversized or unusable identifiers at the persistence boundary."""
    # UTF-8 uses at least one byte per character. Bound work before stripping
    # or encoding so an accidentally huge trusted value is rejected cheaply.
    if (not isinstance(value, str) or len(value) > MAX_TARGET_IDENTIFIER_BYTES
            or not value.strip()):
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(encoded) <= MAX_TARGET_IDENTIFIER_BYTES


def _snapshot_dtmf_routes(routes: Mapping[str, str]) -> dict[str, str]:
    """Copy at most one single-digit menu without unbounded iteration."""
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    # There are only ten valid ASCII digit keys. Read one extra key before
    # copying values so an invalid, unexpectedly large mapping is rejected
    # without allocating or iterating over all of it.
    keys = tuple(islice(routes, MAX_DTMF_ROUTES + 1))
    if len(keys) > MAX_DTMF_ROUTES:
        raise ValueError("routes must contain at most 10 entries")
    return {key: routes[key] for key in keys}


def resolve_dtmf(
    digits: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
) -> str:
    """Resolve one ASCII digit to an opaque, workspace-local target identifier.

    Missing, malformed, or unmapped input returns the configured fallback.
    Configuration errors raise ValueError, even when the input would fall back.
    Targets must be UTF-8 encodable and are returned unchanged, not repaired
    or normalized; they are not phone numbers to
    dial. Callers must authorize and resolve them within the current workspace.
    This function never mutates the menu, logs inputs, or contacts services.
    """
    if not _valid_target(fallback_target):
        raise ValueError("fallback_target must be a nonblank UTF-8 string of at most 256 bytes")

    # Snapshot the caller's menu and validate every entry before routing.
    menu = _snapshot_dtmf_routes(routes)
    for digit, target in menu.items():
        if not isinstance(digit, str) or len(digit) != 1 or digit not in "0123456789":
            raise ValueError("route keys must be single ASCII digits")
        if not _valid_target(target):
            raise ValueError("route targets must be nonblank UTF-8 strings of at most 256 bytes")

    if not isinstance(digits, str) or len(digits) != 1 or digits not in "0123456789":
        return fallback_target
    return menu.get(digits, fallback_target)
