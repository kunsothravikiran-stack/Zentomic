"""Pure single-digit IVR menu routing, without telephony or persistence."""

from collections.abc import Mapping


def _valid_target(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def resolve_dtmf(
    digits: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
) -> str:
    """Resolve one ASCII digit to an opaque, workspace-local target identifier.

    Missing, malformed, or unmapped input returns the configured fallback.
    Configuration errors raise ValueError, even when the input would fall back.
    Target identifiers are returned unchanged; they are not phone numbers to
    dial. Callers must authorize and resolve them within the current workspace.
    This function never mutates the menu, logs inputs, or contacts services.
    """
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    if not _valid_target(fallback_target):
        raise ValueError("fallback_target must be a nonblank string")

    # Snapshot the caller's menu and validate every entry before routing.
    menu = dict(routes)
    for digit, target in menu.items():
        if not isinstance(digit, str) or len(digit) != 1 or digit not in "0123456789":
            raise ValueError("route keys must be single ASCII digits")
        if not _valid_target(target):
            raise ValueError("route targets must be nonblank strings")

    if not isinstance(digits, str) or len(digits) != 1 or digits not in "0123456789":
        return fallback_target
    return menu.get(digits, fallback_target)
