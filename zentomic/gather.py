"""Bounded, offline retry policy for single-digit IVR collection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from zentomic.routing import _snapshot_dtmf_routes, resolve_dtmf


@dataclass(frozen=True)
class GatherDecision:
    """Next action, optional destination, and updated completed-attempt count."""

    action: Literal["route", "retry", "fallback", "hangup"]
    target: str | None
    attempts: int


def _validate_gather_configuration(
    routes: Mapping[str, str], *, fallback_target: str, attempts: int,
    max_attempts: int, hangup_digit: str | None,
) -> dict[str, str]:
    """Return a private menu snapshot after validating all trusted state."""
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    menu = _snapshot_dtmf_routes(routes)
    # Missing input exercises every route and fallback configuration check
    # without selecting a destination or consuming an attempt.
    resolve_dtmf(None, menu, fallback_target=fallback_target)
    if hangup_digit is not None:
        if (not isinstance(hangup_digit, str) or len(hangup_digit) != 1
                or hangup_digit not in "0123456789"):
            raise ValueError("hangup_digit must be a single ASCII digit or None")
        if hangup_digit in menu:
            raise ValueError("hangup_digit must not overlap a configured route")
    return menu


def _resolve_validated_gather(
    digits: str | None, menu: Mapping[str, str], *, fallback_target: str,
    attempts: int, max_attempts: int, hangup_digit: str | None,
) -> GatherDecision:
    """Resolve input after the caller validated and privately copied policy."""
    valid_digit = (
        isinstance(digits, str) and len(digits) == 1
        and digits in "0123456789"
    )
    target = menu.get(digits, fallback_target) if valid_digit else fallback_target
    if attempts >= max_attempts:
        return GatherDecision("fallback", fallback_target, attempts)

    completed = attempts + 1
    if hangup_digit is not None and valid_digit and digits == hangup_digit:
        return GatherDecision("hangup", None, completed)
    # Membership distinguishes a configured reception key from invalid input
    # that happens to resolve to the same fallback destination.
    if valid_digit and digits in menu:
        return GatherDecision("route", target, completed)
    if completed < max_attempts:
        return GatherDecision("retry", None, completed)
    return GatherDecision("fallback", fallback_target, completed)


def resolve_gather(
    digits: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
    attempts: int,
    max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Consume one completed collection attempt without an unbounded retry loop.

    `attempts` is the number already completed, before this input. Missing,
    malformed, or unmapped input retries only while the budget allows it.
    An exhausted budget always falls back, even for otherwise valid input.
    Every configuration entry is validated before making any decision.
    An optional unused ASCII digit explicitly ends the call with no target.
    It consumes one attempt and is honored only while the budget is active.

    This pure helper does not collect digits, persist state, or send TwiML.
    Adapters must atomically deduplicate callbacks and persist the returned
    count in trusted, workspace-scoped call-session state before retrying.
    """
    menu = _validate_gather_configuration(
        routes, fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
    return _resolve_validated_gather(
        digits, menu, fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
