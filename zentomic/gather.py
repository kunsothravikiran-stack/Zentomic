"""Bounded, offline retry policy for single-digit IVR collection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from zentomic.routing import resolve_dtmf


@dataclass(frozen=True)
class GatherDecision:
    """Next action, optional destination, and updated completed-attempt count."""

    action: Literal["route", "retry", "fallback", "hangup"]
    target: str | None
    attempts: int


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
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")

    menu = dict(routes)
    target = resolve_dtmf(digits, menu, fallback_target=fallback_target)
    if hangup_digit is not None:
        if (not isinstance(hangup_digit, str) or len(hangup_digit) != 1
                or hangup_digit not in "0123456789"):
            raise ValueError("hangup_digit must be a single ASCII digit or None")
        if hangup_digit in menu:
            raise ValueError("hangup_digit must not overlap a configured route")
    if attempts >= max_attempts:
        return GatherDecision("fallback", fallback_target, attempts)

    completed = attempts + 1
    if hangup_digit is not None and isinstance(digits, str) and digits == hangup_digit:
        return GatherDecision("hangup", None, completed)
    # Membership distinguishes a configured reception key from invalid input
    # that happens to resolve to the same fallback destination.
    if isinstance(digits, str) and digits in menu:
        return GatherDecision("route", target, completed)
    if completed < max_attempts:
        return GatherDecision("retry", None, completed)
    return GatherDecision("fallback", fallback_target, completed)
