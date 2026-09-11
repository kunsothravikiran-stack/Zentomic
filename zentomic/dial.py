"""Offline outcome policy for hosted Number forwarding callbacks."""

from dataclasses import dataclass
from typing import Literal

from zentomic.routing import _valid_target


_FAILED_STATUSES = frozenset({"busy", "no-answer", "failed"})
_TERMINAL_STATUSES = frozenset({"completed", "canceled"})


@dataclass(frozen=True)
class DialDecision:
    """Next action and optional opaque, workspace-scoped fallback target."""

    action: Literal["fallback", "hangup"]
    target: str | None


def resolve_dial_result(
    status: str, *, fallback_target: str, fallback_used: bool = False,
) -> DialDecision:
    """Allow one fallback after an unsuccessful Number forwarding attempt.

    Read status from authenticated DialCallStatus, not CallStatus. This policy
    does not support Conference's answered status or Number status callbacks.
    Persist fallback_used atomically in trusted session state before executing
    a fallback; never take it from the request or reset it on each callback.
    This helper does not authenticate, deduplicate, persist, render, or dial.
    """
    if not _valid_target(fallback_target):
        raise ValueError("fallback_target must be a nonblank UTF-8 encodable string")
    if type(fallback_used) is not bool:
        raise ValueError("fallback_used must be a boolean")
    if not isinstance(status, str) or status not in _FAILED_STATUSES | _TERMINAL_STATUSES:
        raise ValueError("unsupported Number dial result status")

    if status in _FAILED_STATUSES and not fallback_used:
        return DialDecision("fallback", fallback_target)
    return DialDecision("hangup", None)
