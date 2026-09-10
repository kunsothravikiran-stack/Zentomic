"""Offline admission against a fixed, trusted whole-call deadline."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CallBudget:
    """Whether another step may start, and the remaining milliseconds."""

    action: Literal["continue", "hangup"]
    remaining_ms: int


def resolve_call_budget(
    *, deadline_ms: int, now_ms: int, minimum_remaining_ms: int = 1,
) -> CallBudget:
    """Stop admitting work at or after a previously persisted call deadline.

    Both values must be nonnegative integer timestamps in the same clock domain.
    Set the deadline once in trusted, workspace-scoped call-session state; never
    rebuild it from each callback or accept either value from request fields.
    Inject the current time from a trusted clock, not a cached invocation time.

    minimum_remaining_ms is a trusted positive integer admission threshold for
    the next operation, including any desired safety margin. The default keeps
    admitting work until expiry. Less remaining time returns hangup, while
    remaining_ms still reports the actual time left, not zero or a reservation.

    This pure check does not read a clock, persist state, cancel running work,
    prevent replay, or end a real call. Clock rollback can extend admission;
    adapters must handle clock consistency and enforce active-operation limits.
    """
    for name, value in (("deadline_ms", deadline_ms), ("now_ms", now_ms)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer timestamp")
    if type(minimum_remaining_ms) is not int or minimum_remaining_ms < 1:
        raise ValueError("minimum_remaining_ms must be a positive integer")
    remaining = max(0, deadline_ms - now_ms)
    action = "continue" if remaining >= minimum_remaining_ms else "hangup"
    return CallBudget(action, remaining)
