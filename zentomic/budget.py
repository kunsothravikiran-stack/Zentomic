"""Offline admission against trusted whole-call time and transition budgets."""

from dataclasses import dataclass
from typing import Literal

from zentomic.call_status import is_terminal_call_status


@dataclass(frozen=True)
class TransitionBudget:
    """Next action and the total number of admitted voice transitions."""

    action: Literal["continue", "hangup"]
    transitions: int


def resolve_transition_budget(
    *, transitions: int, max_transitions: int,
) -> TransitionBudget:
    """Admit one next voice step within a call-wide transition count limit.

    Count previously admitted transitions from trusted session state, not form
    fields. A successful admission increments once, including the last allowed
    transition. At or above the limit, hang up without changing the count;
    zero disables all transitions. Keep the same counter across voice steps.

    This complements, but does not replace, the whole-call deadline or per-step
    retry budgets. Authenticate and deduplicate first, then atomically persist
    the new count with the next step before emitting a redirect or other work.
    This pure helper does not persist, deduplicate, or end a real call.
    """
    for name, value in (("transitions", transitions), ("max_transitions", max_transitions)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if transitions >= max_transitions:
        return TransitionBudget("hangup", transitions)
    return TransitionBudget("continue", transitions + 1)


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


@dataclass(frozen=True)
class VoiceStepBudget:
    """Combined admission, actual time left, and next persisted count."""

    action: Literal["continue", "hangup"]
    remaining_ms: int
    transitions: int


def resolve_voice_step_budget(
    *, deadline_ms: int, now_ms: int, transitions: int, max_transitions: int,
    minimum_remaining_ms: int = 1, call_status: str | None = None,
) -> VoiceStepBudget:
    """Admit one voice step only when both whole-call budgets permit it.

    Validate both policies even when one denies work. Increment the count only
    for a combined admission; denial preserves it and reports actual time left.
    Inputs must come from trusted call state/configuration and a fresh clock.

    An optional stored call_status denies work for a terminal call without
    spending a transition, even with time/count headroom. None means no stored
    observation and preserves budget-only behavior, not proof of an active
    call. Unknown statuses raise ValueError, even when a budget denies work.
    All budget configuration is still validated for terminal calls.

    Use after authentication, call/step binding, replay and terminal-state
    checks. Use the stored parent/session status, not a child Dial result or
    caller-supplied field. Atomically check that status with the returned count
    and next step before emitting work; on a conflict reload and recompute.
    A hangup decision is denial of new work, not permission to repeat cleanup
    or send another operation to an ended call. This pure composition
    does not persist, read a clock, cancel work, or enforce operation timeouts.
    """
    terminal = False if call_status is None else is_terminal_call_status(call_status)
    time_budget = resolve_call_budget(
        deadline_ms=deadline_ms, now_ms=now_ms,
        minimum_remaining_ms=minimum_remaining_ms,
    )
    count_budget = resolve_transition_budget(
        transitions=transitions, max_transitions=max_transitions,
    )
    if terminal or time_budget.action == "hangup" or count_budget.action == "hangup":
        return VoiceStepBudget("hangup", time_budget.remaining_ms, transitions)
    return VoiceStepBudget("continue", time_budget.remaining_ms, count_budget.transitions)


def resolve_operation_timeout(
    *, deadline_ms: int, now_ms: int, maximum_ms: int,
    minimum_ms: int = 1, reserve_ms: int = 0, granularity_ms: int = 1,
    invocation_remaining_ms: int | None = None,
) -> int | None:
    """Compute an operation's timeout within the remaining call budget.

    Return None when less than minimum_ms remains after reserving reserve_ms
    for cleanup/transport. Cap the usable time at maximum_ms, then round down
    to a multiple of granularity_ms. Return None if that falls below minimum_ms.
    The result is still in milliseconds, never a count of timeout units. All
    limits are trusted integer milliseconds, not caller/model configuration.
    An optional invocation_remaining_ms also caps available time before the
    reserve is deducted. Supply a fresh nonnegative remaining duration from
    the trusted runtime, not a timestamp or the original invocation timeout.
    None preserves call-deadline-only behavior; zero admits no work.
    This does not reserve time in storage or enforce a timeout: an adapter must
    apply the result to its operation and recheck the fixed deadline on retry.
    Never round up when converting to a dependency's coarser timeout units.
    """
    for name, value in (("maximum_ms", maximum_ms), ("minimum_ms", minimum_ms),
                        ("granularity_ms", granularity_ms)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(reserve_ms) is not int or reserve_ms < 0:
        raise ValueError("reserve_ms must be a nonnegative integer")
    if invocation_remaining_ms is not None and (
        type(invocation_remaining_ms) is not int or invocation_remaining_ms < 0
    ):
        raise ValueError("invocation_remaining_ms must be a nonnegative integer or None")
    if minimum_ms > maximum_ms:
        raise ValueError("minimum_ms must not exceed maximum_ms")

    budget = resolve_call_budget(
        deadline_ms=deadline_ms, now_ms=now_ms,
        minimum_remaining_ms=minimum_ms + reserve_ms,
    )
    if budget.action == "hangup":
        return None
    remaining_ms = budget.remaining_ms
    if invocation_remaining_ms is not None:
        remaining_ms = min(remaining_ms, invocation_remaining_ms)
    if remaining_ms < minimum_ms + reserve_ms:
        return None
    usable_ms = min(maximum_ms, remaining_ms - reserve_ms)
    timeout_ms = (usable_ms // granularity_ms) * granularity_ms
    return timeout_ms if timeout_ms >= minimum_ms else None
