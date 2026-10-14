"""Dependency-free operation and retry budgets from a trusted Lambda context."""

from collections.abc import Callable
from typing import Protocol

from zentomic.budget import resolve_operation_timeout
from zentomic.call_status_event import make_budgeted_status_conflict_retry_hook


class LambdaTimeContext(Protocol):
    """Only the runtime method required to bound an operation's duration."""

    def get_remaining_time_in_millis(self) -> int:
        """Return fresh invocation time remaining in integer milliseconds."""


def _lambda_time_reader(context: LambdaTimeContext) -> Callable[[], int]:
    """Bind the trusted runtime method without sampling its remaining time."""
    try:
        remaining_time = getattr(context, "get_remaining_time_in_millis", None)
    except Exception:
        raise ValueError("Lambda invocation time is unavailable") from None
    if not callable(remaining_time):
        raise ValueError(
            "context must provide a trusted callable get_remaining_time_in_millis"
        )

    def read() -> int:
        try:
            remaining_ms = remaining_time()
        except Exception:
            raise ValueError("Lambda invocation time is unavailable") from None
        if type(remaining_ms) is not int or remaining_ms < 0:
            raise ValueError("Lambda invocation time must be a nonnegative integer")
        return remaining_ms

    return read


def resolve_lambda_operation_timeout(
    *, context: LambdaTimeContext, deadline_ms: int, now_ms: int,
    maximum_ms: int, minimum_ms: int = 1, reserve_ms: int = 0,
    granularity_ms: int = 1,
) -> int | None:
    """Cap work against both a persisted call deadline and this invocation.

    Supply the handler's trusted context, not callback fields or cached runtime
    readings. Sample it once per decision and delegate all timeout policy to
    resolve_operation_timeout. Missing, unreadable or invalid runtime time
    raises ValueError instead of silently dropping the invocation bound.

    The result remains milliseconds; None means do not start the operation.
    The caller must supply a fresh trusted now_ms and apply the timeout to its
    dependency. This reads no wall clock, imports no AWS SDK, logs no context,
    performs no network calls and neither cancels work nor deploys anything.
    """
    remaining_ms = _lambda_time_reader(context)()
    return resolve_operation_timeout(
        deadline_ms=deadline_ms, now_ms=now_ms,
        maximum_ms=maximum_ms, minimum_ms=minimum_ms,
        reserve_ms=reserve_ms, granularity_ms=granularity_ms,
        invocation_remaining_ms=remaining_ms,
    )


def make_lambda_status_conflict_retry_hook(
    *, context: LambdaTimeContext, wait_ms: Callable[[int], None],
    randbelow: Callable[[int], int], reserve_ms: int,
    minimum_retry_attempt_ms: int = 1,
    base_delay_ms: int = 25, max_delay_ms: int = 400,
) -> Callable[[int], None]:
    """Bind lifecycle conflict backoff to one trusted Lambda invocation.

    Validate dependencies and policy at construction without sampling runtime.
    The shared retry hook reads fresh remaining time before and after each
    injected wait, preserves cleanup/attempt reserves and rejects increasing
    readings. Context failures use the same safe errors as timeout admission.

    Pass the result as before_retry to retry_call_status_event. Build a new hook
    for each invocation; never reuse its monotonic history across contexts.
    This does not bound the first attempt, enforce storage timeouts, sleep, or
    read a call deadline. Adapters must enforce those limits separately.
    """
    return make_budgeted_status_conflict_retry_hook(
        invocation_remaining_ms=_lambda_time_reader(context),
        wait_ms=wait_ms, randbelow=randbelow, reserve_ms=reserve_ms,
        minimum_retry_attempt_ms=minimum_retry_attempt_ms,
        base_delay_ms=base_delay_ms, max_delay_ms=max_delay_ms,
    )
