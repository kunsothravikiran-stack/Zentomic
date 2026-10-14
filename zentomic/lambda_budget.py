"""Dependency-free operation and retry budgets from a trusted Lambda context."""

from collections.abc import Callable
from typing import Protocol

from zentomic.budget import resolve_call_budget, resolve_operation_timeout
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


def _call_remaining_time_reader(
    *, runtime_reader: Callable[[], int], deadline_ms: int,
    now_ms: Callable[[], int],
) -> Callable[[], int]:
    """Keep independent clock histories for one fixed call and invocation."""
    resolve_call_budget(deadline_ms=deadline_ms, now_ms=0)
    if not callable(now_ms):
        raise ValueError("now_ms must be a trusted callable")
    last_runtime_ms: int | None = None
    last_now_ms: int | None = None

    def read() -> int:
        nonlocal last_runtime_ms, last_now_ms
        runtime_ms = runtime_reader()
        try:
            current_ms = now_ms()
        except Exception:
            raise ValueError("call budget clock is unavailable") from None
        if type(current_ms) is not int or current_ms < 0:
            raise ValueError("call budget clock must be a nonnegative integer")
        if last_now_ms is not None and current_ms < last_now_ms:
            raise ValueError("call budget clock must not move backwards")
        if last_runtime_ms is not None and runtime_ms > last_runtime_ms:
            raise ValueError("Lambda invocation time must not increase")
        call_remaining_ms = resolve_call_budget(
            deadline_ms=deadline_ms, now_ms=current_ms,
        ).remaining_ms
        last_runtime_ms, last_now_ms = runtime_ms, current_ms
        return min(runtime_ms, call_remaining_ms)

    return read


def make_lambda_operation_timeout_resolver(
    *, context: LambdaTimeContext, deadline_ms: int,
    now_ms: Callable[[], int], maximum_ms: int, minimum_ms: int = 1,
    reserve_ms: int = 0, granularity_ms: int = 1,
) -> Callable[[], int | None]:
    """Bind timeout admission to one invocation and a fixed call deadline.

    Validate the policy without sampling either clock. Each decision reads
    fresh trusted time and rejects wall-clock rollback or runtime increases
    independently, even when the other clock would hide the reset. Use a new
    resolver per invocation, retaining the persisted call deadline. This bounds
    first attempts as well as retries; unlike a retry hook it performs no wait.

    The result is milliseconds, or None to skip work. The adapter must apply
    the timeout immediately and handle denial/failure. This does not persist
    admission, check terminal status, cancel work, or contact any service.
    """
    remaining_time = _call_remaining_time_reader(
        runtime_reader=_lambda_time_reader(context),
        deadline_ms=deadline_ms, now_ms=now_ms,
    )
    # Use synthetic exhausted time solely to validate every static limit.
    resolve_operation_timeout(
        deadline_ms=0, now_ms=0, maximum_ms=maximum_ms,
        minimum_ms=minimum_ms, reserve_ms=reserve_ms,
        granularity_ms=granularity_ms, invocation_remaining_ms=0,
    )

    def resolve() -> int | None:
        remaining_ms = remaining_time()
        return resolve_operation_timeout(
            deadline_ms=remaining_ms, now_ms=0, maximum_ms=maximum_ms,
            minimum_ms=minimum_ms, reserve_ms=reserve_ms,
            granularity_ms=granularity_ms,
        )

    return resolve


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
    deadline_ms: int | None = None, now_ms: Callable[[], int] | None = None,
) -> Callable[[int], None]:
    """Bind lifecycle conflict backoff to one trusted Lambda invocation.

    Validate dependencies and policy at construction without sampling runtime.
    The shared retry hook reads fresh remaining time before and after each
    injected wait, preserves cleanup/attempt reserves and rejects increasing
    readings. Context failures use the same safe errors as timeout admission.

    Pass the result as before_retry to retry_call_status_event. Build a new hook
    for each invocation; never reuse its monotonic history across contexts.
    Optionally supply both a fixed persisted deadline_ms and a trusted now_ms
    clock in the same timestamp domain. Cap each runtime reading by remaining
    call time before reserving cleanup/attempt time. Reject clock rollback and
    runtime increases independently, so one clock cannot hide the other's reset.
    Omit both to preserve invocation-only behavior. Neither is callback data.
    This does not bound the first attempt, enforce storage timeouts or sleep.
    Adapters must enforce those limits separately.
    """
    remaining_time = _lambda_time_reader(context)
    if (deadline_ms is None) != (now_ms is None):
        raise ValueError("deadline_ms and now_ms must be supplied together")
    if deadline_ms is not None:
        remaining_time = _call_remaining_time_reader(
            runtime_reader=remaining_time, deadline_ms=deadline_ms, now_ms=now_ms,
        )

    return make_budgeted_status_conflict_retry_hook(
        invocation_remaining_ms=remaining_time,
        wait_ms=wait_ms, randbelow=randbelow, reserve_ms=reserve_ms,
        minimum_retry_attempt_ms=minimum_retry_attempt_ms,
        base_delay_ms=base_delay_ms, max_delay_ms=max_delay_ms,
    )
