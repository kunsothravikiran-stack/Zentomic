"""Dependency-free timeout admission using a trusted Lambda runtime context."""

from typing import Protocol

from zentomic.budget import resolve_operation_timeout


class LambdaTimeContext(Protocol):
    """Only the runtime method required to bound an operation's duration."""

    def get_remaining_time_in_millis(self) -> int:
        """Return fresh invocation time remaining in integer milliseconds."""


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
    try:
        remaining_time = getattr(context, "get_remaining_time_in_millis", None)
    except Exception:
        raise ValueError("Lambda invocation time is unavailable") from None
    if not callable(remaining_time):
        raise ValueError(
            "context must provide a trusted callable get_remaining_time_in_millis"
        )
    try:
        remaining_ms = remaining_time()
    except Exception:
        raise ValueError("Lambda invocation time is unavailable") from None
    if type(remaining_ms) is not int or remaining_ms < 0:
        raise ValueError("Lambda invocation time must be a nonnegative integer")
    return resolve_operation_timeout(
        deadline_ms=deadline_ms, now_ms=now_ms,
        maximum_ms=maximum_ms, minimum_ms=minimum_ms,
        reserve_ms=reserve_ms, granularity_ms=granularity_ms,
        invocation_remaining_ms=remaining_ms,
    )
