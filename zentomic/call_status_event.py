"""Authenticated call-leg lifecycle composition with injectable persistence."""

from collections.abc import Callable, Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.call_status import advance_call_status, is_terminal_call_status
from zentomic.call_status_store import (
    CallStatusSnapshot,
    CallStatusStore,
    StatusConflictError,
    _key as _status_key,
)


MAX_STATUS_CONFLICT_RETRIES = 8
MAX_STATUS_RETRY_DELAY_MS = 10_000


class StatusRetryBudgetExhaustedError(TimeoutError):
    """Raised before retrying when the trusted runtime budget is too small."""


def status_conflict_retry_delay_ms(
    retry_number: int, *, base_delay_ms: int = 25,
    max_delay_ms: int = 400,
) -> int:
    """Return a bounded exponential delay for a one-based conflict retry.

    This pure policy helper does not sleep or read a clock. Pass its result to a
    trusted, runtime-aware ``before_retry`` hook. Production adapters can choose
    a smaller jittered delay within this bound when many writers may contend.
    """
    if (type(retry_number) is not int
            or not 1 <= retry_number <= MAX_STATUS_CONFLICT_RETRIES):
        raise ValueError("retry_number must be an integer from 1 to 8")
    if (type(base_delay_ms) is not int
            or not 1 <= base_delay_ms <= MAX_STATUS_RETRY_DELAY_MS):
        raise ValueError("base_delay_ms must be an integer from 1 to 10000")
    if (type(max_delay_ms) is not int
            or not 1 <= max_delay_ms <= MAX_STATUS_RETRY_DELAY_MS):
        raise ValueError("max_delay_ms must be an integer from 1 to 10000")
    if base_delay_ms > max_delay_ms:
        raise ValueError("base_delay_ms must not exceed max_delay_ms")
    return min(max_delay_ms, base_delay_ms * (1 << (retry_number - 1)))


def jittered_status_conflict_retry_delay_ms(
    retry_number: int, *, randbelow: Callable[[int], int],
    base_delay_ms: int = 25, max_delay_ms: int = 400,
) -> int:
    """Sample a full-jitter delay within the exponential retry ceiling.

    ``randbelow`` is a trusted injected function with the same contract as
    ``secrets.randbelow``: given an exclusive positive upper bound, it returns
    an integer from zero up to that bound. The callback receives no request or
    storage data. This helper validates all policy inputs before sampling, calls
    the sampler exactly once, and rejects malformed sampler results.
    """
    if not callable(randbelow):
        raise ValueError("randbelow must be callable")
    ceiling_ms = status_conflict_retry_delay_ms(
        retry_number,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
    )
    upper_bound = ceiling_ms + 1
    delay_ms = randbelow(upper_bound)
    if type(delay_ms) is not int or not 0 <= delay_ms < upper_bound:
        raise ValueError("randbelow returned an invalid retry delay")
    return delay_ms


def budgeted_status_conflict_retry_delay_ms(
    retry_number: int, *, invocation_remaining_ms: int, reserve_ms: int,
    randbelow: Callable[[int], int], base_delay_ms: int = 25,
    max_delay_ms: int = 400, minimum_retry_attempt_ms: int = 1,
) -> int | None:
    """Sample full jitter while reserving cleanup and the next retry attempt.

    Return ``None`` when the trusted runtime's fresh remaining duration cannot
    preserve both ``reserve_ms`` and ``minimum_retry_attempt_ms``. Otherwise,
    sample from zero through the smaller of the exponential ceiling and the
    duration left after both reservations. This pure helper neither reads the
    runtime clock nor sleeps.
    """
    if not callable(randbelow):
        raise ValueError("randbelow must be callable")
    ceiling_ms = status_conflict_retry_delay_ms(
        retry_number,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
    )
    if type(invocation_remaining_ms) is not int or invocation_remaining_ms < 0:
        raise ValueError("invocation_remaining_ms must be a nonnegative integer")
    if type(reserve_ms) is not int or reserve_ms < 0:
        raise ValueError("reserve_ms must be a nonnegative integer")
    if type(minimum_retry_attempt_ms) is not int or minimum_retry_attempt_ms < 1:
        raise ValueError("minimum_retry_attempt_ms must be a positive integer")
    available_ms = invocation_remaining_ms - reserve_ms - minimum_retry_attempt_ms
    if available_ms < 0:
        return None
    upper_bound = min(ceiling_ms, available_ms) + 1
    delay_ms = randbelow(upper_bound)
    if type(delay_ms) is not int or not 0 <= delay_ms < upper_bound:
        raise ValueError("randbelow returned an invalid retry delay")
    return delay_ms


def make_budgeted_status_conflict_retry_hook(
    *, invocation_remaining_ms: Callable[[], int],
    wait_ms: Callable[[int], None], randbelow: Callable[[int], int],
    reserve_ms: int, minimum_retry_attempt_ms: int = 1,
    base_delay_ms: int = 25, max_delay_ms: int = 400,
) -> Callable[[int], None]:
    """Build a retry hook that refreshes runtime budget before every wait.

    The trusted callbacks receive no request or storage data. Configuration is
    validated when the hook is built, without reading the runtime or sampling
    jitter. Each invocation reads remaining time before and after the wait,
    samples one budgeted delay, and passes that delay to ``wait_ms``. The second
    reading may stay equal or decrease, but cannot increase. Insufficient runtime
    raises ``StatusRetryBudgetExhaustedError`` before sampling or after waiting.
    """
    if not callable(invocation_remaining_ms):
        raise ValueError("invocation_remaining_ms must be callable")
    if not callable(wait_ms):
        raise ValueError("wait_ms must be callable")
    if not callable(randbelow):
        raise ValueError("randbelow must be callable")
    status_conflict_retry_delay_ms(
        1, base_delay_ms=base_delay_ms, max_delay_ms=max_delay_ms,
    )
    if type(reserve_ms) is not int or reserve_ms < 0:
        raise ValueError("reserve_ms must be a nonnegative integer")
    if type(minimum_retry_attempt_ms) is not int or minimum_retry_attempt_ms < 1:
        raise ValueError("minimum_retry_attempt_ms must be a positive integer")

    def before_retry(retry_number: int) -> None:
        remaining_before_wait_ms = invocation_remaining_ms()
        delay_ms = budgeted_status_conflict_retry_delay_ms(
            retry_number,
            invocation_remaining_ms=remaining_before_wait_ms,
            reserve_ms=reserve_ms,
            minimum_retry_attempt_ms=minimum_retry_attempt_ms,
            randbelow=randbelow,
            base_delay_ms=base_delay_ms,
            max_delay_ms=max_delay_ms,
        )
        if delay_ms is None:
            raise StatusRetryBudgetExhaustedError(
                "insufficient runtime for status retry"
            )
        wait_ms(delay_ms)
        remaining_after_wait_ms = invocation_remaining_ms()
        if (type(remaining_after_wait_ms) is not int
                or remaining_after_wait_ms < 0):
            raise ValueError(
                "invocation_remaining_ms must be a nonnegative integer"
            )
        if remaining_after_wait_ms > remaining_before_wait_ms:
            raise ValueError(
                "invocation_remaining_ms must not increase after waiting"
            )
        if remaining_after_wait_ms < reserve_ms + minimum_retry_attempt_ms:
            raise StatusRetryBudgetExhaustedError(
                "insufficient runtime for status retry"
            )

    return before_retry


def _validate_status_snapshot(value: Any) -> CallStatusSnapshot:
    """Keep malformed adapter results from crossing the orchestration boundary."""
    if type(value) is not CallStatusSnapshot:
        raise ValueError("store must return an exact CallStatusSnapshot")
    if type(value.revision) is not int or value.revision < 0:
        raise ValueError("store returned an invalid CallStatusSnapshot")
    if (value.status is None) != (value.revision == 0):
        raise ValueError("store returned an invalid CallStatusSnapshot")
    if value.status is not None:
        try:
            is_terminal_call_status(value.status)
        except ValueError:
            raise ValueError("store returned an invalid CallStatusSnapshot") from None
    return value


def advance_call_status_event(
    current_status: str | None, event: Mapping[str, Any], *, public_url: str,
    validator: SignatureValidator, expected_account_sid: str, expected_call_sid: str,
) -> str:
    """Return proposed state after authenticating a single-leg status callback.

    Current state, expected identifiers, URL and validator must come from trusted
    workspace-scoped configuration/session state, never from caller fields.
    Reject invalid stored state before invoking the injected validator. Even an
    already terminal state requires authenticated, identity-bound input with a
    valid CallStatus; DialCallStatus cannot substitute for it.

    This composes existing transport, signature, identity and monotonic status
    policies. It does not load/save state, acknowledge a webhook, prevent replay,
    or authorize cleanup. Conditionally persist against the loaded state version;
    reload and recompute on conflict. No request fields are returned or logged.
    """
    if current_status is not None:
        is_terminal_call_status(current_status)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return advance_call_status(current_status, fields.get("CallStatus"))


def _prepare_status_observation(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, workspace_id: str,
    store: CallStatusStore,
) -> tuple[Callable[..., Any], tuple[str, str], str, CallStatusSnapshot,
           CallStatusSnapshot]:
    """Validate trusted inputs, authenticate, load, and compute one write."""
    load = getattr(store, "load", None)
    observe = getattr(store, "observe", None)
    if not callable(load) or not callable(observe):
        raise ValueError("store must provide trusted callable load and observe methods")
    key = _status_key(workspace_id, expected_call_sid)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    incoming_status = fields.get("CallStatus")
    is_terminal_call_status(incoming_status)
    snapshot = _validate_status_snapshot(load(*key))
    expected_status = advance_call_status(snapshot.status, incoming_status)
    expected = CallStatusSnapshot(
        expected_status,
        snapshot.revision + (expected_status != snapshot.status),
    )
    return observe, key, incoming_status, snapshot, expected


def _persist_status_observation(
    observe: Callable[..., Any], key: tuple[str, str], incoming_status: str,
    snapshot: CallStatusSnapshot, expected: CallStatusSnapshot,
) -> CallStatusSnapshot:
    """Perform and validate the conditional write for a prepared observation."""
    saved = _validate_status_snapshot(observe(
        *key, incoming_status, expected_revision=snapshot.revision,
    ))
    if saved != expected:
        raise ValueError("store returned an unexpected CallStatusSnapshot")
    return saved


def observe_call_status_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, workspace_id: str,
    store: CallStatusStore,
) -> CallStatusSnapshot:
    """Authenticate, load and conditionally store one lifecycle observation.

    Workspace, account and call identifiers must come from trusted session state.
    The exact workspace/call key and store boundary are validated before request
    authentication. Invalid, unsigned or identity-mismatched callbacks never
    reach storage. After authentication, the incoming status is validated before
    loading the current snapshot and applying the store's conditional write.

    A concurrent writer raises ``StatusConflictError``; call this function again
    to authenticate, reload and recompute. Duplicate or delayed observations may
    return the unchanged snapshot and do not authorize repeated side effects.
    Store failures propagate and fail closed. This helper performs no workspace
    authorization, callback acknowledgement, cleanup or provider I/O.
    """
    prepared = _prepare_status_observation(
        event,
        public_url=public_url,
        validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
        workspace_id=workspace_id,
        store=store,
    )
    return _persist_status_observation(*prepared)


def retry_call_status_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, workspace_id: str,
    store: CallStatusStore, max_conflict_retries: int = 2,
    before_retry: Callable[[int], None] | None = None,
) -> CallStatusSnapshot:
    """Retry a bounded number of conditional-write conflicts.

    Every attempt runs the complete authenticated observation again, including
    trusted configuration validation, signature verification, call binding and
    a fresh load. Only ``StatusConflictError`` raised by the conditional write is
    retried. The same exception from authentication, loading, or validation is an
    adapter failure and propagates immediately. Invalid callbacks, malformed
    adapter results, capacity failures and other storage errors also fail closed.
    When another attempt remains, an optional trusted hook receives its one-based
    retry number. The hook can provide bounded, runtime-aware backoff without
    receiving callback data or the storage exception. Hook failures propagate
    and prevent the next attempt.

    A successful or unchanged terminal snapshot is still not permission to
    repeat cleanup. A production adapter must separately make effects
    idempotent and choose an appropriate contention/backoff policy.
    """
    if (type(max_conflict_retries) is not int
            or not 0 <= max_conflict_retries <= MAX_STATUS_CONFLICT_RETRIES):
        raise ValueError("max_conflict_retries must be an integer from 0 to 8")
    if before_retry is not None and not callable(before_retry):
        raise ValueError("before_retry must be callable or None")
    for attempt in range(max_conflict_retries + 1):
        prepared = _prepare_status_observation(
            event,
            public_url=public_url,
            validator=validator,
            expected_account_sid=expected_account_sid,
            expected_call_sid=expected_call_sid,
            workspace_id=workspace_id,
            store=store,
        )
        try:
            return _persist_status_observation(*prepared)
        except StatusConflictError:
            if attempt == max_conflict_retries:
                raise
            if before_retry is not None:
                before_retry(attempt + 1)
    raise AssertionError("unreachable")
