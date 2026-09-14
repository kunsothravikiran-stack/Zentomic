"""Authenticated call-leg lifecycle composition with injectable persistence."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.call_status import advance_call_status, is_terminal_call_status
from zentomic.call_status_store import (
    CallStatusSnapshot,
    CallStatusStore,
    _key as _status_key,
)


def _validate_status_snapshot(value: Any) -> CallStatusSnapshot:
    """Keep malformed adapter results from crossing the orchestration boundary."""
    if type(value) is not CallStatusSnapshot:
        raise ValueError("store must return an exact CallStatusSnapshot")
    if type(value.revision) is not int or value.revision < 0:
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
    saved = observe(
        *key, incoming_status, expected_revision=snapshot.revision,
    )
    return _validate_status_snapshot(saved)
