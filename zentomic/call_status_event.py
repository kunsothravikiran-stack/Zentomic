"""Authenticated call-leg lifecycle composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.call_status import advance_call_status, is_terminal_call_status


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
