"""Strict, offline classification of a single call leg's lifecycle status."""


_ACTIVE_STATUSES = frozenset({"queued", "ringing", "in-progress"})
_TERMINAL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})
_KNOWN_STATUSES = _ACTIVE_STATUSES | _TERMINAL_STATUSES
_ACTIVE_PROGRESS = {"queued": 0, "ringing": 1, "in-progress": 2}
_MAX_STATUS_CHARACTERS = max(map(len, _KNOWN_STATUSES))


def is_terminal_call_status(status: str) -> bool:
    """Classify an exact CallStatus value; reject unknown values with ValueError.

    A terminal status means this leg ended, not that a human was reached or the
    business task succeeded. Read CallStatus from a signature-validated callback
    bound to the expected account and call leg, not DialCallStatus or the event
    subscription names initiated/answered. Those are different contracts.

    This stateless helper does not order callbacks, persist or deduplicate state,
    end calls, or authorize cleanup. A future adapter must atomically preserve a
    terminal state against delayed active events and keep parent/child legs
    separate. Unknown input is neither assumed active nor assumed terminal.
    """
    # Reject oversized input before hashing it for set membership. Provider
    # callback bodies are bounded elsewhere, but this helper is also public.
    if (not isinstance(status, str) or len(status) > _MAX_STATUS_CHARACTERS
            or status not in _KNOWN_STATUSES):
        raise ValueError("unsupported call status")
    return status in _TERMINAL_STATUSES


def advance_call_status(current_status: str | None, incoming_status: str) -> str:
    """Choose a monotonic application state for one authenticated call leg.

    None means no stored observation. Active observations may skip states but
    never regress; the first stored terminal outcome wins, even against another
    terminal outcome. Validate both values before applying this policy.

    This is not provider event ordering, reconciliation, or atomic persistence.
    Load trusted workspace/call-scoped state and conditionally save against its
    version; on a write conflict, reload and recompute. A returned terminal value
    does not grant permission to repeat cleanup or any other side effect.
    """
    incoming_terminal = is_terminal_call_status(incoming_status)
    if current_status is None:
        return incoming_status
    if is_terminal_call_status(current_status):
        return current_status
    if incoming_terminal or _ACTIVE_PROGRESS[incoming_status] > _ACTIVE_PROGRESS[current_status]:
        return incoming_status
    return current_status
