"""Strict, offline classification of a single call leg's lifecycle status."""


_ACTIVE_STATUSES = frozenset({"queued", "ringing", "in-progress"})
_TERMINAL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})
_KNOWN_STATUSES = _ACTIVE_STATUSES | _TERMINAL_STATUSES


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
    if not isinstance(status, str) or status not in _KNOWN_STATUSES:
        raise ValueError("unsupported call status")
    return status in _TERMINAL_STATUSES
