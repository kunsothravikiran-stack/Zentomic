"""Process-local final voicemail status storage for adapter development."""

from threading import Lock
from typing import Protocol

from zentomic.call_status_store import _key as _call_key
from zentomic.voicemail import MAX_VOICEMAIL_SECONDS
from zentomic.voicemail_status import (
    VoicemailRecordingStatus,
    _validate_recording_sid,
)


class VoicemailStatusConflictError(ValueError):
    """A final status already exists with different sanitized metadata."""


class VoicemailStatusCapacityError(ValueError):
    """A new recording cannot be stored without exceeding the local limit."""


class VoicemailStatusStore(Protocol):
    """Minimal immutable final-status boundary supplied by an adapter."""

    def load(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> VoicemailRecordingStatus | None:
        """Return the final sanitized status for one trusted recording key."""

    def record(
        self, workspace_id: str, call_sid: str,
        status: VoicemailRecordingStatus,
    ) -> VoicemailRecordingStatus:
        """Store one final status, allowing only exact idempotent duplicates."""


class VoicemailStatusRetentionStore(Protocol):
    """Conditional metadata deletion boundary supplied by a retention adapter."""

    def delete(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> bool:
        """Delete exact expected metadata, returning false when already absent."""


def _validate_status(status: VoicemailRecordingStatus) -> VoicemailRecordingStatus:
    """Reject values that could not have crossed the admission boundary."""
    if type(status) is not VoicemailRecordingStatus:
        raise ValueError("status must be an exact VoicemailRecordingStatus")
    _validate_recording_sid(status.recording_sid)
    if status.availability == "available":
        if (type(status.duration_seconds) is not int
                or not 0 <= status.duration_seconds <= MAX_VOICEMAIL_SECONDS):
            raise ValueError("available status must have a bounded duration")
    elif status.availability == "unavailable":
        if status.duration_seconds is not None:
            raise ValueError("unavailable status must not have a duration")
    else:
        raise ValueError("status has an unsupported availability")
    return status


def _key(
    workspace_id: str, call_sid: str, recording_sid: str,
) -> tuple[str, str, str]:
    workspace_id, call_sid = _call_key(workspace_id, call_sid)
    return workspace_id, call_sid, _validate_recording_sid(recording_sid)


def load_voicemail_recording_status(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    store: VoicemailStatusStore,
) -> VoicemailRecordingStatus | None:
    """Load and validate one final status selected only by a trusted key.

    Workspace authorization and every identifier must come from trusted
    application state. The adapter shape and exact key are validated before the
    load. Missing state remains ``None``; a present value must be an exact,
    well-formed status for the requested recording. Adapter errors propagate.

    This helper performs no authentication, authorization, media retrieval,
    logging, network access, or provider operation.
    """
    load = getattr(store, "load", None)
    if not callable(load):
        raise ValueError("store must provide a trusted callable load method")
    key = _key(workspace_id, call_sid, recording_sid)
    status = load(*key)
    if status is None:
        return None
    status = _validate_status(status)
    if status.recording_sid != recording_sid:
        raise ValueError("store returned a status for an unexpected recording")
    return status


def delete_voicemail_recording_status(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_status: VoicemailRecordingStatus,
    store: VoicemailStatusRetentionStore,
) -> bool:
    """Conditionally delete final status metadata selected by a trusted key.

    The expected immutable value prevents stale retention work from deleting a
    different observation. The adapter shape, exact key, and expected value are
    validated before deletion. A missing value is an idempotent ``False``;
    adapter errors propagate and non-boolean results fail closed.

    This helper deletes neither provider media nor caller audio and performs no
    authentication, authorization, logging, network access, or provider work.
    Production must prevent late callbacks from recreating expired metadata.
    """
    delete = getattr(store, "delete", None)
    if not callable(delete):
        raise ValueError("store must provide a trusted callable delete method")
    key = _key(workspace_id, call_sid, recording_sid)
    expected_status = _validate_status(expected_status)
    if expected_status.recording_sid != key[2]:
        raise ValueError("expected status is for an unexpected recording")
    deleted = delete(*key, expected_status=expected_status)
    if type(deleted) is not bool:
        raise ValueError("store delete must return an exact boolean")
    return deleted


class InMemoryVoicemailStatusStore:
    """Thread-safe, process-local test double for immutable final statuses.

    Authenticate and bind the callback before recording its sanitized result.
    This store performs no authentication, workspace authorization, expiry,
    media retrieval, logging, network access, files, or provider operations.
    State is neither durable nor shared across processes or Lambda invocations.
    """

    def __init__(self, *, max_entries: int = 1000) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._states: dict[
            tuple[str, str, str], VoicemailRecordingStatus
        ] = {}
        self._lock = Lock()

    def load(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> VoicemailRecordingStatus | None:
        """Read one exact key without allocating storage or spending capacity."""
        key = _key(workspace_id, call_sid, recording_sid)
        with self._lock:
            return self._states.get(key)

    def record(
        self, workspace_id: str, call_sid: str,
        status: VoicemailRecordingStatus,
    ) -> VoicemailRecordingStatus:
        """Persist the first final status and reject contradictory redelivery."""
        status = _validate_status(status)
        key = _key(workspace_id, call_sid, status.recording_sid)
        with self._lock:
            current = self._states.get(key)
            if current is not None:
                if current != status:
                    raise VoicemailStatusConflictError(
                        "voicemail recording status conflict"
                    )
                return current
            if len(self._states) >= self._max_entries:
                raise VoicemailStatusCapacityError(
                    "voicemail status capacity reached"
                )
            self._states[key] = status
            return status

    def delete(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> bool:
        """Delete only matching metadata; stale expectations fail closed."""
        key = _key(workspace_id, call_sid, recording_sid)
        expected_status = _validate_status(expected_status)
        if expected_status.recording_sid != key[2]:
            raise ValueError("expected status is for an unexpected recording")
        with self._lock:
            current = self._states.get(key)
            if current is None:
                return False
            if current != expected_status:
                raise VoicemailStatusConflictError(
                    "voicemail recording status conflict"
                )
            del self._states[key]
            return True
