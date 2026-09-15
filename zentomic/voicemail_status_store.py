"""Process-local final voicemail status storage for adapter development."""

from dataclasses import dataclass
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


class VoicemailStatusExpiredError(ValueError):
    """A final status cannot be recreated after durable expiry."""


@dataclass(frozen=True)
class VoicemailRecordingStatusSnapshot:
    """Atomic tri-state observation without any deleted status metadata."""

    status: VoicemailRecordingStatus | None = None
    expired: bool = False
    expiry_version: int | None = None


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


class VoicemailStatusExpiryStore(Protocol):
    """Conditional durable-expiry boundary supplied by a retention adapter."""

    def expire(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> bool:
        """Replace exact expected metadata with a durable tombstone."""


class VoicemailStatusSnapshotExpiryStore(Protocol):
    """Atomic expiry-and-receipt boundary supplied by a retention adapter."""

    def expire_and_inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Replace exact metadata and return only the created tombstone."""


class VoicemailStatusExpiryReader(Protocol):
    """Read-only tombstone boundary supplied by a retention adapter."""

    def is_expired(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> bool:
        """Return whether the exact trusted recording key has a tombstone."""


class VoicemailStatusSnapshotStore(Protocol):
    """Atomic status-or-tombstone read boundary supplied by an adapter."""

    def inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> VoicemailRecordingStatusSnapshot:
        """Return one validated active, expired, or missing observation."""


class VoicemailStatusExpiryPurgeStore(Protocol):
    """Tombstone cleanup boundary supplied by a retention adapter."""

    def purge_expired(
        self, workspace_id: str, call_sid: str, recording_sid: str,
        *, expected_version: int,
    ) -> bool:
        """Conditionally remove the exact previously observed tombstone."""


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


def _validate_snapshot(
    snapshot: VoicemailRecordingStatusSnapshot, *, recording_sid: str,
    source: str = "inspect",
) -> VoicemailRecordingStatusSnapshot:
    """Validate one adapter snapshot and bind any status to the trusted key."""
    if type(snapshot) is not VoicemailRecordingStatusSnapshot:
        raise ValueError(
            f"store {source} must return an exact "
            "VoicemailRecordingStatusSnapshot"
        )
    if type(snapshot.expired) is not bool:
        raise ValueError("snapshot expired must be an exact boolean")
    expiry_version = snapshot.expiry_version
    if (expiry_version is not None
            and (type(expiry_version) is not int or expiry_version < 1)):
        raise ValueError("snapshot expiry version must be a positive integer")
    if snapshot.expired != (expiry_version is not None):
        raise ValueError("snapshot expiry state and version must agree")
    if snapshot.status is None:
        return snapshot
    status = _validate_status(snapshot.status)
    if snapshot.expired:
        raise ValueError("snapshot cannot contain active and expired state")
    if status.recording_sid != recording_sid:
        raise ValueError("store returned a status for an unexpected recording")
    return snapshot


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


def expire_voicemail_recording_status(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_status: VoicemailRecordingStatus,
    store: VoicemailStatusExpiryStore,
) -> bool:
    """Conditionally replace final status metadata with a tombstone.

    The adapter must atomically replace only the exact expected immutable value
    and retain a durable tombstone that rejects later writes for the same key.
    A missing or already expired value is an idempotent ``False``. The adapter
    shape, exact key, expected value, and result are validated; errors propagate.

    This helper performs no authentication, authorization, media deletion,
    logging, network access, or provider operation. Delete provider media in a
    separately authorized retention workflow before expiring its metadata.
    """
    expire = getattr(store, "expire", None)
    if not callable(expire):
        raise ValueError("store must provide a trusted callable expire method")
    key = _key(workspace_id, call_sid, recording_sid)
    expected_status = _validate_status(expected_status)
    if expected_status.recording_sid != key[2]:
        raise ValueError("expected status is for an unexpected recording")
    expired = expire(*key, expected_status=expected_status)
    if type(expired) is not bool:
        raise ValueError("store expire must return an exact boolean")
    return expired


def expire_voicemail_recording_status_snapshot(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_status: VoicemailRecordingStatus,
    store: VoicemailStatusSnapshotExpiryStore,
) -> VoicemailRecordingStatusSnapshot | None:
    """Atomically expire metadata and return the created tombstone snapshot.

    The adapter must replace only the exact expected immutable status and
    return the newly created, versioned tombstone in the same operation. A
    missing or already expired value returns ``None``. This keeps a retention
    worker from adopting a tombstone version created by a later recording
    lifecycle through a separate inspection.

    The helper validates the adapter, trusted key, expected value, and returned
    snapshot. It performs no authentication, authorization, clock check, media
    deletion, logging, network access, or provider operation.
    """
    expire_and_inspect = getattr(store, "expire_and_inspect", None)
    if not callable(expire_and_inspect):
        raise ValueError(
            "store must provide a trusted callable expire_and_inspect method"
        )
    key = _key(workspace_id, call_sid, recording_sid)
    expected_status = _validate_status(expected_status)
    if expected_status.recording_sid != key[2]:
        raise ValueError("expected status is for an unexpected recording")
    snapshot = expire_and_inspect(*key, expected_status=expected_status)
    if snapshot is None:
        return None
    snapshot = _validate_snapshot(
        snapshot, recording_sid=key[2], source="expire_and_inspect",
    )
    if snapshot.status is not None or snapshot.expiry_version is None:
        raise ValueError("store expiry result must identify a tombstone")
    return snapshot


def is_voicemail_recording_status_expired(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    store: VoicemailStatusExpiryReader,
) -> bool:
    """Inspect durable expiry state for one trusted recording key.

    The adapter shape and exact key are validated before the lookup, and only
    an exact boolean result is accepted. This distinguishes a tombstone from a
    recording that never had status metadata without exposing deleted status.
    Adapter errors propagate.

    This observation is not authorization to accept a callback or remove a
    tombstone. Production workflows still need an atomic durable write when
    expiry state and another decision must remain consistent.
    """
    is_expired = getattr(store, "is_expired", None)
    if not callable(is_expired):
        raise ValueError("store must provide a trusted callable is_expired method")
    expired = is_expired(*_key(workspace_id, call_sid, recording_sid))
    if type(expired) is not bool:
        raise ValueError("store is_expired must return an exact boolean")
    return expired


def inspect_voicemail_recording_status(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    store: VoicemailStatusSnapshotStore,
) -> VoicemailRecordingStatusSnapshot:
    """Atomically inspect active, expired, or missing state for a trusted key.

    The adapter shape and exact key are validated before the lookup. A result
    must be an exact snapshot containing either one validated active status or
    an expiry marker, never both. Deleted status metadata is not exposed and
    adapter errors propagate.

    This read does not reserve state or authorize a callback, mutation, or
    tombstone purge. Production decisions that depend on the observation still
    need a conditional durable write in the same authorized workflow.
    """
    inspect = getattr(store, "inspect", None)
    if not callable(inspect):
        raise ValueError("store must provide a trusted callable inspect method")
    key = _key(workspace_id, call_sid, recording_sid)
    snapshot = inspect(*key)
    return _validate_snapshot(snapshot, recording_sid=key[2])


def purge_voicemail_recording_status_expiry(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_snapshot: VoicemailRecordingStatusSnapshot,
    store: VoicemailStatusExpiryPurgeStore,
) -> bool:
    """Conditionally remove one previously observed retained tombstone.

    Call this only after a separately authorized retention workflow establishes
    that the provider replay window has elapsed. The exact snapshot version
    prevents stale cleanup from removing a newer tombstone for the same key.
    The adapter must atomically remove only that tombstone, never active status
    metadata. Missing or active state is an idempotent ``False``; a different
    version is a conflict. Inputs and results are validated; errors propagate.

    Purging releases the recreation guard, so subsequent callback delivery may
    store status for the key again. This helper performs no authentication,
    authorization, clock check, media deletion, logging, network access, or
    provider operation.
    """
    purge_expired = getattr(store, "purge_expired", None)
    if not callable(purge_expired):
        raise ValueError(
            "store must provide a trusted callable purge_expired method"
        )
    key = _key(workspace_id, call_sid, recording_sid)
    expected_snapshot = _validate_snapshot(
        expected_snapshot, recording_sid=key[2],
    )
    expected_version = expected_snapshot.expiry_version
    if expected_snapshot.status is not None or expected_version is None:
        raise ValueError("expected snapshot must identify an expiry tombstone")
    purged = purge_expired(*key, expected_version=expected_version)
    if type(purged) is not bool:
        raise ValueError("store purge_expired must return an exact boolean")
    return purged


class InMemoryVoicemailStatusStore:
    """Thread-safe, process-local test double for immutable final statuses.

    Authenticate and bind the callback before recording its sanitized result.
    This store performs no authentication, workspace authorization, timed
    retention, media retrieval, logging, network access, files, or provider
    operations.
    State is neither durable nor shared across processes or Lambda invocations.
    """

    def __init__(self, *, max_entries: int = 1000) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._states: dict[
            tuple[str, str, str], VoicemailRecordingStatus
        ] = {}
        self._expired: dict[tuple[str, str, str], int] = {}
        self._next_expiry_version = 1
        self._lock = Lock()

    def load(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> VoicemailRecordingStatus | None:
        """Read one exact key without allocating storage or spending capacity."""
        key = _key(workspace_id, call_sid, recording_sid)
        with self._lock:
            return self._states.get(key)

    def is_expired(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> bool:
        """Report whether one exact key has a retained expiry tombstone."""
        key = _key(workspace_id, call_sid, recording_sid)
        with self._lock:
            return key in self._expired

    def inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str,
    ) -> VoicemailRecordingStatusSnapshot:
        """Read active, expired, or missing state under one local lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        with self._lock:
            status = self._states.get(key)
            expiry_version = self._expired.get(key)
            return VoicemailRecordingStatusSnapshot(
                status=status,
                expired=status is None and expiry_version is not None,
                expiry_version=(
                    expiry_version if status is None else None
                ),
            )

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
            if key in self._expired:
                raise VoicemailStatusExpiredError(
                    "voicemail recording status has expired"
                )
            if len(self._states) + len(self._expired) >= self._max_entries:
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

    def expire(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> bool:
        """Replace matching metadata with a tombstone that blocks recreation."""
        return self.expire_and_inspect(
            workspace_id, call_sid, recording_sid,
            expected_status=expected_status,
        ) is not None

    def expire_and_inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_status: VoicemailRecordingStatus,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Replace matching metadata and return its tombstone under one lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        expected_status = _validate_status(expected_status)
        if expected_status.recording_sid != key[2]:
            raise ValueError("expected status is for an unexpected recording")
        with self._lock:
            if key in self._expired:
                return None
            current = self._states.get(key)
            if current is None:
                return None
            if current != expected_status:
                raise VoicemailStatusConflictError(
                    "voicemail recording status conflict"
                )
            del self._states[key]
            self._expired[key] = self._next_expiry_version
            self._next_expiry_version += 1
            return VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=self._expired[key],
            )

    def purge_expired(
        self, workspace_id: str, call_sid: str, recording_sid: str,
        *, expected_version: int,
    ) -> bool:
        """Remove only the exact observed tombstone for one trusted key."""
        key = _key(workspace_id, call_sid, recording_sid)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected expiry version must be a positive integer")
        with self._lock:
            current_version = self._expired.get(key)
            if current_version is None:
                return False
            if current_version != expected_version:
                raise VoicemailStatusConflictError(
                    "voicemail expiry tombstone conflict"
                )
            del self._expired[key]
            return True
