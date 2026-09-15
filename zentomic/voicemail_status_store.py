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
    purge_after_ms: int | None = None


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
        purge_after_ms: int | None = None,
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
        """Remove an exact unscheduled tombstone, never one with a deadline."""


class VoicemailStatusSnapshotPurgeStore(Protocol):
    """Atomic tombstone-purge receipt boundary supplied by an adapter."""

    def purge_expired_and_inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str,
        *, expected_version: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Remove and return an exact unscheduled tombstone only."""


class VoicemailStatusDuePurgeStore(Protocol):
    """Atomic due-tombstone cleanup boundary supplied by an adapter."""

    def purge_expired_if_due(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_version: int, expected_purge_after_ms: int, now_ms: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Remove and return the exact tombstone only after its deadline."""


class VoicemailStatusDeadlineExtensionStore(Protocol):
    """Atomic scheduled-tombstone extension boundary supplied by an adapter."""

    def extend_expiry_deadline(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_version: int, expected_purge_after_ms: int,
        new_purge_after_ms: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Extend and return the exact scheduled tombstone, if present."""


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
    purge_after_ms = snapshot.purge_after_ms
    if (purge_after_ms is not None
            and (type(purge_after_ms) is not int or purge_after_ms < 0)):
        raise ValueError("snapshot purge deadline must be a nonnegative integer")
    if not snapshot.expired and purge_after_ms is not None:
        raise ValueError("only an expiry snapshot may contain a purge deadline")
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
    purge_after_ms: int | None = None,
    store: VoicemailStatusSnapshotExpiryStore,
) -> VoicemailRecordingStatusSnapshot | None:
    """Atomically expire metadata and return the created tombstone snapshot.

    The adapter must replace only the exact expected immutable status and
    return the newly created, versioned tombstone in the same operation. A
    missing or already expired value returns ``None``. This keeps a retention
    worker from adopting a tombstone version created by a later recording
    lifecycle through a separate inspection. When ``purge_after_ms`` is set,
    the adapter must persist that trusted replay-window deadline atomically.

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
    if (purge_after_ms is not None
            and (type(purge_after_ms) is not int or purge_after_ms < 0)):
        raise ValueError("purge_after_ms must be a nonnegative integer or None")
    if purge_after_ms is None:
        snapshot = expire_and_inspect(*key, expected_status=expected_status)
    else:
        snapshot = expire_and_inspect(
            *key,
            expected_status=expected_status,
            purge_after_ms=purge_after_ms,
        )
    if snapshot is None:
        return None
    snapshot = _validate_snapshot(
        snapshot, recording_sid=key[2], source="expire_and_inspect",
    )
    if (snapshot.status is not None or snapshot.expiry_version is None
            or snapshot.purge_after_ms != purge_after_ms):
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
    version is a conflict. Deadline-bearing tombstones require the due-purge
    boundary and are rejected. Inputs and results are validated; errors propagate.

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
    if expected_snapshot.purge_after_ms is not None:
        raise ValueError("scheduled expiry requires a due-purge adapter")
    purged = purge_expired(*key, expected_version=expected_version)
    if type(purged) is not bool:
        raise ValueError("store purge_expired must return an exact boolean")
    return purged


def purge_voicemail_recording_status_expiry_snapshot(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_snapshot: VoicemailRecordingStatusSnapshot,
    store: VoicemailStatusSnapshotPurgeStore,
) -> VoicemailRecordingStatusSnapshot | None:
    """Atomically purge and return one exact retained tombstone.

    Call this only after a separately authorized retention workflow establishes
    that the provider replay window has elapsed. The adapter must atomically
    remove the exact expected version and return that removed tombstone in the
    same operation. Missing or active state returns ``None``; a newer version
    is a conflict. This gives cleanup work a receipt without a racy follow-up
    read after the recreation guard has been released. Deadline-bearing
    tombstones require the due-purge boundary and are rejected.

    The helper validates the adapter, trusted key, expected snapshot, and
    returned receipt. It performs no authentication, authorization, clock
    check, media deletion, logging, network access, or provider operation.
    """
    purge_expired_and_inspect = getattr(
        store, "purge_expired_and_inspect", None,
    )
    if not callable(purge_expired_and_inspect):
        raise ValueError(
            "store must provide a trusted callable "
            "purge_expired_and_inspect method"
        )
    key = _key(workspace_id, call_sid, recording_sid)
    expected_snapshot = _validate_snapshot(
        expected_snapshot, recording_sid=key[2],
    )
    expected_version = expected_snapshot.expiry_version
    if expected_snapshot.status is not None or expected_version is None:
        raise ValueError("expected snapshot must identify an expiry tombstone")
    if expected_snapshot.purge_after_ms is not None:
        raise ValueError("scheduled expiry requires a due-purge adapter")
    receipt = purge_expired_and_inspect(
        *key, expected_version=expected_version,
    )
    if receipt is None:
        return None
    receipt = _validate_snapshot(
        receipt, recording_sid=key[2], source="purge_expired_and_inspect",
    )
    if receipt.status is not None or receipt.expiry_version is None:
        raise ValueError("store purge result must identify a tombstone")
    if receipt != expected_snapshot:
        raise ValueError("store purge result must match the expected tombstone")
    return receipt


def purge_due_voicemail_recording_status_expiry(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_snapshot: VoicemailRecordingStatusSnapshot, now_ms: int,
    store: VoicemailStatusDuePurgeStore,
) -> VoicemailRecordingStatusSnapshot | None:
    """Atomically purge one exact tombstone only after its stored deadline.

    The adapter must compare the exact expected version and deadline with its
    durable tombstone and enforce ``purge_after_ms <= now_ms`` in the same
    conditional removal. Missing, active, or not-yet-due state returns ``None``;
    a different tombstone is a conflict. A successful result is an atomic
    receipt and must exactly match ``expected_snapshot``.

    Supply ``now_ms`` from a fresh trusted clock in the adapter's clock domain.
    This helper reads no clock and performs no authentication, authorization,
    media deletion, logging, network access, or provider operation.
    """
    purge_expired_if_due = getattr(store, "purge_expired_if_due", None)
    if not callable(purge_expired_if_due):
        raise ValueError(
            "store must provide a trusted callable purge_expired_if_due method"
        )
    key = _key(workspace_id, call_sid, recording_sid)
    expected_snapshot = _validate_snapshot(
        expected_snapshot, recording_sid=key[2],
    )
    expected_version = expected_snapshot.expiry_version
    purge_after_ms = expected_snapshot.purge_after_ms
    if (expected_snapshot.status is not None or expected_version is None
            or purge_after_ms is None):
        raise ValueError(
            "expected snapshot must identify a scheduled expiry tombstone"
        )
    if type(now_ms) is not int or now_ms < 0:
        raise ValueError("now_ms must be a nonnegative integer timestamp")
    if now_ms < purge_after_ms:
        return None
    receipt = purge_expired_if_due(
        *key,
        expected_version=expected_version,
        expected_purge_after_ms=purge_after_ms,
        now_ms=now_ms,
    )
    if receipt is None:
        return None
    receipt = _validate_snapshot(
        receipt, recording_sid=key[2], source="purge_expired_if_due",
    )
    if receipt != expected_snapshot:
        raise ValueError("store due purge result must match the expected tombstone")
    return receipt


def extend_voicemail_recording_status_expiry_deadline(
    workspace_id: str, call_sid: str, recording_sid: str, *,
    expected_snapshot: VoicemailRecordingStatusSnapshot,
    new_purge_after_ms: int,
    store: VoicemailStatusDeadlineExtensionStore,
) -> VoicemailRecordingStatusSnapshot | None:
    """Atomically extend one exact scheduled tombstone's purge deadline.

    The adapter must compare the expected version and persisted deadline before
    replacing only that deadline and returning the updated tombstone in the
    same operation. Missing or active state returns ``None``; a different
    tombstone is a conflict. Deadlines may only increase, preventing stale
    retention work from shortening an established provider replay window.

    The helper validates the adapter, trusted key, snapshots, and exact result.
    It reads no clock and performs no authentication, authorization, media
    deletion, logging, network access, or provider operation.
    """
    extend_expiry_deadline = getattr(store, "extend_expiry_deadline", None)
    if not callable(extend_expiry_deadline):
        raise ValueError(
            "store must provide a trusted callable "
            "extend_expiry_deadline method"
        )
    key = _key(workspace_id, call_sid, recording_sid)
    expected_snapshot = _validate_snapshot(
        expected_snapshot, recording_sid=key[2],
    )
    expected_version = expected_snapshot.expiry_version
    expected_purge_after_ms = expected_snapshot.purge_after_ms
    if (expected_snapshot.status is not None or expected_version is None
            or expected_purge_after_ms is None):
        raise ValueError(
            "expected snapshot must identify a scheduled expiry tombstone"
        )
    if type(new_purge_after_ms) is not int or new_purge_after_ms < 0:
        raise ValueError("new_purge_after_ms must be a nonnegative integer")
    if new_purge_after_ms <= expected_purge_after_ms:
        raise ValueError("new purge deadline must be later than the current deadline")
    receipt = extend_expiry_deadline(
        *key,
        expected_version=expected_version,
        expected_purge_after_ms=expected_purge_after_ms,
        new_purge_after_ms=new_purge_after_ms,
    )
    if receipt is None:
        return None
    receipt = _validate_snapshot(
        receipt, recording_sid=key[2], source="extend_expiry_deadline",
    )
    expected_receipt = VoicemailRecordingStatusSnapshot(
        expired=True,
        expiry_version=expected_version,
        purge_after_ms=new_purge_after_ms,
    )
    if receipt != expected_receipt:
        raise ValueError(
            "store deadline extension result must match the requested tombstone"
        )
    return receipt


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
        self._expired: dict[
            tuple[str, str, str], VoicemailRecordingStatusSnapshot
        ] = {}
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
            expiry = self._expired.get(key)
            if status is not None:
                return VoicemailRecordingStatusSnapshot(status=status)
            return expiry or VoicemailRecordingStatusSnapshot()

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
        purge_after_ms: int | None = None,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Replace matching metadata and return its tombstone under one lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        expected_status = _validate_status(expected_status)
        if expected_status.recording_sid != key[2]:
            raise ValueError("expected status is for an unexpected recording")
        if (purge_after_ms is not None
                and (type(purge_after_ms) is not int or purge_after_ms < 0)):
            raise ValueError("purge_after_ms must be a nonnegative integer or None")
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
            snapshot = VoicemailRecordingStatusSnapshot(
                expired=True,
                expiry_version=self._next_expiry_version,
                purge_after_ms=purge_after_ms,
            )
            self._expired[key] = snapshot
            self._next_expiry_version += 1
            return snapshot

    def purge_expired(
        self, workspace_id: str, call_sid: str, recording_sid: str,
        *, expected_version: int,
    ) -> bool:
        """Remove only the exact observed tombstone for one trusted key."""
        return self.purge_expired_and_inspect(
            workspace_id, call_sid, recording_sid,
            expected_version=expected_version,
        ) is not None

    def purge_expired_and_inspect(
        self, workspace_id: str, call_sid: str, recording_sid: str,
        *, expected_version: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Remove and return the exact observed tombstone under one lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected expiry version must be a positive integer")
        with self._lock:
            current = self._expired.get(key)
            if current is None:
                return None
            if current.expiry_version != expected_version:
                raise VoicemailStatusConflictError(
                    "voicemail expiry tombstone conflict"
                )
            if current.purge_after_ms is not None:
                raise ValueError("scheduled expiry requires a due-purge adapter")
            del self._expired[key]
            return current

    def purge_expired_if_due(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_version: int, expected_purge_after_ms: int, now_ms: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Remove one exact scheduled tombstone when due under one lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected expiry version must be a positive integer")
        for name, value in (
            ("expected purge deadline", expected_purge_after_ms),
            ("current time", now_ms),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        with self._lock:
            current = self._expired.get(key)
            if current is None:
                return None
            if (current.expiry_version != expected_version
                    or current.purge_after_ms != expected_purge_after_ms):
                raise VoicemailStatusConflictError(
                    "voicemail expiry tombstone conflict"
                )
            if now_ms < expected_purge_after_ms:
                return None
            del self._expired[key]
            return current

    def extend_expiry_deadline(
        self, workspace_id: str, call_sid: str, recording_sid: str, *,
        expected_version: int, expected_purge_after_ms: int,
        new_purge_after_ms: int,
    ) -> VoicemailRecordingStatusSnapshot | None:
        """Extend one exact scheduled tombstone under one local lock."""
        key = _key(workspace_id, call_sid, recording_sid)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected expiry version must be a positive integer")
        for name, value in (
            ("expected purge deadline", expected_purge_after_ms),
            ("new purge deadline", new_purge_after_ms),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if new_purge_after_ms <= expected_purge_after_ms:
            raise ValueError("new purge deadline must be later than the current deadline")
        with self._lock:
            current = self._expired.get(key)
            if current is None:
                return None
            if (current.expiry_version != expected_version
                    or current.purge_after_ms != expected_purge_after_ms):
                raise VoicemailStatusConflictError(
                    "voicemail expiry tombstone conflict"
                )
            updated = VoicemailRecordingStatusSnapshot(
                expired=True,
                expiry_version=expected_version,
                purge_after_ms=new_purge_after_ms,
            )
            self._expired[key] = updated
            return updated
