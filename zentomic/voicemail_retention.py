"""Bounded voicemail retention-worker orchestration."""

from typing import Protocol

from zentomic.voicemail_status_store import (
    DueVoicemailExpiry,
    list_due_voicemail_recording_status_expiries,
    purge_due_voicemail_recording_status_expiry,
    VoicemailStatusConflictError,
    VoicemailStatusDueExpiryStore,
    VoicemailStatusDuePurgeStore,
)


class VoicemailStatusDueExpiryPurgeStore(
    VoicemailStatusDueExpiryStore, VoicemailStatusDuePurgeStore, Protocol,
):
    """Discovery and conditional cleanup boundaries supplied by an adapter."""


def purge_due_voicemail_recording_status_expiry_batch(
    workspace_id: str, *, now_ms: int, limit: int = 100,
    store: VoicemailStatusDueExpiryPurgeStore,
) -> tuple[DueVoicemailExpiry, ...]:
    """Discover and conditionally purge one bounded batch of due tombstones.

    Each discovered snapshot is passed unchanged to the atomic due-purge
    boundary. A missing tombstone or an expected conflict from concurrent
    cleanup, extension, or retention-hold work is skipped so one stale
    observation does not block the rest of the batch. Unexpected adapter and
    validation errors propagate; this helper is deliberately not a transaction
    across the batch.

    The returned tuple contains only candidates with confirmed atomic purge
    receipts, in discovery order. This helper reads no clock and performs no
    authentication, authorization, media deletion, logging, network access, or
    provider operation.
    """
    purge_expired_if_due = getattr(store, "purge_expired_if_due", None)
    if not callable(purge_expired_if_due):
        raise ValueError(
            "store must provide a trusted callable purge_expired_if_due method"
        )
    candidates = list_due_voicemail_recording_status_expiries(
        workspace_id, now_ms=now_ms, limit=limit, store=store,
    )
    purged: list[DueVoicemailExpiry] = []
    for candidate in candidates:
        try:
            receipt = purge_due_voicemail_recording_status_expiry(
                workspace_id,
                candidate.call_sid,
                candidate.recording_sid,
                expected_snapshot=candidate.snapshot,
                now_ms=now_ms,
                store=store,
            )
        except VoicemailStatusConflictError:
            continue
        if receipt is not None:
            purged.append(candidate)
    return tuple(purged)
