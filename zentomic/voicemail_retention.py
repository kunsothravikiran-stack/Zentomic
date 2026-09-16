"""Bounded voicemail retention-worker orchestration."""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class VoicemailExpiryPurgeBatchCounts:
    """Stable scalar metrics for one bounded cleanup pass."""

    discovered: int
    purged: int
    conflicted: int
    missing: int
    discovery_limit_reached: bool

    @property
    def no_receipt(self) -> int:
        """Return candidates for which no atomic purge receipt was returned.

        ``missing`` is retained as the original field name for compatibility,
        but a ``None`` adapter result does not prove the key is absent without
        a separate read. Workers should publish this clearer metric name.
        """
        return self.missing


@dataclass(frozen=True)
class VoicemailExpiryPurgeBatchReport:
    """Immutable outcome partitions for one bounded cleanup pass."""

    discovered: tuple[DueVoicemailExpiry, ...]
    purged: tuple[DueVoicemailExpiry, ...]
    conflicted: tuple[DueVoicemailExpiry, ...]
    missing: tuple[DueVoicemailExpiry, ...]
    limit: int = 100

    @property
    def no_receipt(self) -> tuple[DueVoicemailExpiry, ...]:
        """Return candidates whose purge produced no atomic receipt.

        This is an alias for the legacy ``missing`` partition. The batch does
        not re-read storage, so consumers must not infer current absence from
        membership in this partition.
        """
        return self.missing

    @property
    def counts(self) -> VoicemailExpiryPurgeBatchCounts:
        """Return immutable metrics without exposing mutable worker state."""
        return VoicemailExpiryPurgeBatchCounts(
            discovered=len(self.discovered),
            purged=len(self.purged),
            conflicted=len(self.conflicted),
            missing=len(self.missing),
            discovery_limit_reached=len(self.discovered) == self.limit,
        )


def purge_due_voicemail_recording_status_expiry_batch_report(
    workspace_id: str, *, now_ms: int, limit: int = 100,
    store: VoicemailStatusDueExpiryPurgeStore,
) -> VoicemailExpiryPurgeBatchReport:
    """Purge one bounded due batch and classify every discovered candidate.

    Candidates without atomic purge receipts and expected conflicts are
    reported separately so a retention worker can publish useful metrics
    without re-reading mutable state. The legacy ``missing`` partition means
    only "no receipt"; it is not proof of current storage state. Unexpected
    adapter and validation errors still propagate; this helper is deliberately
    not a transaction across the batch.
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
    conflicted: list[DueVoicemailExpiry] = []
    no_receipt: list[DueVoicemailExpiry] = []
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
            conflicted.append(candidate)
            continue
        if receipt is None:
            no_receipt.append(candidate)
        else:
            purged.append(candidate)
    return VoicemailExpiryPurgeBatchReport(
        discovered=candidates,
        purged=tuple(purged),
        conflicted=tuple(conflicted),
        missing=tuple(no_receipt),
        limit=limit,
    )


def purge_due_voicemail_recording_status_expiry_batch(
    workspace_id: str, *, now_ms: int, limit: int = 100,
    store: VoicemailStatusDueExpiryPurgeStore,
) -> tuple[DueVoicemailExpiry, ...]:
    """Discover and conditionally purge one bounded batch of due tombstones.

    Each discovered snapshot is passed unchanged to the atomic due-purge
    boundary. A candidate without an atomic receipt or an expected conflict
    from concurrent cleanup, extension, or retention-hold work is skipped so
    one stale observation does not block the rest of the batch. Unexpected
    adapter and validation errors propagate; this helper is deliberately not a
    transaction across the batch.

    The returned tuple contains only candidates with confirmed atomic purge
    receipts, in discovery order. This helper reads no clock and performs no
    authentication, authorization, media deletion, logging, network access, or
    provider operation.
    """
    return purge_due_voicemail_recording_status_expiry_batch_report(
        workspace_id, now_ms=now_ms, limit=limit, store=store,
    ).purged
