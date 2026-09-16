# Voicemail retention worker

`purge_due_voicemail_recording_status_expiry_batch` composes bounded discovery
with conditional deletion for one retention-worker pass. It returns only
entries with confirmed atomic purge receipts and preserves discovery order.
The companion
`purge_due_voicemail_recording_status_expiry_batch_report` returns immutable
`discovered`, `purged`, `conflicted`, and `missing` partitions. Workers can use
those partitions for metrics without re-reading state that may have changed.
The clearer `no_receipt` alias exposes the legacy `missing` partition without
claiming that storage was observed to be absent. A `None` purge result provides
no atomic receipt, and the batch deliberately does not perform a follow-up read.
Its immutable `counts` summary exposes scalar values for telemetry and a
matching `no_receipt` alias, plus a `discovery_limit_reached` flag. A true flag
means the bounded query filled its requested limit, so more due work may remain
and the worker can schedule another pass without treating the flag as proof of
backlog.

The counts and report also expose a `follow_up_recommended` scheduling hint. It
is true when discovery filled the batch, any candidate conflicted, or any purge
returned no receipt. This gives workers one conservative retry decision without
re-reading mutable state. It is only a scheduling hint: saturation does not
prove a backlog, and an unconfirmed candidate may disappear before the next
pass. The companion `follow_up_reasons` tuple identifies those conditions with
stable `discovery_limit_reached`, `conflicted`, and `no_receipt` reason codes in
that order. Workers can use the codes for metrics or choose different retry
delays without inspecting mutable candidate state.

Reports validate that every discovered candidate appears in exactly one outcome
partition and that each partition preserves discovery order. Scalar summaries
likewise require nonnegative exact integers whose outcomes total the discovered
count, preventing malformed worker telemetry from being published silently.

The worker skips candidates without receipts and conflicting candidates. This
lets concurrent cleanup, deadline extension, and retention-hold work proceed
without one stale observation blocking unrelated entries. Unexpected adapter
or validation errors still propagate. The batch is not a transaction, so an
error can occur after earlier candidates were purged successfully.

The caller must provide a fresh trusted `now_ms` and an already authorized
workspace. Provider-media deletion, policy decisions, logging, retries, and
network access remain outside the helper. Every purge still compares the exact
discovered tombstone version and deadline atomically before deletion.
