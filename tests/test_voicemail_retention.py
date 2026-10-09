"""Bounded voicemail tombstone cleanup using synthetic identifiers only."""

import json
import unittest
from unittest.mock import Mock

from zentomic.voicemail_retention import (
    purge_due_voicemail_recording_status_expiry_batch,
    purge_due_voicemail_recording_status_expiry_batch_report,
    VOICEMAIL_EXPIRY_PURGE_TELEMETRY_SCHEMA_VERSION,
    VoicemailExpiryPurgeBatchCounts,
    VoicemailExpiryPurgeBatchReport,
)
from zentomic.voicemail_status import VoicemailRecordingStatus
from zentomic.voicemail_status_store import (
    DueVoicemailExpiry,
    InMemoryVoicemailStatusStore,
    list_due_voicemail_recording_status_expiries,
    VoicemailRecordingStatusSnapshot,
    VoicemailStatusConflictError,
)


class PurgeDueVoicemailExpiryBatchTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVoicemailStatusStore()

    def add_expiry(self, call_sid, recording_byte, deadline):
        recording_sid = "RE" + recording_byte * 32
        status = VoicemailRecordingStatus("available", recording_sid, 12)
        self.store.record("workspace", call_sid, status)
        snapshot = self.store.expire_and_inspect(
            "workspace", call_sid, recording_sid,
            expected_status=status, purge_after_ms=deadline,
        )
        return DueVoicemailExpiry(call_sid, recording_sid, snapshot)

    def list_due(self):
        return list_due_voicemail_recording_status_expiries(
            "workspace", now_ms=10_000, store=self.store,
        )

    def purge_batch(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "now_ms": 10_000,
            "limit": 100,
            "store": self.store,
        }
        config.update(overrides)
        return purge_due_voicemail_recording_status_expiry_batch(**config)

    def purge_batch_report(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "now_ms": 10_000,
            "limit": 100,
            "store": self.store,
        }
        config.update(overrides)
        return purge_due_voicemail_recording_status_expiry_batch_report(
            **config
        )

    def test_purges_a_bounded_batch_in_discovery_order(self):
        last = self.add_expiry("call-b", "b", 9_000)
        first = self.add_expiry("call-b", "a", 8_000)
        second = self.add_expiry("call-a", "c", 9_000)

        self.assertEqual(self.purge_batch(limit=2), (first, second))
        self.assertEqual(self.list_due(), (last,))
        for candidate in (first, second):
            self.assertEqual(
                self.store.inspect(
                    "workspace", candidate.call_sid, candidate.recording_sid,
                ),
                VoicemailRecordingStatusSnapshot(),
            )

    def test_stale_candidate_does_not_block_the_rest_of_the_batch(self):
        stale = self.add_expiry("call-a", "a", 8_000)
        current = self.add_expiry("call-b", "b", 9_000)

        class HoldFirstCandidateStore:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def list_due_expiries(self, workspace_id, *, now_ms, limit):
                candidates = self.wrapped.list_due_expiries(
                    workspace_id, now_ms=now_ms, limit=limit,
                )
                candidate = candidates[0]
                self.wrapped.unschedule_expiry_deadline(
                    workspace_id,
                    candidate.call_sid,
                    candidate.recording_sid,
                    expected_version=candidate.snapshot.expiry_version,
                    expected_purge_after_ms=candidate.snapshot.purge_after_ms,
                )
                return candidates

            def purge_expired_if_due(self, *args, **kwargs):
                return self.wrapped.purge_expired_if_due(*args, **kwargs)

        store = HoldFirstCandidateStore(self.store)
        self.assertEqual(self.purge_batch(store=store), (current,))
        self.assertEqual(
            self.store.inspect(
                "workspace", stale.call_sid, stale.recording_sid,
            ),
            VoicemailRecordingStatusSnapshot(
                expired=True,
                expiry_version=stale.snapshot.expiry_version,
            ),
        )

    def test_candidate_without_receipt_is_skipped(self):
        candidate = self.add_expiry("call", "a", 8_000)
        store = Mock()
        store.list_due_expiries.return_value = (candidate,)
        store.purge_expired_if_due.return_value = None

        report = self.purge_batch_report(store=store)

        self.assertEqual(report.purged, ())
        self.assertEqual(report.no_receipt, (candidate,))
        self.assertEqual(report.missing, report.no_receipt)
        self.assertEqual(report.unconfirmed, (candidate,))
        self.assertEqual(report.counts.no_receipt, 1)
        self.assertEqual(report.counts.missing, report.counts.no_receipt)
        self.assertEqual(report.counts.unconfirmed, 1)
        store.purge_expired_if_due.assert_called_once()

    def test_report_classifies_each_candidate_in_discovery_order(self):
        purged = self.add_expiry("call-a", "a", 8_000)
        missing = self.add_expiry("call-b", "b", 9_000)
        conflicted = self.add_expiry("call-c", "c", 9_500)
        store = Mock()
        store.list_due_expiries.return_value = (
            purged, missing, conflicted,
        )
        store.purge_expired_if_due.side_effect = (
            purged.snapshot,
            None,
            VoicemailStatusConflictError("synthetic contention"),
        )

        report = self.purge_batch_report(store=store)

        self.assertEqual(
            report,
            VoicemailExpiryPurgeBatchReport(
                discovered=(purged, missing, conflicted),
                purged=(purged,),
                conflicted=(conflicted,),
                missing=(missing,),
            ),
        )
        self.assertEqual(store.purge_expired_if_due.call_count, 3)
        self.assertEqual(report.unconfirmed, (missing, conflicted))
        self.assertEqual(report.counts.unconfirmed, 2)

    def test_report_exposes_scalar_counts_and_discovery_saturation(self):
        purged = self.add_expiry("call-a", "a", 8_000)
        conflicted = self.add_expiry("call-b", "b", 9_000)
        store = Mock()
        store.list_due_expiries.return_value = (purged, conflicted)
        store.purge_expired_if_due.side_effect = (
            purged.snapshot,
            VoicemailStatusConflictError("synthetic contention"),
        )

        report = self.purge_batch_report(store=store, limit=2)

        self.assertEqual(report.limit, 2)
        self.assertEqual(
            report.counts,
            VoicemailExpiryPurgeBatchCounts(
                discovered=2,
                purged=1,
                conflicted=1,
                missing=0,
                discovery_limit_reached=True,
            ),
        )

    def test_counts_reject_incomplete_or_malformed_telemetry(self):
        invalid = (
            {"discovered": 1, "purged": 0, "conflicted": 0, "missing": 0,
             "discovery_limit_reached": False},
            {"discovered": True, "purged": 1, "conflicted": 0, "missing": 0,
             "discovery_limit_reached": False},
            {"discovered": 0, "purged": 0, "conflicted": 0, "missing": 0,
             "discovery_limit_reached": 0},
            {"discovered": 0, "purged": 0, "conflicted": 0, "missing": 0,
             "discovery_limit_reached": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                VoicemailExpiryPurgeBatchCounts(**values)

    def test_counts_classify_follow_up_for_saturation_or_unconfirmed_work(self):
        cases = (
            ({"discovered": 1, "purged": 1, "conflicted": 0, "missing": 0,
              "discovery_limit_reached": True},
             ("discovery_limit_reached",), "drain"),
            ({"discovered": 1, "purged": 0, "conflicted": 1, "missing": 0,
              "discovery_limit_reached": False}, ("conflicted",), "retry"),
            ({"discovered": 1, "purged": 0, "conflicted": 0, "missing": 1,
              "discovery_limit_reached": False}, ("no_receipt",), "retry"),
            ({"discovered": 3, "purged": 0, "conflicted": 2, "missing": 1,
              "discovery_limit_reached": True},
             ("discovery_limit_reached", "conflicted", "no_receipt"),
             "retry"),
            ({"discovered": 1, "purged": 1, "conflicted": 0,
              "missing": 0, "discovery_limit_reached": False}, (), "none"),
            ({"discovered": 0, "purged": 0, "conflicted": 0, "missing": 0,
              "discovery_limit_reached": False}, (), "none"),
        )
        for values, expected_reasons, expected_action in cases:
            with self.subTest(values=values):
                counts = VoicemailExpiryPurgeBatchCounts(**values)
                self.assertEqual(counts.follow_up_reasons, expected_reasons)
                self.assertEqual(counts.follow_up_action, expected_action)
                self.assertIs(
                    counts.follow_up_recommended, bool(expected_reasons)
                )

    def test_report_exposes_follow_up_scheduling_hint(self):
        candidate = self.add_expiry("call", "a", 8_000)
        store = Mock()
        store.list_due_expiries.return_value = (candidate,)
        store.purge_expired_if_due.return_value = None

        report = self.purge_batch_report(store=store)

        self.assertTrue(report.follow_up_recommended)
        self.assertEqual(
            report.follow_up_recommended,
            report.counts.follow_up_recommended,
        )
        self.assertEqual(report.follow_up_reasons, ("no_receipt",))
        self.assertEqual(
            report.follow_up_reasons,
            report.counts.follow_up_reasons,
        )
        self.assertEqual(report.follow_up_action, "retry")
        self.assertEqual(report.follow_up_action, report.counts.follow_up_action)

    def test_report_exposes_canonical_json_telemetry(self):
        purged = self.add_expiry("call-a", "a", 8_000)
        conflicted = self.add_expiry("call-b", "b", 9_000)
        store = Mock()
        store.list_due_expiries.return_value = (purged, conflicted)
        store.purge_expired_if_due.side_effect = (
            purged.snapshot,
            VoicemailStatusConflictError("synthetic contention"),
        )

        report = self.purge_batch_report(store=store, limit=2)
        expected = {
            "schema_version": 1,
            "discovered": 2,
            "purged": 1,
            "conflicted": 1,
            "no_receipt": 0,
            "unconfirmed": 1,
            "discovery_limit_reached": True,
            "follow_up_recommended": True,
            "follow_up_action": "retry",
            "follow_up_reasons": ("discovery_limit_reached", "conflicted"),
        }

        telemetry = report.as_telemetry()
        self.assertEqual(telemetry, expected)
        self.assertEqual(telemetry, report.counts.as_telemetry())
        self.assertEqual(
            telemetry["schema_version"],
            VOICEMAIL_EXPIRY_PURGE_TELEMETRY_SCHEMA_VERSION,
        )
        self.assertNotIn("missing", telemetry)
        self.assertEqual(json.loads(json.dumps(telemetry))["no_receipt"], 0)
        telemetry["purged"] = 99
        self.assertEqual(report.as_telemetry(), expected)

        payload = report.to_telemetry_json()
        self.assertNotIn(" ", payload)
        self.assertEqual(payload, report.counts.to_telemetry_json())
        self.assertEqual(
            VoicemailExpiryPurgeBatchCounts.from_telemetry_json(payload),
            report.counts,
        )

    def test_counts_export_deterministic_compact_json_telemetry(self):
        counts = VoicemailExpiryPurgeBatchCounts(
            discovered=3,
            purged=1,
            conflicted=1,
            missing=1,
            discovery_limit_reached=True,
        )

        payload = counts.to_telemetry_json()

        self.assertEqual(payload, counts.to_telemetry_json())
        self.assertEqual(payload, json.dumps(
            counts.as_telemetry(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ))
        self.assertEqual(
            VoicemailExpiryPurgeBatchCounts.from_telemetry_json(payload),
            counts,
        )

    def test_counts_restore_direct_and_json_round_tripped_telemetry(self):
        counts = VoicemailExpiryPurgeBatchCounts(
            discovered=3,
            purged=1,
            conflicted=1,
            missing=1,
            discovery_limit_reached=True,
        )
        direct = counts.as_telemetry()
        round_tripped = json.loads(json.dumps(direct))

        self.assertEqual(
            VoicemailExpiryPurgeBatchCounts.from_telemetry(direct), counts,
        )
        self.assertEqual(
            VoicemailExpiryPurgeBatchCounts.from_telemetry(round_tripped),
            counts,
        )

    def test_counts_restore_strict_json_telemetry(self):
        counts = VoicemailExpiryPurgeBatchCounts(
            discovered=3,
            purged=1,
            conflicted=1,
            missing=1,
            discovery_limit_reached=True,
        )
        payload = json.dumps(counts.as_telemetry())

        for encoded in (payload, payload.encode(), bytearray(payload, "utf-8")):
            with self.subTest(payload_type=type(encoded).__name__):
                self.assertEqual(
                    VoicemailExpiryPurgeBatchCounts.from_telemetry_json(encoded),
                    counts,
                )

    def test_counts_reject_ambiguous_or_nonstandard_json_telemetry(self):
        valid = json.dumps(
            VoicemailExpiryPurgeBatchCounts(
                discovered=0,
                purged=0,
                conflicted=0,
                missing=0,
                discovery_limit_reached=False,
            ).as_telemetry()
        )
        duplicate = valid.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
        )

        for payload, message in (
            (duplicate, "duplicate field"),
            (valid.replace('"discovered": 0', '"discovered": NaN'),
             "non-standard constant"),
            ("[]", "must be a mapping"),
            (b"\xff", "utf-8"),
            ({}, "text or bytes"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, message,
            ):
                VoicemailExpiryPurgeBatchCounts.from_telemetry_json(payload)

    def test_counts_reject_invalid_or_inconsistent_telemetry(self):
        valid = VoicemailExpiryPurgeBatchCounts(
            discovered=1,
            purged=0,
            conflicted=1,
            missing=0,
            discovery_limit_reached=False,
        ).as_telemetry()
        invalid = []
        for field, value in (
            ("schema_version", 2),
            ("unconfirmed", 0),
            ("follow_up_recommended", False),
            ("follow_up_action", "none"),
            ("follow_up_reasons", ("no_receipt",)),
        ):
            payload = dict(valid)
            payload[field] = value
            invalid.append(payload)
        missing_field = dict(valid)
        missing_field.pop("purged")
        invalid.append(missing_field)
        extra_field = dict(valid)
        extra_field["missing"] = 0
        invalid.append(extra_field)

        for telemetry in invalid:
            with self.subTest(telemetry=telemetry), self.assertRaises(ValueError):
                VoicemailExpiryPurgeBatchCounts.from_telemetry(telemetry)

    def test_report_rejects_incomplete_overlapping_or_reordered_outcomes(self):
        first = self.add_expiry("call-a", "a", 8_000)
        second = self.add_expiry("call-b", "b", 9_000)
        discovered = (first, second)
        invalid = (
            {"purged": (first,), "conflicted": (), "missing": ()},
            {"purged": (first,), "conflicted": (first,), "missing": (second,)},
            {"purged": (second, first), "conflicted": (), "missing": ()},
        )
        for outcomes in invalid:
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                VoicemailExpiryPurgeBatchReport(
                    discovered=discovered,
                    **outcomes,
                )

    def test_unexpected_purge_errors_propagate(self):
        candidate = self.add_expiry("call", "a", 8_000)
        store = Mock()
        store.list_due_expiries.return_value = (candidate,)
        store.purge_expired_if_due.side_effect = RuntimeError(
            "synthetic purge failure"
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic purge failure"):
            self.purge_batch(store=store)

    def test_both_adapter_capabilities_are_required_before_discovery(self):
        store = Mock(spec=["list_due_expiries"])
        with self.assertRaisesRegex(
            ValueError,
            "^store must provide a trusted callable purge_expired_if_due method$",
        ):
            self.purge_batch(store=store)
        store.list_due_expiries.assert_not_called()


if __name__ == "__main__":
    unittest.main()
