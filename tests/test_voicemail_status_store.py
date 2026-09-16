"""Immutable voicemail status persistence using synthetic identifiers only."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

from zentomic.voicemail_status import VoicemailRecordingStatus
from zentomic.voicemail_status_store import (
    delete_voicemail_recording_status,
    expire_voicemail_recording_status,
    expire_voicemail_recording_status_snapshot,
    extend_voicemail_recording_status_expiry_deadline,
    InMemoryVoicemailStatusStore,
    inspect_voicemail_recording_status,
    is_voicemail_recording_status_expired,
    purge_voicemail_recording_status_expiry,
    purge_voicemail_recording_status_expiry_snapshot,
    purge_due_voicemail_recording_status_expiry,
    schedule_voicemail_recording_status_expiry_deadline,
    unschedule_voicemail_recording_status_expiry_deadline,
    VoicemailRecordingStatusSnapshot,
    VoicemailStatusCapacityError,
    VoicemailStatusConflictError,
    VoicemailStatusExpiredError,
    load_voicemail_recording_status,
)


RECORDING_SID = "RE" + "a1" * 16


class VoicemailStatusStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVoicemailStatusStore()
        self.available = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )

    def test_record_is_idempotent_and_reads_do_not_spend_capacity(self):
        store = InMemoryVoicemailStatusStore(max_entries=1)
        self.assertIsNone(store.load("workspace", "call", RECORDING_SID))
        self.assertEqual(store.record("workspace", "call", self.available), self.available)
        self.assertEqual(store.record("workspace", "call", self.available), self.available)
        self.assertEqual(store.load("workspace", "call", RECORDING_SID), self.available)

    def test_keys_isolate_workspaces_calls_and_recordings(self):
        other_recording = "RE" + "b2" * 16
        cases = (
            ("workspace-a", "call-a", self.available),
            ("workspace-b", "call-a", self.available),
            ("workspace-a", "call-b", self.available),
            ("workspace-a", "call-a", VoicemailRecordingStatus(
                "unavailable", other_recording, None,
            )),
        )
        for workspace, call, status in cases:
            self.store.record(workspace, call, status)
        for workspace, call, status in cases:
            self.assertEqual(
                self.store.load(workspace, call, status.recording_sid), status,
            )

    def test_contradictory_final_status_never_overwrites_first_observation(self):
        self.store.record("workspace", "call", self.available)
        for conflicting in (
            VoicemailRecordingStatus("available", RECORDING_SID, 13),
            VoicemailRecordingStatus("unavailable", RECORDING_SID, None),
        ):
            with self.subTest(conflicting=conflicting), self.assertRaisesRegex(
                VoicemailStatusConflictError,
                "^voicemail recording status conflict$",
            ):
                self.store.record("workspace", "call", conflicting)
        self.assertEqual(
            self.store.load("workspace", "call", RECORDING_SID), self.available,
        )

    def test_capacity_rejects_only_new_recording_keys(self):
        store = InMemoryVoicemailStatusStore(max_entries=1)
        store.record("workspace", "call", self.available)
        self.assertEqual(store.record("workspace", "call", self.available), self.available)
        other = VoicemailRecordingStatus("unavailable", "RE" + "b2" * 16, None)
        with self.assertRaisesRegex(
            VoicemailStatusCapacityError, "^voicemail status capacity reached$",
        ):
            store.record("workspace", "call", other)
        self.assertIsNone(store.load("workspace", "call", other.recording_sid))

    def test_invalid_values_fail_without_changing_state(self):
        malformed = (
            None,
            {"availability": "available"},
            VoicemailRecordingStatus("available", RECORDING_SID, True),
            VoicemailRecordingStatus("available", RECORDING_SID, -1),
            VoicemailRecordingStatus("available", RECORDING_SID, 601),
            VoicemailRecordingStatus("unavailable", RECORDING_SID, 0),
            VoicemailRecordingStatus("pending", RECORDING_SID, None),
            VoicemailRecordingStatus("available", "CA" + "1" * 32, 1),
        )
        for status in malformed:
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.store.record("workspace", "call", status)
        self.assertIsNone(self.store.load("workspace", "call", RECORDING_SID))

    def test_identifier_validation_is_shared_with_call_and_recording_boundaries(self):
        for workspace, call, recording in (
            ("", "call", RECORDING_SID),
            ("workspace", " ", RECORDING_SID),
            ("workspace", "call", "RE" + "g" * 32),
        ):
            with self.subTest(workspace=workspace, call=call, recording=recording):
                with self.assertRaises(ValueError):
                    self.store.load(workspace, call, recording)
        self.assertIsNone(self.store.load("workspace", "call", RECORDING_SID))

    def test_concurrent_conflicting_writers_cannot_both_win(self):
        barrier = Barrier(2)
        statuses = (
            self.available,
            VoicemailRecordingStatus("unavailable", RECORDING_SID, None),
        )

        def record(status):
            barrier.wait(timeout=5)
            try:
                return self.store.record("workspace", "call", status)
            except VoicemailStatusConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(record, statuses))
        self.assertEqual(sum(result is not None for result in results), 1)
        saved = self.store.load("workspace", "call", RECORDING_SID)
        self.assertIn(saved, statuses)

    def test_capacity_must_be_a_positive_integer(self):
        for limit in (None, True, False, 0, -1, 1.0, "1"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                InMemoryVoicemailStatusStore(max_entries=limit)

    def test_conditional_delete_is_idempotent_and_releases_capacity(self):
        store = InMemoryVoicemailStatusStore(max_entries=1)
        store.record("workspace", "call", self.available)
        self.assertTrue(store.delete(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        ))
        self.assertFalse(store.delete(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        ))
        other = VoicemailRecordingStatus(
            "unavailable", "RE" + "b2" * 16, None,
        )
        self.assertEqual(store.record("workspace", "call", other), other)

    def test_stale_delete_preserves_the_first_observation(self):
        self.store.record("workspace", "call", self.available)
        stale = VoicemailRecordingStatus(
            "available", RECORDING_SID, 13,
        )
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail recording status conflict$",
        ):
            self.store.delete(
                "workspace", "call", RECORDING_SID,
                expected_status=stale,
            )
        self.assertEqual(
            self.store.load("workspace", "call", RECORDING_SID), self.available,
        )

    def test_expiry_is_idempotent_and_blocks_late_redelivery(self):
        self.assertFalse(self.store.is_expired(
            "workspace", "call", RECORDING_SID,
        ))
        self.store.record("workspace", "call", self.available)
        self.assertFalse(self.store.is_expired(
            "workspace", "call", RECORDING_SID,
        ))
        self.assertTrue(self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        ))
        self.assertIsNone(self.store.load("workspace", "call", RECORDING_SID))
        self.assertTrue(self.store.is_expired(
            "workspace", "call", RECORDING_SID,
        ))
        self.assertFalse(self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        ))
        with self.assertRaisesRegex(
            VoicemailStatusExpiredError,
            "^voicemail recording status has expired$",
        ):
            self.store.record("workspace", "call", self.available)

    def test_atomic_expiry_returns_only_the_tombstone_it_created(self):
        self.store.record("workspace", "call", self.available)
        first = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        self.assertEqual(
            first, VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
        )
        self.assertIsNone(self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        ))
        self.assertTrue(self.store.purge_expired(
            "workspace", "call", RECORDING_SID,
            expected_version=first.expiry_version,
        ))
        self.store.record("workspace", "call", self.available)
        second = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        self.assertNotEqual(first.expiry_version, second.expiry_version)

    def test_atomic_inspection_distinguishes_all_three_storage_states(self):
        missing = VoicemailRecordingStatusSnapshot()
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), missing,
        )
        self.store.record("workspace", "call", self.available)
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            VoicemailRecordingStatusSnapshot(status=self.available),
        )
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        expired = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertEqual(
            expired,
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
        )

    def test_tombstone_remains_inside_the_configured_capacity_bound(self):
        store = InMemoryVoicemailStatusStore(max_entries=1)
        store.record("workspace", "call", self.available)
        store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        other = VoicemailRecordingStatus(
            "unavailable", "RE" + "b2" * 16, None,
        )
        with self.assertRaisesRegex(
            VoicemailStatusCapacityError, "^voicemail status capacity reached$",
        ):
            store.record("workspace", "call", other)

    def test_stale_expiry_preserves_the_first_observation(self):
        self.store.record("workspace", "call", self.available)
        stale = VoicemailRecordingStatus("available", RECORDING_SID, 13)
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail recording status conflict$",
        ):
            self.store.expire(
                "workspace", "call", RECORDING_SID,
                expected_status=stale,
            )
        self.assertEqual(
            self.store.load("workspace", "call", RECORDING_SID), self.available,
        )

    def test_purging_a_tombstone_releases_capacity_and_recreation_guard(self):
        store = InMemoryVoicemailStatusStore(max_entries=1)
        store.record("workspace", "call", self.available)
        store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        expired = store.inspect("workspace", "call", RECORDING_SID)
        self.assertTrue(store.purge_expired(
            "workspace", "call", RECORDING_SID,
            expected_version=expired.expiry_version,
        ))
        self.assertFalse(store.purge_expired(
            "workspace", "call", RECORDING_SID,
            expected_version=expired.expiry_version,
        ))
        self.assertFalse(store.is_expired(
            "workspace", "call", RECORDING_SID,
        ))
        self.assertEqual(
            store.record("workspace", "call", self.available), self.available,
        )

    def test_atomic_purge_returns_only_the_tombstone_it_removed(self):
        self.store.record("workspace", "call", self.available)
        expired = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        self.assertEqual(self.store.purge_expired_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_version=expired.expiry_version,
        ), expired)
        self.assertIsNone(self.store.purge_expired_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_version=expired.expiry_version,
        ))

    def test_concurrent_atomic_purgers_issue_only_one_receipt(self):
        self.store.record("workspace", "call", self.available)
        expired = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        barrier = Barrier(2)

        def purge(_worker):
            barrier.wait(timeout=5)
            return self.store.purge_expired_and_inspect(
                "workspace", "call", RECORDING_SID,
                expected_version=expired.expiry_version,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(purge, range(2)))
        self.assertEqual(results.count(expired), 1)
        self.assertEqual(results.count(None), 1)

    def test_purging_missing_or_active_state_never_removes_metadata(self):
        self.assertFalse(self.store.purge_expired(
            "workspace", "call", RECORDING_SID, expected_version=1,
        ))
        self.store.record("workspace", "call", self.available)
        self.assertFalse(self.store.purge_expired(
            "workspace", "call", RECORDING_SID, expected_version=1,
        ))
        self.assertEqual(
            self.store.load("workspace", "call", RECORDING_SID), self.available,
        )

    def test_purge_version_must_be_a_positive_integer(self):
        self.store.record("workspace", "call", self.available)
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.available,
        )
        snapshot = self.store.inspect("workspace", "call", RECORDING_SID)
        for version in (None, True, False, 0, -1, 1.0, "1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.store.purge_expired(
                    "workspace", "call", RECORDING_SID,
                    expected_version=version,
                )
            self.assertEqual(
                self.store.inspect("workspace", "call", RECORDING_SID),
                snapshot,
            )


class LoadVoicemailRecordingStatusTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVoicemailStatusStore()
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )

    def load(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "store": self.store,
        }
        config.update(overrides)
        return load_voicemail_recording_status(**config)

    def test_missing_and_present_statuses_are_loaded_without_mutation(self):
        self.assertIsNone(self.load())
        self.store.record("workspace", "call", self.status)
        self.assertEqual(self.load(), self.status)

    def test_invalid_adapter_or_key_fails_before_loading(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load(**overrides)
            store.load.assert_not_called()

    def test_malformed_or_cross_recording_adapter_values_fail_closed(self):
        returned_values = (
            {"availability": "available"},
            VoicemailRecordingStatus("available", RECORDING_SID, True),
            VoicemailRecordingStatus(
                "available", "RE" + "b2" * 16, 12,
            ),
        )
        for returned in returned_values:
            store = Mock()
            store.load.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.load(store=store)
            store.load.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.load.side_effect = RuntimeError("synthetic adapter failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.load(store=store)


class DeleteVoicemailRecordingStatusTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def delete(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_status": self.status,
            "store": self.store,
        }
        config.update(overrides)
        return delete_voicemail_recording_status(**config)

    def test_present_and_missing_metadata_are_deleted_idempotently(self):
        self.store.record("workspace", "call", self.status)
        self.assertTrue(self.delete())
        self.assertIsNone(self.store.load("workspace", "call", RECORDING_SID))
        self.assertFalse(self.delete())

    def test_invalid_adapter_key_or_expected_value_fails_before_delete(self):
        other = VoicemailRecordingStatus(
            "available", "RE" + "b2" * 16, 12,
        )
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
            {"expected_status": other},
            {"expected_status": {"availability": "available"}},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.delete(**overrides)
            store.delete.assert_not_called()

    def test_adapter_result_must_be_an_exact_boolean(self):
        for returned in (None, 0, 1, "true"):
            store = Mock()
            store.delete.return_value = returned
            with self.subTest(returned=returned), self.assertRaisesRegex(
                ValueError, "^store delete must return an exact boolean$",
            ):
                self.delete(store=store)
            store.delete.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_status=self.status,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.delete.side_effect = RuntimeError("synthetic adapter failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.delete(store=store)


class ExpireVoicemailRecordingStatusTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def expire(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_status": self.status,
            "store": self.store,
        }
        config.update(overrides)
        return expire_voicemail_recording_status(**config)

    def test_present_and_missing_metadata_are_expired_idempotently(self):
        self.store.record("workspace", "call", self.status)
        self.assertTrue(self.expire())
        self.assertIsNone(self.store.load("workspace", "call", RECORDING_SID))
        self.assertFalse(self.expire())
        with self.assertRaises(VoicemailStatusExpiredError):
            self.store.record("workspace", "call", self.status)

    def test_invalid_adapter_key_or_expected_value_fails_before_expiry(self):
        other = VoicemailRecordingStatus(
            "available", "RE" + "b2" * 16, 12,
        )
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
            {"expected_status": other},
            {"expected_status": {"availability": "available"}},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.expire(**overrides)
            store.expire.assert_not_called()

    def test_adapter_result_must_be_an_exact_boolean(self):
        for returned in (None, 0, 1, "true"):
            store = Mock()
            store.expire.return_value = returned
            with self.subTest(returned=returned), self.assertRaisesRegex(
                ValueError, "^store expire must return an exact boolean$",
            ):
                self.expire(store=store)
            store.expire.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_status=self.status,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.expire.side_effect = RuntimeError("synthetic adapter failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.expire(store=store)


class ExpireVoicemailRecordingStatusSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def expire(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_status": self.status,
            "store": self.store,
        }
        config.update(overrides)
        return expire_voicemail_recording_status_snapshot(**config)

    def test_returns_created_tombstone_without_a_followup_read(self):
        self.assertIsNone(self.expire())
        self.store.record("workspace", "call", self.status)
        snapshot = self.expire()
        self.assertEqual(
            snapshot,
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
        )
        self.assertIsNone(self.expire())

    def test_invalid_adapter_key_or_expected_value_fails_before_expiry(self):
        other = VoicemailRecordingStatus(
            "available", "RE" + "b2" * 16, 12,
        )
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
            {"expected_status": other},
            {"expected_status": {"availability": "available"}},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.expire(**overrides)
            store.expire_and_inspect.assert_not_called()

    def test_result_must_be_none_or_an_exact_expiry_snapshot(self):
        invalid = (
            False,
            {"expired": True, "expiry_version": 1},
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(expired=True),
            VoicemailRecordingStatusSnapshot(expiry_version=1),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=True,
            ),
        )
        for returned in invalid:
            store = Mock()
            store.expire_and_inspect.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.expire(store=store)
            store.expire_and_inspect.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_status=self.status,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.expire_and_inspect.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.expire(store=store)


class IsVoicemailRecordingStatusExpiredTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def is_expired(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "store": self.store,
        }
        config.update(overrides)
        return is_voicemail_recording_status_expired(**config)

    def test_missing_active_and_expired_states_are_distinguished(self):
        self.assertFalse(self.is_expired())
        self.store.record("workspace", "call", self.status)
        self.assertFalse(self.is_expired())
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        self.assertTrue(self.is_expired())

    def test_invalid_adapter_or_key_fails_before_lookup(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.is_expired(**overrides)
            store.is_expired.assert_not_called()

    def test_adapter_result_must_be_an_exact_boolean(self):
        for returned in (None, 0, 1, "true"):
            store = Mock()
            store.is_expired.return_value = returned
            with self.subTest(returned=returned), self.assertRaisesRegex(
                ValueError, "^store is_expired must return an exact boolean$",
            ):
                self.is_expired(store=store)
            store.is_expired.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.is_expired.side_effect = RuntimeError("synthetic adapter failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.is_expired(store=store)


class InspectVoicemailRecordingStatusTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def inspect(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "store": self.store,
        }
        config.update(overrides)
        return inspect_voicemail_recording_status(**config)

    def test_missing_active_and_expired_snapshots_are_returned(self):
        self.assertEqual(self.inspect(), VoicemailRecordingStatusSnapshot())
        self.store.record("workspace", "call", self.status)
        self.assertEqual(
            self.inspect(), VoicemailRecordingStatusSnapshot(status=self.status),
        )
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        expired = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertEqual(
            self.inspect(), expired,
        )
        self.assertTrue(expired.expired)
        self.assertIsInstance(expired.expiry_version, int)

    def test_invalid_adapter_or_key_fails_before_inspection(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.inspect(**overrides)
            store.inspect.assert_not_called()

    def test_malformed_or_cross_recording_snapshots_fail_closed(self):
        other_recording = "RE" + "b2" * 16
        returned_values = (
            {"status": None, "expired": False},
            VoicemailRecordingStatusSnapshot(expired=1),
            VoicemailRecordingStatusSnapshot(expired=True),
            VoicemailRecordingStatusSnapshot(expiry_version=1),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=True,
            ),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=0,
            ),
            VoicemailRecordingStatusSnapshot(
                status=self.status, expired=True, expiry_version=1,
            ),
            VoicemailRecordingStatusSnapshot(
                status=VoicemailRecordingStatus(
                    "available", other_recording, 12,
                ),
            ),
            VoicemailRecordingStatusSnapshot(
                status=VoicemailRecordingStatus(
                    "available", RECORDING_SID, True,
                ),
            ),
        )
        for returned in returned_values:
            store = Mock()
            store.inspect.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.inspect(store=store)
            store.inspect.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.inspect.side_effect = RuntimeError("synthetic adapter failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.inspect(store=store)


class PurgeVoicemailRecordingStatusExpiryTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def purge(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
            "store": self.store,
        }
        config.update(overrides)
        return purge_voicemail_recording_status_expiry(**config)

    def test_only_an_expiry_tombstone_is_purged_idempotently(self):
        self.assertFalse(self.purge())
        self.store.record("workspace", "call", self.status)
        self.assertFalse(self.purge())
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        snapshot = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertTrue(self.purge(expected_snapshot=snapshot))
        self.assertFalse(self.purge())

    def test_stale_snapshot_cannot_purge_a_newer_tombstone(self):
        self.store.record("workspace", "call", self.status)
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        stale = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertTrue(self.purge(expected_snapshot=stale))
        self.store.record("workspace", "call", self.status)
        self.store.expire(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        current = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertNotEqual(stale.expiry_version, current.expiry_version)
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            self.purge(expected_snapshot=stale)
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), current,
        )

    def test_expected_snapshot_must_be_a_valid_expiry_observation(self):
        for snapshot in (
            None,
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(expired=True),
            VoicemailRecordingStatusSnapshot(expiry_version=1),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=0,
            ),
        ):
            store = Mock()
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.purge(expected_snapshot=snapshot, store=store)
            store.purge_expired.assert_not_called()

    def test_invalid_adapter_or_key_fails_before_purge(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.purge(**overrides)
            store.purge_expired.assert_not_called()

    def test_adapter_result_must_be_an_exact_boolean(self):
        for returned in (None, 0, 1, "true"):
            store = Mock()
            store.purge_expired.return_value = returned
            with self.subTest(returned=returned), self.assertRaisesRegex(
                ValueError,
                "^store purge_expired must return an exact boolean$",
            ):
                self.purge(store=store)
            store.purge_expired.assert_called_once_with(
                "workspace", "call", RECORDING_SID, expected_version=1,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.purge_expired.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.purge(store=store)


class PurgeVoicemailRecordingStatusExpirySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()
        self.expected = VoicemailRecordingStatusSnapshot(
            expired=True, expiry_version=1,
        )

    def purge(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": self.expected,
            "store": self.store,
        }
        config.update(overrides)
        return purge_voicemail_recording_status_expiry_snapshot(**config)

    def test_returns_removed_tombstone_without_a_followup_read(self):
        self.assertIsNone(self.purge())
        self.store.record("workspace", "call", self.status)
        expired = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        self.assertEqual(self.purge(expected_snapshot=expired), expired)
        self.assertIsNone(self.purge(expected_snapshot=expired))

    def test_stale_snapshot_cannot_purge_a_newer_tombstone(self):
        self.store.record("workspace", "call", self.status)
        first = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        self.assertEqual(self.purge(expected_snapshot=first), first)
        self.store.record("workspace", "call", self.status)
        second = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status,
        )
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            self.purge(expected_snapshot=first)
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), second,
        )

    def test_invalid_input_fails_before_purge(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
            {"expected_snapshot": None},
            {"expected_snapshot": VoicemailRecordingStatusSnapshot()},
            {"expected_snapshot": VoicemailRecordingStatusSnapshot(
                status=self.status,
            )},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.purge(**overrides)
            store.purge_expired_and_inspect.assert_not_called()

    def test_result_must_be_none_or_the_exact_expected_tombstone(self):
        invalid = (
            False,
            {"expired": True, "expiry_version": 1},
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(expired=True),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=2,
            ),
        )
        for returned in invalid:
            store = Mock()
            store.purge_expired_and_inspect.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.purge(store=store)
            store.purge_expired_and_inspect.assert_called_once_with(
                "workspace", "call", RECORDING_SID, expected_version=1,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.purge_expired_and_inspect.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.purge(store=store)


class ScheduledVoicemailExpiryTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()

    def expire(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_status": self.status,
            "purge_after_ms": 10_000,
            "store": self.store,
        }
        config.update(overrides)
        return expire_voicemail_recording_status_snapshot(**config)

    def test_returns_the_new_deadline_bearing_tombstone(self):
        self.store.record("workspace", "call", self.status)
        expired = self.expire()
        self.assertEqual(expired, VoicemailRecordingStatusSnapshot(
            expired=True, expiry_version=1, purge_after_ms=10_000,
        ))
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), expired,
        )
        self.assertIsNone(self.expire())

    def test_deadline_is_validated_before_the_adapter_call(self):
        for value in (True, False, -1, 1.0, "10000"):
            store = Mock()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.expire(purge_after_ms=value, store=store)
            store.expire_and_inspect.assert_not_called()

    def test_adapter_must_return_the_requested_deadline(self):
        for returned in (
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1, purge_after_ms=10_001,
            ),
            VoicemailRecordingStatusSnapshot(
                status=self.status, purge_after_ms=10_000,
            ),
        ):
            store = Mock()
            store.expire_and_inspect.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.expire(store=store)


class ExtendVoicemailExpiryDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()
        self.store.record("workspace", "call", self.status)
        self.expected = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status, purge_after_ms=10_000,
        )

    def extend(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": self.expected,
            "new_purge_after_ms": 20_000,
            "store": self.store,
        }
        config.update(overrides)
        return extend_voicemail_recording_status_expiry_deadline(**config)

    def test_extension_atomically_updates_the_expected_tombstone(self):
        updated = self.extend()
        self.assertEqual(updated, VoicemailRecordingStatusSnapshot(
            expired=True, expiry_version=1, purge_after_ms=20_000,
        ))
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), updated,
        )
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            purge_due_voicemail_recording_status_expiry(
                "workspace", "call", RECORDING_SID,
                expected_snapshot=self.expected, now_ms=20_000,
                store=self.store,
            )
        self.assertIsNone(purge_due_voicemail_recording_status_expiry(
            "workspace", "call", RECORDING_SID,
            expected_snapshot=updated, now_ms=19_999, store=self.store,
        ))
        self.assertEqual(purge_due_voicemail_recording_status_expiry(
            "workspace", "call", RECORDING_SID,
            expected_snapshot=updated, now_ms=20_000, store=self.store,
        ), updated)

    def test_only_a_later_strict_integer_deadline_is_accepted(self):
        for value in (None, True, False, -1, 1.0, "20000", 9_999, 10_000):
            store = Mock()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.extend(new_purge_after_ms=value, store=store)
            store.extend_expiry_deadline.assert_not_called()

    def test_expected_snapshot_and_adapter_are_validated_before_mutation(self):
        invalid_snapshots = (
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(expired=True, expiry_version=1),
        )
        for snapshot in invalid_snapshots:
            store = Mock()
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.extend(expected_snapshot=snapshot, store=store)
            store.extend_expiry_deadline.assert_not_called()
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.extend(**overrides)
            store.extend_expiry_deadline.assert_not_called()

    def test_missing_state_is_a_noop_and_stale_snapshot_conflicts(self):
        empty = InMemoryVoicemailStatusStore()
        self.assertIsNone(self.extend(store=empty))
        updated = self.extend()
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            self.extend()
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), updated,
        )

    def test_adapter_result_must_match_the_requested_tombstone(self):
        invalid = (
            False,
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=2, purge_after_ms=20_000,
            ),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1, purge_after_ms=20_001,
            ),
        )
        for returned in invalid:
            store = Mock()
            store.extend_expiry_deadline.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.extend(store=store)
            store.extend_expiry_deadline.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_version=1,
                expected_purge_after_ms=10_000,
                new_purge_after_ms=20_000,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.extend_expiry_deadline.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.extend(store=store)


class UnscheduleVoicemailExpiryTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()
        self.store.record("workspace", "call", self.status)
        self.expected = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status, purge_after_ms=10_000,
        )

    def unschedule(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": self.expected,
            "store": self.store,
        }
        config.update(overrides)
        return unschedule_voicemail_recording_status_expiry_deadline(**config)

    def test_unschedule_atomically_removes_only_the_deadline(self):
        held = self.unschedule()
        self.assertEqual(held, VoicemailRecordingStatusSnapshot(
            expired=True, expiry_version=1,
        ))
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), held,
        )
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            purge_due_voicemail_recording_status_expiry(
                "workspace", "call", RECORDING_SID,
                expected_snapshot=self.expected, now_ms=10_000,
                store=self.store,
            )
        self.assertEqual(
            purge_voicemail_recording_status_expiry_snapshot(
                "workspace", "call", RECORDING_SID,
                expected_snapshot=held, store=self.store,
            ),
            held,
        )

    def test_due_purge_and_retention_hold_cannot_both_win(self):
        barrier = Barrier(2)

        def unschedule():
            barrier.wait(timeout=5)
            return self.unschedule()

        def purge():
            barrier.wait(timeout=5)
            try:
                return purge_due_voicemail_recording_status_expiry(
                    "workspace", "call", RECORDING_SID,
                    expected_snapshot=self.expected, now_ms=10_000,
                    store=self.store,
                )
            except VoicemailStatusConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda operation: operation(), (
                unschedule, purge,
            )))

        self.assertEqual(sum(result is not None for result in results), 1)
        state = self.store.inspect("workspace", "call", RECORDING_SID)
        self.assertIn(state, (
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
        ))

    def test_concurrent_retention_holds_issue_only_one_receipt(self):
        barrier = Barrier(2)

        def unschedule():
            barrier.wait(timeout=5)
            try:
                return self.unschedule()
            except VoicemailStatusConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: unschedule(), range(2)))

        winners = [result for result in results if result is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(
            winners[0],
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
        )
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            winners[0],
        )

    def test_expected_snapshot_and_adapter_are_validated_before_mutation(self):
        invalid_snapshots = (
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(expired=True, expiry_version=1),
        )
        for snapshot in invalid_snapshots:
            store = Mock()
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.unschedule(expected_snapshot=snapshot, store=store)
            store.unschedule_expiry_deadline.assert_not_called()
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.unschedule(**overrides)
            store.unschedule_expiry_deadline.assert_not_called()

    def test_missing_or_active_state_is_a_noop_and_stale_snapshot_conflicts(self):
        empty = InMemoryVoicemailStatusStore()
        self.assertIsNone(self.unschedule(store=empty))
        active = InMemoryVoicemailStatusStore()
        active.record("workspace", "call", self.status)
        self.assertIsNone(self.unschedule(store=active))
        self.assertEqual(
            active.load("workspace", "call", RECORDING_SID), self.status,
        )
        held = self.unschedule()
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            self.unschedule()
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), held,
        )

    def test_local_store_rejects_invalid_expectations_without_mutation(self):
        cases = (
            {"expected_version": None, "expected_purge_after_ms": 10_000},
            {"expected_version": True, "expected_purge_after_ms": 10_000},
            {"expected_version": 0, "expected_purge_after_ms": 10_000},
            {"expected_version": 1.0, "expected_purge_after_ms": 10_000},
            {"expected_version": 1, "expected_purge_after_ms": None},
            {"expected_version": 1, "expected_purge_after_ms": True},
            {"expected_version": 1, "expected_purge_after_ms": -1},
            {"expected_version": 1, "expected_purge_after_ms": 10_000.0},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.unschedule_expiry_deadline(
                    "workspace", "call", RECORDING_SID, **kwargs,
                )
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            self.expected,
        )

    def test_adapter_result_must_match_the_unscheduled_tombstone(self):
        invalid = (
            False,
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=2,
            ),
            self.expected,
        )
        for returned in invalid:
            store = Mock()
            store.unschedule_expiry_deadline.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.unschedule(store=store)
            store.unschedule_expiry_deadline.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_version=1,
                expected_purge_after_ms=10_000,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.unschedule_expiry_deadline.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.unschedule(store=store)


class ScheduleVoicemailExpiryTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()
        self.store.record("workspace", "call", self.status)
        scheduled = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status, purge_after_ms=10_000,
        )
        self.expected = unschedule_voicemail_recording_status_expiry_deadline(
            "workspace", "call", RECORDING_SID,
            expected_snapshot=scheduled, store=self.store,
        )

    def schedule(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": self.expected,
            "new_purge_after_ms": 20_000,
            "store": self.store,
        }
        config.update(overrides)
        return schedule_voicemail_recording_status_expiry_deadline(**config)

    def test_schedule_atomically_restores_timed_cleanup(self):
        scheduled = self.schedule()
        self.assertEqual(scheduled, VoicemailRecordingStatusSnapshot(
            expired=True, expiry_version=1, purge_after_ms=20_000,
        ))
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), scheduled,
        )
        with self.assertRaisesRegex(
            ValueError,
            "^scheduled expiry requires a due-purge adapter$",
        ):
            purge_voicemail_recording_status_expiry_snapshot(
                "workspace", "call", RECORDING_SID,
                expected_snapshot=self.expected, store=self.store,
            )
        self.assertIsNone(purge_due_voicemail_recording_status_expiry(
            "workspace", "call", RECORDING_SID,
            expected_snapshot=scheduled, now_ms=19_999, store=self.store,
        ))
        self.assertEqual(purge_due_voicemail_recording_status_expiry(
            "workspace", "call", RECORDING_SID,
            expected_snapshot=scheduled, now_ms=20_000, store=self.store,
        ), scheduled)

    def test_concurrent_hold_releases_cannot_overwrite_each_other(self):
        barrier = Barrier(2)

        def schedule(deadline):
            barrier.wait(timeout=5)
            try:
                return self.schedule(new_purge_after_ms=deadline)
            except VoicemailStatusConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(schedule, (20_000, 30_000)))

        winners = [result for result in results if result is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            winners[0],
        )
        self.assertIn(winners[0].purge_after_ms, (20_000, 30_000))

    def test_deadline_is_validated_before_the_adapter_call(self):
        for value in (None, True, False, -1, 1.0, "20000"):
            store = Mock()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.schedule(new_purge_after_ms=value, store=store)
            store.schedule_expiry_deadline.assert_not_called()

    def test_expected_snapshot_and_adapter_are_validated_before_mutation(self):
        invalid_snapshots = (
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1, purge_after_ms=10_000,
            ),
        )
        for snapshot in invalid_snapshots:
            store = Mock()
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.schedule(expected_snapshot=snapshot, store=store)
            store.schedule_expiry_deadline.assert_not_called()
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"call_sid": " "},
            {"recording_sid": "CA" + "1" * 32},
        ):
            store = Mock()
            if "store" not in overrides:
                overrides["store"] = store
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.schedule(**overrides)
            store.schedule_expiry_deadline.assert_not_called()

    def test_missing_or_active_state_is_a_noop_and_stale_snapshot_conflicts(self):
        empty = InMemoryVoicemailStatusStore()
        self.assertIsNone(self.schedule(store=empty))
        active = InMemoryVoicemailStatusStore()
        active.record("workspace", "call", self.status)
        self.assertIsNone(self.schedule(store=active))
        self.assertEqual(
            active.load("workspace", "call", RECORDING_SID), self.status,
        )
        scheduled = self.schedule()
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail expiry tombstone conflict$",
        ):
            self.schedule()
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID), scheduled,
        )

    def test_local_store_rejects_invalid_expectations_without_mutation(self):
        cases = (
            {"expected_version": None, "new_purge_after_ms": 20_000},
            {"expected_version": True, "new_purge_after_ms": 20_000},
            {"expected_version": 0, "new_purge_after_ms": 20_000},
            {"expected_version": 1.0, "new_purge_after_ms": 20_000},
            {"expected_version": 1, "new_purge_after_ms": None},
            {"expected_version": 1, "new_purge_after_ms": True},
            {"expected_version": 1, "new_purge_after_ms": -1},
            {"expected_version": 1, "new_purge_after_ms": 20_000.0},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.schedule_expiry_deadline(
                    "workspace", "call", RECORDING_SID, **kwargs,
                )
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            self.expected,
        )

    def test_adapter_result_must_match_the_scheduled_tombstone(self):
        invalid = (
            False,
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=2, purge_after_ms=20_000,
            ),
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1, purge_after_ms=20_001,
            ),
        )
        for returned in invalid:
            store = Mock()
            store.schedule_expiry_deadline.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.schedule(store=store)
            store.schedule_expiry_deadline.assert_called_once_with(
                "workspace", "call", RECORDING_SID,
                expected_version=1,
                new_purge_after_ms=20_000,
            )

    def test_adapter_errors_propagate(self):
        store = Mock()
        store.schedule_expiry_deadline.side_effect = RuntimeError(
            "synthetic adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self.schedule(store=store)


class PurgeDueVoicemailExpiryTests(unittest.TestCase):
    def setUp(self):
        self.status = VoicemailRecordingStatus(
            "available", RECORDING_SID, 12,
        )
        self.store = InMemoryVoicemailStatusStore()
        self.store.record("workspace", "call", self.status)
        self.expected = self.store.expire_and_inspect(
            "workspace", "call", RECORDING_SID,
            expected_status=self.status, purge_after_ms=10_000,
        )

    def purge(self, **overrides):
        config = {
            "workspace_id": "workspace",
            "call_sid": "call",
            "recording_sid": RECORDING_SID,
            "expected_snapshot": self.expected,
            "now_ms": 10_000,
            "store": self.store,
        }
        config.update(overrides)
        return purge_due_voicemail_recording_status_expiry(**config)

    def test_not_due_is_a_noop_and_exact_deadline_returns_receipt(self):
        self.assertIsNone(self.purge(now_ms=9_999))
        with self.assertRaisesRegex(
            ValueError, "^scheduled expiry requires a due-purge adapter$",
        ):
            self.store.purge_expired(
                "workspace", "call", RECORDING_SID, expected_version=1,
            )
        self.assertEqual(
            self.store.inspect("workspace", "call", RECORDING_SID),
            self.expected,
        )
        self.assertEqual(self.purge(), self.expected)
        self.assertIsNone(self.purge())

    def test_before_deadline_does_not_reach_the_adapter(self):
        store = Mock()
        self.assertIsNone(self.purge(now_ms=9_999, store=store))
        store.purge_expired_if_due.assert_not_called()

    def test_now_and_expected_snapshot_are_strictly_validated(self):
        invalid_snapshots = (
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1,
            ),
            VoicemailRecordingStatusSnapshot(),
            VoicemailRecordingStatusSnapshot(status=self.status),
        )
        for snapshot in invalid_snapshots:
            store = Mock()
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.purge(expected_snapshot=snapshot, store=store)
            store.purge_expired_if_due.assert_not_called()
        for value in (None, True, False, -1, 1.0, "10000"):
            store = Mock()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.purge(now_ms=value, store=store)
            store.purge_expired_if_due.assert_not_called()

    def test_legacy_purge_helpers_cannot_bypass_a_stored_deadline(self):
        for helper in (
            purge_voicemail_recording_status_expiry,
            purge_voicemail_recording_status_expiry_snapshot,
        ):
            store = Mock()
            with self.subTest(helper=helper), self.assertRaisesRegex(
                ValueError, "^scheduled expiry requires a due-purge adapter$",
            ):
                helper(
                    "workspace", "call", RECORDING_SID,
                    expected_snapshot=self.expected, store=store,
                )
            store.purge_expired.assert_not_called()
            store.purge_expired_and_inspect.assert_not_called()

    def test_adapter_result_must_match_the_expected_tombstone(self):
        store = Mock()
        store.purge_expired_if_due.return_value = (
            VoicemailRecordingStatusSnapshot(
                expired=True, expiry_version=1, purge_after_ms=10_001,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "^store due purge result must match the expected tombstone$",
        ):
            self.purge(store=store)
        store.purge_expired_if_due.assert_called_once_with(
            "workspace", "call", RECORDING_SID,
            expected_version=1,
            expected_purge_after_ms=10_000,
            now_ms=10_000,
        )


if __name__ == "__main__":
    unittest.main()
