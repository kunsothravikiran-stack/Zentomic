"""Immutable voicemail status persistence using synthetic identifiers only."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

from zentomic.voicemail_status import VoicemailRecordingStatus
from zentomic.voicemail_status_store import (
    InMemoryVoicemailStatusStore,
    VoicemailStatusCapacityError,
    VoicemailStatusConflictError,
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


if __name__ == "__main__":
    unittest.main()
