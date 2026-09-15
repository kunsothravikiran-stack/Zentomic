"""Authenticated final voicemail-status storage with synthetic data only."""

import copy
import unittest
from unittest.mock import Mock

from tests import test_voicemail_status as voicemail_fixtures
from zentomic.voicemail_status import VoicemailRecordingStatus
from zentomic.voicemail_status_event import (
    record_voicemail_recording_status_event,
)
from zentomic.voicemail_status_store import (
    InMemoryVoicemailStatusStore,
    VoicemailStatusConflictError,
)


RECORDING_SID = voicemail_fixtures.RECORDING_SID
OTHER_RECORDING_SID = "RE" + "b2" * 16
WORKSPACE = "synthetic-workspace"
CALL = "synthetic-call"
URL = "https://example.com/voice/voicemail-status"


class VoicemailStatusStoreEventTests(unittest.TestCase):
    event = voicemail_fixtures.VoicemailRecordingStatusEventTests.event

    def setUp(self):
        self.validator = Mock(return_value=True)
        self.store = InMemoryVoicemailStatusStore()

    def record(self, event, **overrides):
        config = {
            "public_url": URL,
            "validator": self.validator,
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": CALL,
            "expected_recording_sid": RECORDING_SID,
            "workspace_id": WORKSPACE,
            "store": self.store,
            "max_length": 120,
        }
        config.update(overrides)
        return record_voicemail_recording_status_event(event, **config)

    def test_authenticated_final_status_is_stored_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.store = InMemoryVoicemailStatusStore()
                    event = self.event(version, encoded)
                    original = copy.deepcopy(event)
                    saved = self.record(event)
                    self.assertEqual(
                        saved,
                        VoicemailRecordingStatus("available", RECORDING_SID, 12),
                    )
                    self.assertEqual(
                        self.store.load(WORKSPACE, CALL, RECORDING_SID), saved,
                    )
                    self.assertEqual(event, original)
                    self.assertNotIn("private-recording", repr(saved))

    def test_exact_redelivery_is_idempotent(self):
        event = self.event()
        first = self.record(event)
        self.assertEqual(self.record(event), first)
        self.assertEqual(self.store.load(WORKSPACE, CALL, RECORDING_SID), first)

    def test_contradictory_redelivery_never_overwrites_first_result(self):
        first = self.record(self.event())
        with self.assertRaisesRegex(
            VoicemailStatusConflictError,
            "^voicemail recording status conflict$",
        ):
            self.record(self.event(RecordingDuration="13"))
        self.assertEqual(self.store.load(WORKSPACE, CALL, RECORDING_SID), first)

    def test_rejected_callback_never_reaches_storage(self):
        store = Mock()
        with self.assertRaises(ValueError):
            self.record(self.event(RecordingChannels="2"), store=store)
        store.record.assert_not_called()

    def test_signed_fields_cannot_select_the_storage_key(self):
        event = self.event(
            workspace_id="synthetic-other-workspace",
            expected_call_sid="synthetic-other-call",
            expected_recording_sid=OTHER_RECORDING_SID,
        )
        saved = self.record(event)
        self.assertEqual(
            self.store.load(WORKSPACE, CALL, RECORDING_SID), saved,
        )
        self.assertIsNone(
            self.store.load("synthetic-other-workspace", CALL, RECORDING_SID),
        )
        self.assertIsNone(self.store.load(WORKSPACE, CALL, OTHER_RECORDING_SID))

    def test_recording_mismatch_never_reaches_storage(self):
        store = Mock()
        with self.assertRaisesRegex(
            ValueError, "^recording SID does not match the expected recording$",
        ):
            self.record(
                self.event(RecordingSid=OTHER_RECORDING_SID), store=store,
            )
        store.record.assert_not_called()

    def test_invalid_store_or_key_fails_before_authentication(self):
        for overrides in (
            {"store": object()},
            {"workspace_id": ""},
            {"expected_call_sid": " "},
            {"expected_recording_sid": "CA" + "1" * 32},
        ):
            with self.subTest(overrides=overrides):
                self.validator.reset_mock()
                with self.assertRaises(ValueError):
                    self.record(self.event(), **overrides)
                self.validator.assert_not_called()

    def test_malformed_or_unexpected_adapter_result_fails_closed(self):
        for returned in (
            None,
            VoicemailRecordingStatus("unavailable", RECORDING_SID, None),
        ):
            store = Mock()
            store.record.return_value = returned
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                self.record(self.event(), store=store)


if __name__ == "__main__":
    unittest.main()
