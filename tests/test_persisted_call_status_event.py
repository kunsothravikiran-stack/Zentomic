"""Authenticated lifecycle callbacks through an injectable conditional store."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.call_status_event import observe_call_status_event
from zentomic.call_status_store import (
    CallStatusSnapshot,
    InMemoryCallStatusStore,
    StatusConflictError,
)


WORKSPACE = "synthetic-workspace"
URL = "https://example.invalid/voice/status"


class PersistedCallStatusEventTests(unittest.TestCase):
    def setUp(self):
        self.validator = Mock(return_value=True)
        self.store = InMemoryCallStatusStore()

    def save(self, event, *, store=None, **overrides):
        config = dict(
            public_url=URL,
            validator=self.validator,
            expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
            workspace_id=WORKSPACE,
            store=store or self.store,
        )
        config.update(overrides)
        return observe_call_status_event(event, **config)

    def test_proxy_formats_store_monotonic_status_and_revision(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.store = InMemoryCallStatusStore()
                    history = []
                    for status in (
                        "ringing", "queued", "in-progress", "completed", "ringing", "failed",
                    ):
                        event = callback({"CallStatus": status}, version, encoded)
                        original = copy.deepcopy(event)
                        saved = self.save(event)
                        history.append((saved.status, saved.revision))
                        self.assertEqual(event, original)
                    self.assertEqual(
                        history,
                        [("ringing", 1), ("ringing", 1), ("in-progress", 2)]
                        + [("completed", 3)] * 3,
                    )

    def test_rejected_callbacks_never_load_or_write_status(self):
        cases = (
            ({"CallSid": "synthetic-child"}, True),
            ({"AccountSid": "synthetic-other"}, True),
            ({"CallStatus": "unknown"}, True),
            ({}, False),
        )
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for overrides, valid in cases:
                    with self.subTest(version=version, encoded=encoded, overrides=overrides):
                        self.store = InMemoryCallStatusStore()
                        self.store.load = Mock(wraps=self.store.load)
                        self.store.observe = Mock(wraps=self.store.observe)
                        self.validator.return_value = valid
                        event = callback(
                            {"CallStatus": "completed", **overrides}, version, encoded,
                        )
                        with self.assertRaises(ValueError):
                            self.save(event)
                        self.store.load.assert_not_called()
                        self.store.observe.assert_not_called()

    def test_signed_storage_fields_cannot_select_key_or_revision(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.store = InMemoryCallStatusStore()
                    other = self.store.observe(
                        "untrusted", CALL, "busy", expected_revision=0,
                    )
                    self.store.load = Mock(wraps=self.store.load)
                    self.store.observe = Mock(wraps=self.store.observe)
                    fields = {
                        "CallStatus": "ringing", "workspace_id": "untrusted",
                        "revision": "999", "expected_revision": "999",
                    }
                    saved = self.save(callback(fields, version, encoded))
                    self.assertEqual(saved, CallStatusSnapshot("ringing", 1))
                    self.store.load.assert_called_once_with(WORKSPACE, CALL)
                    self.store.observe.assert_called_once_with(
                        WORKSPACE, CALL, "ringing", expected_revision=0,
                    )
                    self.assertEqual(self.store.load("untrusted", CALL), other)
                    self.validator.assert_called_with(
                        URL, {"AccountSid": ACCOUNT, "CallSid": CALL, **fields}, SIGNATURE,
                    )

    def test_concurrent_change_conflicts_and_retry_reloads_terminal_state(self):
        initial = self.store.load(WORKSPACE, CALL)
        original_observe = self.store.observe

        def competing_observe(workspace, call, incoming, *, expected_revision):
            original_observe(workspace, call, "completed", expected_revision=initial.revision)
            return original_observe(
                workspace, call, incoming, expected_revision=expected_revision,
            )

        self.store.observe = competing_observe
        event = callback({"CallStatus": "ringing"}, "2.0", True)
        with self.assertRaisesRegex(StatusConflictError, "^call status revision conflict$"):
            self.save(event)
        self.assertEqual(self.store.load(WORKSPACE, CALL), CallStatusSnapshot("completed", 1))

        self.store.observe = original_observe
        self.assertEqual(self.save(event), CallStatusSnapshot("completed", 1))
        self.assertEqual(self.validator.call_count, 2)

    def test_invalid_trusted_store_configuration_precedes_authentication(self):
        for overrides in (
            {"workspace_id": ""},
            {"expected_call_sid": "x" * 257},
            {"store": object()},
            {"store": SimpleNamespace(load=Mock())},
        ):
            self.validator.reset_mock()
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.save(callback({"CallStatus": "completed"}, "2.0", False), **overrides)
            self.validator.assert_not_called()

    def test_malformed_store_snapshots_fail_closed(self):
        for snapshot in (
            object(),
            CallStatusSnapshot("private-invalid", 0),
            CallStatusSnapshot("ringing", 0),
            CallStatusSnapshot(None, 1),
            CallStatusSnapshot("ringing", True),
            CallStatusSnapshot("ringing", -1),
        ):
            store = Mock()
            store.load.return_value = snapshot
            with self.subTest(snapshot=snapshot), self.assertRaisesRegex(
                ValueError, "CallStatusSnapshot",
            ):
                self.save(callback({"CallStatus": "completed"}, "2.0", False), store=store)
            store.observe.assert_not_called()

    def test_store_result_must_match_the_requested_transition(self):
        cases = (
            (CallStatusSnapshot(), "ringing", CallStatusSnapshot("queued", 1)),
            (CallStatusSnapshot(), "ringing", CallStatusSnapshot("ringing", 2)),
            (CallStatusSnapshot("ringing", 1), "completed",
             CallStatusSnapshot("ringing", 1)),
            (CallStatusSnapshot("completed", 2), "ringing",
             CallStatusSnapshot("busy", 3)),
            (CallStatusSnapshot("in-progress", 4), "queued",
             CallStatusSnapshot("in-progress", 5)),
        )
        for loaded, incoming, returned in cases:
            store = Mock()
            store.load.return_value = loaded
            store.observe.return_value = returned
            with self.subTest(loaded=loaded, incoming=incoming, returned=returned):
                with self.assertRaisesRegex(
                    ValueError, "^store returned an unexpected CallStatusSnapshot$",
                ):
                    self.save(
                        callback({"CallStatus": incoming}, "2.0", False), store=store,
                    )
                store.observe.assert_called_once_with(
                    WORKSPACE, CALL, incoming, expected_revision=loaded.revision,
                )


if __name__ == "__main__":
    unittest.main()
