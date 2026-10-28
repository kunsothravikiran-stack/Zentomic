"""Authenticated observations through the process-local conditional store.

This sequential test harness has fake signature validation and trusted synthetic
workspace bindings. It is not a production adapter, durable session store,
automatic retry loop, atomic voice-step claim, or cleanup implementation.
"""

import copy
import unittest
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.budget import resolve_voice_step_budget
from zentomic.call_status_event import advance_call_status_event
from zentomic.call_status_store import InMemoryCallStatusStore, StatusConflictError


WORKSPACE = "synthetic-workspace"
URL = "https://example.invalid/voice/status"


class CallStatusStoreFlowTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryCallStatusStore()
        self.validator = Mock(return_value=True)

    def save(self, event, snapshot):
        # All binding and revision arguments come from the test-owned session,
        # never from fields in the callback, even when those fields are signed.
        proposed = advance_call_status_event(
            snapshot.status, event, public_url=URL, validator=self.validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
        )
        return self.store.observe(
            WORKSPACE, CALL, proposed, expected_revision=snapshot.revision,
        )

    def test_conflict_requires_reload_and_terminal_state_survives_recomputation(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for terminal in ("completed", "busy", "failed", "no-answer", "canceled"):
                    with self.subTest(version=version, encoded=encoded, terminal=terminal):
                        self.store = InMemoryCallStatusStore()
                        stale = self.store.load(WORKSPACE, CALL)
                        delayed = callback({"CallStatus": "ringing"}, version, encoded)
                        ended = self.save(
                            callback({"CallStatus": terminal}, version, encoded), stale,
                        )
                        with self.assertRaises(StatusConflictError):
                            self.save(delayed, stale)
                        self.assertEqual(self.store.load(WORKSPACE, CALL), ended)

                        # A retry must recompute from a reload, not reuse the
                        # proposal made from the stale snapshot.
                        latest = self.store.load(WORKSPACE, CALL)
                        self.validator.reset_mock()
                        retried = self.save(delayed, latest)
                        self.validator.assert_called_once_with(
                            URL, {"AccountSid": ACCOUNT, "CallSid": CALL,
                                  "CallStatus": "ringing"}, SIGNATURE,
                        )
                        self.assertEqual(retried, ended)
                        self.assertEqual((retried.status, retried.revision), (terminal, 1))
                        decision = resolve_voice_step_budget(
                            call_status=retried.status, transitions=2,
                            deadline_ms=10_000, now_ms=100, max_transitions=10,
                        )
                        self.assertEqual((decision.action, decision.transitions), ("hangup", 2))

    def test_rejected_callbacks_never_attempt_a_conditional_write(self):
        cases = (({"AccountSid": "synthetic-other-account"}, True),
                 ({"CallSid": "synthetic-child"}, True),
                 ({"CallStatus": "unknown"}, True), ({}, False))
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for overrides, valid in cases:
                    for initial in ("ringing", "completed"):
                        with self.subTest(version=version, encoded=encoded,
                                          overrides=overrides, valid=valid, initial=initial):
                            self.store = InMemoryCallStatusStore()
                            saved = self.store.observe(WORKSPACE, CALL, initial, expected_revision=0)
                            self.store.observe = Mock(wraps=self.store.observe)
                            self.validator.return_value = valid
                            event = callback({"CallStatus": "completed", **overrides}, version, encoded)
                            original = copy.deepcopy(event)
                            with self.assertRaises(ValueError):
                                self.save(event, saved)
                            self.store.observe.assert_not_called()
                            self.assertEqual(self.store.load(WORKSPACE, CALL), saved)
                            self.assertEqual(event, original)

    def test_signed_storage_fields_cannot_select_workspace_or_supply_revision(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.store = InMemoryCallStatusStore()
                    other = self.store.observe("synthetic-other", CALL, "busy", expected_revision=0)
                    initial = self.store.load(WORKSPACE, CALL)
                    fields = {"CallStatus": "ringing", "workspace_id": "synthetic-other",
                              "expected_revision": "999", "revision": "999"}
                    event = callback(fields, version, encoded)
                    original = copy.deepcopy(event)
                    saved = self.save(event, initial)
                    self.assertEqual((saved.status, saved.revision), ("ringing", 1))
                    self.assertEqual(self.store.load("synthetic-other", CALL), other)
                    self.validator.assert_called_with(
                        URL, {"AccountSid": ACCOUNT, "CallSid": CALL, **fields}, SIGNATURE,
                    )
                    self.assertEqual(event, original)

                    # A signed current revision cannot repair a stale trusted
                    # snapshot, even when the resulting observation is a no-op.
                    event = callback({**fields, "expected_revision": "1", "revision": "1"},
                                     version, encoded)
                    with self.assertRaises(StatusConflictError):
                        self.save(event, initial)
                    self.assertEqual(self.store.load(WORKSPACE, CALL), saved)
                    self.assertEqual(self.store.load("synthetic-other", CALL), other)


if __name__ == "__main__":
    unittest.main()
