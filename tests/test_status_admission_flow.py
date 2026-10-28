"""Sequential lifecycle-to-admission contracts with test-owned session state.

The fake validator checks no cryptography. Assignments below are not durable
claims, replay protection, atomic writes, cleanup, or a live webhook adapter.
"""

import copy
import unittest
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.authentication import validate_call_event
from zentomic.budget import resolve_voice_step_budget
from zentomic.call_status_event import advance_call_status_event


class StatusAdmissionFlowTests(unittest.TestCase):
    def setUp(self):
        self.validator = Mock(return_value=True)
        self.state = dict(status="in-progress", transitions=0, deadline_ms=10_000)

    def authenticate(self, event, path):
        return validate_call_event(
            event, public_url="https://example.invalid" + path,
            validator=self.validator, expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
        )

    def observe(self, event):
        self.state["status"] = advance_call_status_event(
            self.state["status"], event,
            public_url="https://example.invalid/voice/status",
            validator=self.validator, expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
        )

    def admit(self, event):
        # Signed voice fields are not the persisted lifecycle/budget state.
        self.authenticate(event, "/voice/next")
        decision = resolve_voice_step_budget(
            call_status=self.state["status"], transitions=self.state["transitions"],
            deadline_ms=self.state["deadline_ms"], now_ms=100,
            max_transitions=10,
        )
        self.state["transitions"] = decision.transitions
        return decision

    def test_terminal_observations_stop_admission_despite_delayed_active_callbacks(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for terminal in ("completed", "busy", "failed", "no-answer", "canceled"):
                    with self.subTest(version=version, encoded=encoded, terminal=terminal):
                        self.state.update(status="in-progress", transitions=0)
                        voice_fields = dict(CallStatus="in-progress", transitions="0",
                                            deadline_ms="99999999", max_transitions="999")
                        voice = callback(voice_fields, version, encoded)
                        original = copy.deepcopy(voice)
                        self.assertEqual(self.admit(voice).action, "continue")
                        self.assertEqual(self.state["transitions"], 1)
                        for incoming in (terminal, "queued", "ringing", "in-progress",
                                         terminal, "failed"):
                            self.observe(callback({"CallStatus": incoming}, version, encoded))
                            before = dict(self.state)
                            decision = self.admit(voice)
                            self.assertEqual(
                                (decision.action, decision.transitions, decision.remaining_ms),
                                ("hangup", 1, 9_900),
                            )
                            self.assertEqual(self.state, before)
                            self.assertEqual(self.state["status"], terminal)
                        self.assertEqual(voice, original)
                        self.validator.assert_called_with(
                            "https://example.invalid/voice/next",
                            {"AccountSid": ACCOUNT, "CallSid": CALL, **voice_fields},
                            SIGNATURE,
                        )

    def test_rejected_status_callbacks_do_not_end_parent_or_spend_admission(self):
        cases = (({"CallSid": "synthetic-child"}, True),
                 ({"AccountSid": "synthetic-other-account"}, True),
                 ({"CallStatus": "unknown"}, True), ({}, False))
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for overrides, valid in cases:
                    with self.subTest(version=version, encoded=encoded, overrides=overrides,
                                      valid=valid):
                        self.state.update(status="in-progress", transitions=0)
                        self.validator.return_value = valid
                        event = callback({"CallStatus": "completed", **overrides}, version, encoded)
                        before = dict(self.state)
                        with self.assertRaises(ValueError):
                            self.observe(event)
                        self.assertEqual(self.state, before)
                        self.validator.return_value = True
                        result = self.admit(callback({}, version, encoded))
                        self.assertEqual((result.action, result.transitions), ("continue", 1))
                        self.assertEqual(self.state["status"], "in-progress")

    def test_reloading_after_a_status_write_invalidates_a_stale_admission(self):
        # Model only the recomputation after an adapter detects a version
        # conflict. No database, compare-and-swap, or concurrency is simulated.
        snapshot = dict(self.state)
        stale = resolve_voice_step_budget(
            call_status=snapshot["status"], transitions=snapshot["transitions"],
            deadline_ms=snapshot["deadline_ms"], now_ms=100, max_transitions=10,
        )
        self.assertEqual(stale.action, "continue")
        self.observe(callback({"CallStatus": "completed"}, "2.0", True))
        # Discard the stale result instead of persisting it after the conflict.
        fresh = self.admit(callback({}, "2.0", True))
        self.assertEqual((fresh.action, fresh.transitions), ("hangup", 0))
        self.assertEqual(self.state, {**snapshot, "status": "completed"})


if __name__ == "__main__":
    unittest.main()
