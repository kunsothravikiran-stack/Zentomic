"""Authenticated lifecycle callbacks through an injectable conditional store."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.call_status_event import (
    MAX_STATUS_CONFLICT_RETRIES,
    MAX_STATUS_RETRY_DELAY_MS,
    observe_call_status_event,
    retry_call_status_event,
    status_conflict_retry_delay_ms,
)
from zentomic.call_status_store import (
    CallStatusSnapshot,
    InMemoryCallStatusStore,
    StatusConflictError,
)


WORKSPACE = "synthetic-workspace"
URL = "https://example.invalid/voice/status"


class StatusConflictRetryDelayTests(unittest.TestCase):
    def test_default_delay_grows_exponentially_and_caps(self):
        self.assertEqual(
            [status_conflict_retry_delay_ms(retry) for retry in range(1, 9)],
            [25, 50, 100, 200, 400, 400, 400, 400],
        )

    def test_custom_delay_boundaries_are_exact(self):
        self.assertEqual(
            status_conflict_retry_delay_ms(
                1, base_delay_ms=1, max_delay_ms=MAX_STATUS_RETRY_DELAY_MS,
            ),
            1,
        )
        self.assertEqual(
            status_conflict_retry_delay_ms(
                8, base_delay_ms=MAX_STATUS_RETRY_DELAY_MS,
                max_delay_ms=MAX_STATUS_RETRY_DELAY_MS,
            ),
            MAX_STATUS_RETRY_DELAY_MS,
        )

    def test_configuration_is_strict(self):
        cases = (
            ({"retry_number": 0}, "retry_number"),
            ({"retry_number": 9}, "retry_number"),
            ({"retry_number": True}, "retry_number"),
            ({"retry_number": 1.0}, "retry_number"),
            ({"retry_number": 1, "base_delay_ms": 0}, "base_delay_ms"),
            ({"retry_number": 1, "base_delay_ms": True}, "base_delay_ms"),
            ({"retry_number": 1, "base_delay_ms": 10_001}, "base_delay_ms"),
            ({"retry_number": 1, "max_delay_ms": 0}, "max_delay_ms"),
            ({"retry_number": 1, "max_delay_ms": False}, "max_delay_ms"),
            ({"retry_number": 1, "max_delay_ms": 10_001}, "max_delay_ms"),
            ({"retry_number": 1, "base_delay_ms": 2, "max_delay_ms": 1},
             "must not exceed"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, message,
            ):
                status_conflict_retry_delay_ms(**arguments)

    def test_policy_has_no_runtime_dependencies(self):
        with patch("time.sleep") as sleep:
            self.assertEqual(status_conflict_retry_delay_ms(3), 100)
        sleep.assert_not_called()


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


class RetriedCallStatusEventTests(unittest.TestCase):
    def setUp(self):
        self.validator = Mock(return_value=True)
        self.event = callback({"CallStatus": "ringing"}, "2.0", True)

    def save(self, store, **overrides):
        config = dict(
            public_url=URL,
            validator=self.validator,
            expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
            workspace_id=WORKSPACE,
            store=store,
        )
        config.update(overrides)
        return retry_call_status_event(self.event, **config)

    def test_conflict_reloads_reauthenticates_and_then_succeeds(self):
        store = InMemoryCallStatusStore()
        store.load = Mock(wraps=store.load)
        observe = Mock(side_effect=[
            StatusConflictError("synthetic conflict"),
            CallStatusSnapshot("ringing", 1),
        ])
        store.observe = observe
        original = copy.deepcopy(self.event)

        saved = self.save(store)

        self.assertEqual(saved, CallStatusSnapshot("ringing", 1))
        self.assertEqual(self.validator.call_count, 2)
        self.assertEqual(store.load.call_count, 2)
        self.assertEqual(observe.call_count, 2)
        self.assertEqual(store.load(WORKSPACE, CALL), CallStatusSnapshot())
        self.assertEqual(self.event, original)

    def test_retry_limit_rethrows_the_final_conflict(self):
        store = Mock()
        store.load.return_value = CallStatusSnapshot()
        store.observe.side_effect = StatusConflictError("synthetic conflict")
        before_retry = Mock()

        with self.assertRaisesRegex(StatusConflictError, "^synthetic conflict$"):
            self.save(
                store, max_conflict_retries=3, before_retry=before_retry,
            )

        self.assertEqual(self.validator.call_count, 4)
        self.assertEqual(store.load.call_count, 4)
        self.assertEqual(store.observe.call_count, 4)
        self.assertEqual(before_retry.call_args_list, [call(1), call(2), call(3)])

    def test_retry_hook_runs_only_between_conflicted_attempts(self):
        store = Mock()
        store.load.return_value = CallStatusSnapshot()
        store.observe.side_effect = [
            StatusConflictError("first conflict"),
            StatusConflictError("second conflict"),
            CallStatusSnapshot("ringing", 1),
        ]
        before_retry = Mock()

        saved = self.save(store, before_retry=before_retry)

        self.assertEqual(saved, CallStatusSnapshot("ringing", 1))
        self.assertEqual(before_retry.call_args_list, [call(1), call(2)])
        self.assertEqual(self.validator.call_count, 3)

    def test_retry_hook_failure_aborts_before_reauthentication(self):
        store = Mock()
        store.load.return_value = CallStatusSnapshot()
        store.observe.side_effect = StatusConflictError("synthetic conflict")
        failure = RuntimeError("synthetic backoff failure")
        before_retry = Mock(side_effect=failure)

        with self.assertRaises(RuntimeError) as caught:
            self.save(store, before_retry=before_retry)

        self.assertIs(caught.exception, failure)
        before_retry.assert_called_once_with(1)
        self.validator.assert_called_once()
        store.load.assert_called_once_with(WORKSPACE, CALL)
        store.observe.assert_called_once_with(
            WORKSPACE, CALL, "ringing", expected_revision=0,
        )

    def test_zero_retries_preserves_single_attempt_behavior(self):
        store = Mock()
        store.load.return_value = CallStatusSnapshot()
        conflict = StatusConflictError("synthetic conflict")
        store.observe.side_effect = conflict

        with self.assertRaises(StatusConflictError) as caught:
            self.save(store, max_conflict_retries=0)

        self.assertIs(caught.exception, conflict)
        self.validator.assert_called_once()
        store.load.assert_called_once_with(WORKSPACE, CALL)
        store.observe.assert_called_once_with(
            WORKSPACE, CALL, "ringing", expected_revision=0,
        )

    def test_only_conflicts_are_retried(self):
        for failure in (
            ValueError("synthetic invalid adapter"),
            RuntimeError("synthetic unavailable store"),
        ):
            store = Mock()
            store.load.side_effect = failure
            self.validator.reset_mock()
            with self.subTest(failure=type(failure)), self.assertRaises(
                type(failure),
            ) as caught:
                self.save(store, max_conflict_retries=8)
            self.assertIs(caught.exception, failure)
            self.validator.assert_called_once()
            store.load.assert_called_once_with(WORKSPACE, CALL)
            store.observe.assert_not_called()

    def test_conflict_from_load_is_not_mistaken_for_a_write_conflict(self):
        store = Mock()
        conflict = StatusConflictError("synthetic load failure")
        store.load.side_effect = conflict
        before_retry = Mock()

        with self.assertRaises(StatusConflictError) as caught:
            self.save(
                store, max_conflict_retries=8, before_retry=before_retry,
            )

        self.assertIs(caught.exception, conflict)
        self.validator.assert_called_once()
        store.load.assert_called_once_with(WORKSPACE, CALL)
        store.observe.assert_not_called()
        before_retry.assert_not_called()

    def test_retry_configuration_is_strict_and_checked_before_authentication(self):
        invalid = (
            -1, MAX_STATUS_CONFLICT_RETRIES + 1, True, False, 1.0, "2", None,
        )
        for value in invalid:
            self.validator.reset_mock()
            store = Mock()
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "^max_conflict_retries must be an integer from 0 to 8$",
            ):
                self.save(store, max_conflict_retries=value)
            self.validator.assert_not_called()
            store.load.assert_not_called()
            store.observe.assert_not_called()

    def test_retry_hook_configuration_is_checked_before_authentication(self):
        for value in (False, 0, "hook", object()):
            self.validator.reset_mock()
            store = Mock()
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "^before_retry must be callable or None$",
            ):
                self.save(store, before_retry=value)
            self.validator.assert_not_called()
            store.load.assert_not_called()
            store.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
