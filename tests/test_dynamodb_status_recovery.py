"""Exercise lost write acknowledgments without an SDK or service connection."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import SyntheticClient, item
from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore


class AmbiguousWriteClient(SyntheticClient):
    """Persist the first update, then lose or corrupt its acknowledgment."""

    def __init__(self, fault):
        super().__init__()
        self.fault = fault
        self.reads = 0

    def get_item(self, **request):
        self.reads += 1
        return super().get_item(**request)

    def update_item(self, **request):
        result = super().update_item(**request)
        fault, self.fault = self.fault, None
        if isinstance(fault, Exception):
            raise fault
        return fault if fault is not None else result


class DynamoDBStatusRecoveryTests(unittest.TestCase):
    def observe(self, client, status, *, validator, before_retry):
        return retry_call_status_event(
            callback({"CallStatus": status}, "2.0", False),
            public_url="https://example.invalid/voice/status", validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id="demo-workspace",
            store=DynamoDBCallStatusStore(client=client, table_name="demo-status"),
            max_conflict_retries=8, before_retry=before_retry,
        )

    def test_committed_but_unacknowledged_writes_are_not_conflict_retried(self):
        for initial, incoming, expected in (
            (None, "ringing", CallStatusSnapshot("ringing", 1)),
            (item(call=CALL), "completed", CallStatusSnapshot("completed", 2)),
            (item("completed", "2", call=CALL), "ringing",
             CallStatusSnapshot("completed", 2)),
        ):
            for fault in (TimeoutError("synthetic lost acknowledgment"),
                          {}, {"Attributes": item("failed", "9", call=CALL)}):
                with self.subTest(initial=initial, incoming=incoming, fault=fault):
                    client = AmbiguousWriteClient(fault)
                    if initial is not None:
                        client.rows["demo-workspace", CALL] = deepcopy(initial)
                    validator, before_retry = Mock(return_value=True), Mock()
                    failure_type = TimeoutError if isinstance(fault, Exception) else ValueError
                    with self.assertRaises(failure_type) as raised:
                        self.observe(client, incoming, validator=validator,
                                     before_retry=before_retry)
                    if isinstance(fault, Exception):
                        self.assertIs(raised.exception, fault)
                    self.assertEqual(client.reads, 2)
                    self.assertEqual(len(client.writes), 1)
                    validator.assert_called_once()
                    before_retry.assert_not_called()

                    # Failure is not rollback: a consistent reload sees the write.
                    store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
                    self.assertEqual(store.load("demo-workspace", CALL), expected)
                    # A later authenticated delivery reconciles without advancing
                    # the revision again, even when the original write was a no-op.
                    self.assertEqual(self.observe(client, incoming, validator=validator,
                                                  before_retry=before_retry), expected)
                    self.assertEqual(len(client.writes), 2)
                    self.assertEqual(validator.call_count, 2)
                    before_retry.assert_not_called()

    def test_redelivery_preserves_a_terminal_observation_after_an_uncertain_write(self):
        client = AmbiguousWriteClient(TimeoutError("synthetic lost acknowledgment"))
        validator, before_retry = Mock(return_value=True), Mock()
        with self.assertRaises(TimeoutError):
            self.observe(client, "ringing", validator=validator, before_retry=before_retry)
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
        terminal = store.observe("demo-workspace", CALL, "completed", expected_revision=1)
        self.assertEqual(terminal, CallStatusSnapshot("completed", 2))
        for delayed in ("ringing", "queued", "failed", "completed"):
            self.assertEqual(self.observe(client, delayed, validator=validator,
                                          before_retry=before_retry), terminal)
        self.assertEqual(store.load("demo-workspace", CALL), terminal)
        self.assertEqual(validator.call_count, 5)
        self.assertEqual(len(client.writes), 6)
        before_retry.assert_not_called()

    def test_recovery_still_authenticates_before_any_storage_access(self):
        client = AmbiguousWriteClient(TimeoutError("synthetic lost acknowledgment"))
        validator, before_retry = Mock(return_value=True), Mock()
        with self.assertRaises(TimeoutError):
            self.observe(client, "completed", validator=validator, before_retry=before_retry)
        saved = deepcopy(client.rows)
        reads, writes = client.reads, len(client.writes)
        validator.return_value = False
        with self.assertRaisesRegex(ValueError, "^webhook signature validation failed$"):
            self.observe(client, "ringing", validator=validator, before_retry=before_retry)
        self.assertEqual(client.rows, saved)
        self.assertEqual(client.reads, reads)
        self.assertEqual(len(client.writes), writes)
        self.assertEqual(validator.call_count, 2)
        before_retry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
