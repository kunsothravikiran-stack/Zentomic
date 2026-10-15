"""Fresh per-request budget gates using synthetic clients and clocks only."""

import unittest
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import ConditionalFailure, SyntheticClient, item
from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore
from zentomic.lambda_budget import make_lambda_operation_timeout_resolver


class DynamoDBRequestAdmissionTests(unittest.TestCase):
    def store(self, client, hook):
        return DynamoDBCallStatusStore(
            client=client, table_name="demo-status", before_request=hook,
        )

    def test_construction_validates_without_invoking_the_hook_or_client(self):
        client, hook = Mock(spec=["get_item", "update_item"]), Mock()
        self.store(client, hook)
        hook.assert_not_called()
        client.get_item.assert_not_called()
        client.update_item.assert_not_called()
        for invalid in (False, True, 0, "gate", object()):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "^before_request must be a trusted callable or None$",
            ):
                self.store(client, invalid)

    def test_hook_precedes_each_read_and_write_including_noops(self):
        events = []
        client = Mock(spec=["get_item", "update_item"])
        client.get_item.side_effect = lambda **kw: events.append("read") or {"Item": item()}
        client.update_item.side_effect = lambda **kw: events.append("write") or {"Attributes": item()}
        store = self.store(client, lambda: events.append("admit"))
        self.assertEqual(store.load("demo-workspace", "demo-call"), CallStatusSnapshot("ringing", 1))
        self.assertEqual(events, ["admit", "read"])
        events.clear()
        self.assertEqual(store.observe("demo-workspace", "demo-call", "queued",
                                       expected_revision=1), CallStatusSnapshot("ringing", 1))
        self.assertEqual(events, ["admit", "read", "admit", "write"])

    def test_denied_read_never_reaches_storage_and_preserves_exception_identity(self):
        client = Mock(spec=["get_item", "update_item"])
        failure = TimeoutError("synthetic admission denied")
        hook = Mock(side_effect=failure)
        store = self.store(client, hook)
        for operation in (
            lambda: store.load("demo-workspace", "demo-call"),
            lambda: store.observe("demo-workspace", "demo-call", "ringing", expected_revision=0),
        ):
            with self.assertRaises(TimeoutError) as raised:
                operation()
            self.assertIs(raised.exception, failure)
        self.assertEqual(hook.call_count, 2)
        client.get_item.assert_not_called()
        client.update_item.assert_not_called()

    def test_write_admission_failure_is_not_reclassified_as_a_client_conflict(self):
        client = Mock(spec=["get_item", "update_item"])
        client.get_item.return_value = {}
        failure = ConditionalFailure("synthetic gate failure")
        hook = Mock(side_effect=[None, failure])
        with self.assertRaises(ConditionalFailure) as raised:
            self.store(client, hook).observe("demo-workspace", "demo-call", "ringing",
                                             expected_revision=0)
        self.assertIs(raised.exception, failure)
        client.get_item.assert_called_once()
        client.update_item.assert_not_called()

    def test_validation_and_invalid_reads_do_not_admit_a_write(self):
        client, hook = Mock(spec=["get_item", "update_item"]), Mock()
        store = self.store(client, hook)
        with self.assertRaises(ValueError):
            store.observe("demo-workspace", "demo-call", "unknown", expected_revision=0)
        hook.assert_not_called()
        client.get_item.assert_not_called()
        client.get_item.return_value = {"Item": {}}
        with self.assertRaises(ValueError):
            store.observe("demo-workspace", "demo-call", "ringing", expected_revision=0)
        hook.assert_called_once_with()
        client.update_item.assert_not_called()

    def test_hook_return_values_do_not_replace_exception_based_denial(self):
        client, hook = SyntheticClient(), Mock(return_value=False)
        store = self.store(client, hook)
        self.assertEqual(store.observe("demo-workspace", "demo-call", "ringing",
                                       expected_revision=0), CallStatusSnapshot("ringing", 1))
        self.assertEqual(hook.call_count, 2)
        self.assertEqual(len(client.writes), 1)

    def test_failed_authentication_does_not_sample_admission_or_storage(self):
        client, hook = Mock(spec=["get_item", "update_item"]), Mock()
        with self.assertRaisesRegex(ValueError, "^webhook signature validation failed$"):
            retry_call_status_event(
                callback({"CallStatus": "ringing"}, "2.0", False),
                public_url="https://example.invalid/voice/status", validator=Mock(return_value=False),
                expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                workspace_id="demo-workspace", store=self.store(client, hook),
            )
        hook.assert_not_called()
        client.get_item.assert_not_called()
        client.update_item.assert_not_called()

    def test_call_or_invocation_exhaustion_after_read_prevents_the_write(self):
        for runtime, now in (([100, 0], [0, 0]), ([100, 100], [0, 100])):
            with self.subTest(runtime=runtime, now=now):
                context = Mock(get_remaining_time_in_millis=Mock(side_effect=runtime))
                resolve = make_lambda_operation_timeout_resolver(
                    context=context, deadline_ms=100, now_ms=Mock(side_effect=now),
                    maximum_ms=50, reserve_ms=10,
                )

                def admit():
                    if resolve() is None:
                        raise TimeoutError("synthetic request budget exhausted")

                client = Mock(spec=["get_item", "update_item"])
                client.get_item.return_value = {"Item": item("completed", "2")}
                with self.assertRaises(TimeoutError):
                    self.store(client, admit).observe("demo-workspace", "demo-call", "ringing",
                                                       expected_revision=2)
                client.get_item.assert_called_once()
                client.update_item.assert_not_called()
                self.assertEqual(context.get_remaining_time_in_millis.call_count, 2)

    def test_authenticated_conflict_retry_read_is_gated_again_after_backoff(self):
        client = SyntheticClient()
        client.before_update = lambda: client.rows.update({
            ("demo-workspace", CALL): item("completed", "1", call=CALL),
        })
        failure = TimeoutError("synthetic budget spent during backoff")
        # Outer snapshot load, observe's reload, then conditional write. The
        # next attempt authenticates again before its denied snapshot load.
        hook = Mock(side_effect=[None, None, None, failure])
        validator, before_retry = Mock(return_value=True), Mock()
        with self.assertRaises(TimeoutError) as raised:
            retry_call_status_event(
                callback({"CallStatus": "ringing"}, "2.0", False),
                public_url="https://example.invalid/voice/status", validator=validator,
                expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                workspace_id="demo-workspace", store=self.store(client, hook),
                max_conflict_retries=2, before_retry=before_retry,
            )
        self.assertIs(raised.exception, failure)
        self.assertEqual(validator.call_count, 2)
        before_retry.assert_called_once_with(1)
        self.assertEqual(hook.call_count, 4)
        self.assertEqual(len(client.writes), 1)
        self.assertEqual(client.rows["demo-workspace", CALL]["call_status"], {"S": "completed"})


if __name__ == "__main__":
    unittest.main()
