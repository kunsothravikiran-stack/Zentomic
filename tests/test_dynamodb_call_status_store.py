"""Synthetic low-level DynamoDB requests; no SDK, credentials or network."""

import unittest
from copy import deepcopy
from itertools import product
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import (
    CallStatusSnapshot, InMemoryCallStatusStore, StatusConflictError,
)
from zentomic.dynamodb_call_status_store import (
    MAX_STATUS_REVISION, DynamoDBCallStatusStore,
)


def item(status="ringing", revision="1", workspace="demo-workspace", call="demo-call"):
    return {
        "workspace_id": {"S": workspace}, "call_sid": {"S": call},
        "call_status": {"S": status}, "status_revision": {"N": revision},
    }


class ConditionalFailure(Exception):
    response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class SyntheticClient:
    """Model only this adapter's two conditions, not a DynamoDB emulator."""

    def __init__(self):
        self.rows = {}
        self.writes = []
        self.before_update = None

    def get_item(self, **request):
        key = (request["Key"]["workspace_id"]["S"], request["Key"]["call_sid"]["S"])
        return {"Item": deepcopy(self.rows[key])} if key in self.rows else {}

    def update_item(self, **request):
        self.writes.append(deepcopy(request))
        if self.before_update is not None:
            hook, self.before_update = self.before_update, None
            hook()
        key = (request["Key"]["workspace_id"]["S"], request["Key"]["call_sid"]["S"])
        current = self.rows.get(key)
        values = request["ExpressionAttributeValues"]
        if ":expected_revision" in values:
            matches = (current is not None
                       and current.get("status_revision") == values[":expected_revision"]
                       and current.get("call_status") == values[":expected_status"])
        else:
            matches = current is None
        if not matches:
            raise ConditionalFailure("synthetic-private-provider-message")
        updated = {
            "call_status": values[":status"], "status_revision": values[":revision"],
        }
        self.rows[key] = {**(current or {}), **deepcopy(request["Key"]), **deepcopy(updated)}
        return {"Attributes": deepcopy(updated)}


class DynamoDBCallStatusStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock(spec=["get_item", "update_item"])
        self.store = DynamoDBCallStatusStore(client=self.client, table_name="demo-status")

    def load(self):
        return self.store.load("demo-workspace", "demo-call")

    def observe(self, status="ringing", revision=0):
        return self.store.observe("demo-workspace", "demo-call", status,
                                  expected_revision=revision)

    def test_construction_is_offline_and_validates_configuration(self):
        self.client.get_item.assert_not_called()
        self.client.update_item.assert_not_called()
        for name in (None, True, "", "ab", "a" * 256, "name/part", " demo ", "ééé"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                DynamoDBCallStatusStore(client=self.client, table_name=name)
        for client in (None, {}, object(), Mock(get_item=None, update_item=None)):
            with self.subTest(client=type(client).__name__), self.assertRaises(ValueError):
                DynamoDBCallStatusStore(client=client, table_name="demo-status")
        for name in ("abc", "demo_status.v1-archive", "a" * 255):
            DynamoDBCallStatusStore(client=self.client, table_name=name)
        self.client.get_item.assert_not_called()

    def test_load_requests_a_consistent_projected_read_and_preserves_opaque_keys(self):
        self.client.get_item.return_value = {"Item": item(workspace=" a/b ", call="c/😀")}
        self.assertEqual(self.store.load(" a/b ", "c/😀"), CallStatusSnapshot("ringing", 1))
        self.client.get_item.assert_called_once_with(
            TableName="demo-status",
            Key={"workspace_id": {"S": " a/b "}, "call_sid": {"S": "c/😀"}},
            ConsistentRead=True,
            ProjectionExpression="#workspace, #call, #status, #revision",
            ExpressionAttributeNames={
                "#workspace": "workspace_id", "#call": "call_sid",
                "#status": "call_status", "#revision": "status_revision",
            },
        )

    def test_only_an_absent_item_means_no_observation(self):
        self.client.get_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
        self.assertEqual(self.load(), CallStatusSnapshot())
        for result in (None, [], {"Item": None}, {"Item": {}}, {"Item": item(workspace="other")}):
            with self.subTest(result=result), self.assertRaisesRegex(
                ValueError, "^DynamoDB returned an invalid call status snapshot$",
            ):
                self.client.get_item.return_value = result
                self.load()

    def test_malformed_stored_state_never_becomes_an_empty_session(self):
        malformed = [
            {**item(), "call_status": {"S": "private-unknown"}},
            {**item(), "call_status": {"NULL": True}},
            {**item(), "call_status": {"S": "ringing", "N": "1"}},
            {**item(), "status_revision": {"S": "1"}},
        ]
        for number in (None, True, 1, "", "0", "-1", "1.5", "NaN", "Infinity",
                       " 1 ", "1_0", "١", "1\n",
                       "1e999999999999999999", str(MAX_STATUS_REVISION + 1), "1" * 129):
            malformed.append({**item(), "status_revision": {"N": number}})
        for row in malformed:
            with self.subTest(row=row), self.assertRaises(ValueError) as raised:
                self.client.get_item.return_value = {"Item": row}
                self.observe()
            self.assertNotIn("private-", str(raised.exception))
        self.client.update_item.assert_not_called()

    def test_valid_integral_dynamodb_number_encodings_are_supported(self):
        for number, expected in (("1", 1), ("1.0", 1), ("1E+2", 100),
                                 (str(MAX_STATUS_REVISION), MAX_STATUS_REVISION)):
            with self.subTest(number=number):
                self.client.get_item.return_value = {"Item": item(revision=number)}
                self.assertEqual(self.load(), CallStatusSnapshot("ringing", expected))

    def test_creation_requires_a_wholly_absent_row(self):
        self.client.get_item.return_value = {}
        self.client.update_item.return_value = {"Attributes": {
            "call_status": {"S": "ringing"}, "status_revision": {"N": "1"},
        }}
        self.assertEqual(self.observe(), CallStatusSnapshot("ringing", 1))
        self.client.update_item.assert_called_once_with(
            TableName="demo-status",
            Key={"workspace_id": {"S": "demo-workspace"}, "call_sid": {"S": "demo-call"}},
            UpdateExpression="SET #status = :status, #revision = :revision",
            ConditionExpression="attribute_not_exists(#workspace) AND attribute_not_exists(#call)",
            ExpressionAttributeNames={
                "#status": "call_status", "#revision": "status_revision",
                "#workspace": "workspace_id", "#call": "call_sid",
            },
            ExpressionAttributeValues={":status": {"S": "ringing"}, ":revision": {"N": "1"}},
            ReturnValues="UPDATED_NEW",
        )

    def test_existing_and_noop_writes_compare_status_and_revision(self):
        for incoming, expected, revision in (("completed", "completed", 2), ("queued", "ringing", 1)):
            with self.subTest(incoming=incoming):
                self.client.get_item.return_value = {"Item": item()}
                self.client.update_item.return_value = {"Attributes": {
                    "call_status": {"S": expected}, "status_revision": {"N": str(revision)},
                }}
                self.assertEqual(self.observe(incoming, 1), CallStatusSnapshot(expected, revision))
                request = self.client.update_item.call_args.kwargs
                self.assertEqual(request["ConditionExpression"],
                                 "#revision = :expected_revision AND #status = :expected_status")
                self.assertEqual(request["ExpressionAttributeValues"], {
                    ":status": {"S": expected}, ":revision": {"N": str(revision)},
                    ":expected_status": {"S": "ringing"}, ":expected_revision": {"N": "1"},
                })

    def test_stale_revision_conflicts_before_writing_even_for_a_noop(self):
        self.client.get_item.return_value = {"Item": item("completed", "2")}
        with self.assertRaisesRegex(StatusConflictError, "^call status revision conflict$"):
            self.observe("completed", 1)
        self.client.update_item.assert_not_called()

    def test_only_conditional_check_failures_are_mapped_to_conflicts(self):
        self.client.get_item.return_value = {}
        self.client.update_item.side_effect = ConditionalFailure("synthetic-private-provider-message")
        with self.assertRaisesRegex(StatusConflictError, "^call status revision conflict$") as raised:
            self.observe()
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertIsNone(raised.exception.__cause__)
        self.assertNotIn("private-", str(raised.exception))
        for failure in (RuntimeError("offline failure"), TimeoutError("uncertain write"),
                        Exception("not a conditional exception")):
            with self.subTest(failure=type(failure).__name__):
                self.client.update_item.side_effect = failure
                with self.assertRaises(type(failure)) as raised:
                    self.observe()
                self.assertIs(raised.exception, failure)

    def test_read_failures_are_not_reclassified_or_followed_by_writes(self):
        failure = ConditionalFailure("read failure, not a write conflict")
        self.client.get_item.side_effect = failure
        with self.assertRaises(ConditionalFailure) as raised:
            self.observe()
        self.assertIs(raised.exception, failure)
        self.client.update_item.assert_not_called()

    def test_malformed_or_unexpected_update_results_fail_closed(self):
        self.client.get_item.return_value = {}
        for result in (None, {}, {"Attributes": {}}, {"Attributes": item("completed")},
                       {"Attributes": item(revision="2")}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.client.update_item.return_value = result
                self.observe()

    def test_invalid_inputs_never_reach_the_client(self):
        for identity in (None, "", " ", "x" * 257, "é" * 129, "private-\ud800"):
            for workspace, call in ((identity, "demo-call"), ("demo-workspace", identity)):
                for operation in (
                    lambda: self.store.load(workspace, call),
                    lambda: self.store.observe(workspace, call, "ringing", expected_revision=0),
                ):
                    with self.assertRaises(ValueError):
                        operation()
        for revision in (None, True, -1, 1.0, "1", MAX_STATUS_REVISION + 1):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.observe(revision=revision)
        for status in (None, "private-unknown", [], True):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.observe(status)
        self.client.get_item.assert_not_called()
        self.client.update_item.assert_not_called()

    def test_revision_limit_never_wraps_and_terminal_noops_still_work(self):
        self.client.get_item.return_value = {"Item": item(revision=str(MAX_STATUS_REVISION))}
        with self.assertRaisesRegex(ValueError, "^call status revision limit reached$"):
            self.observe("completed", MAX_STATUS_REVISION)
        self.client.update_item.assert_not_called()
        self.client.get_item.return_value = {"Item": item("completed", str(MAX_STATUS_REVISION))}
        self.client.update_item.return_value = {"Attributes": item("completed", str(MAX_STATUS_REVISION))}
        self.assertEqual(self.observe("ringing", MAX_STATUS_REVISION),
                         CallStatusSnapshot("completed", MAX_STATUS_REVISION))

    def test_all_three_observation_sequences_match_the_inmemory_policy(self):
        statuses = ("queued", "ringing", "in-progress", "completed", "busy",
                    "failed", "no-answer", "canceled")
        for sequence in product(statuses, repeat=3):
            with self.subTest(sequence=sequence):
                client = SyntheticClient()
                store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
                memory = InMemoryCallStatusStore()
                revision = 0
                for status in sequence:
                    saved = store.observe("workspace", "call", status, expected_revision=revision)
                    self.assertEqual(saved, memory.observe("workspace", "call", status,
                                                          expected_revision=revision))
                    self.assertEqual(store.load("workspace", "call"), saved)
                    revision = saved.revision

    def test_workspace_call_keys_remain_separate_and_unrelated_fields_survive(self):
        client = SyntheticClient()
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
        for workspace, call, status in (("a/b", "c", "completed"), ("a", "b/c", "ringing"),
                                        ("other", "c", "busy")):
            store.observe(workspace, call, status, expected_revision=0)
            client.rows[workspace, call]["unrelated"] = {"S": "synthetic-metadata"}
            saved = store.observe(workspace, call, "queued", expected_revision=1)
            self.assertEqual(saved.status, status)
            self.assertEqual(client.rows[workspace, call]["unrelated"], {"S": "synthetic-metadata"})
        self.assertEqual(len(client.rows), 3)

    def test_racing_creation_and_noop_updates_cannot_overwrite_terminal_state(self):
        for initial in (None, item()):
            with self.subTest(initial=initial):
                client = SyntheticClient()
                if initial is not None:
                    client.rows["demo-workspace", "demo-call"] = initial
                client.before_update = lambda: client.rows.update({
                    ("demo-workspace", "demo-call"): item("completed", "2"),
                })
                store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
                with self.assertRaises(StatusConflictError):
                    store.observe("demo-workspace", "demo-call", "queued",
                                  expected_revision=0 if initial is None else 1)
                self.assertEqual(store.load("demo-workspace", "demo-call"),
                                 CallStatusSnapshot("completed", 2))

    def test_authenticated_retry_reloads_after_a_conditional_conflict(self):
        client = SyntheticClient()
        client.before_update = lambda: client.rows.update({
            ("demo-workspace", CALL): item("completed", "1", call=CALL),
        })
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
        validator = Mock(return_value=True)
        saved = retry_call_status_event(
            callback({"CallStatus": "ringing"}, "2.0", False),
            public_url="https://example.invalid/voice/status", validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id="demo-workspace", store=store, max_conflict_retries=1,
        )
        self.assertEqual(saved, CallStatusSnapshot("completed", 1))
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(len(client.writes), 2)

    def test_same_revision_status_changes_and_row_deletion_also_conflict(self):
        for replacement in (item("completed", "1"), None):
            with self.subTest(replacement=replacement):
                client = SyntheticClient()
                client.rows["demo-workspace", "demo-call"] = item()

                def change_row():
                    if replacement is None:
                        client.rows.pop(("demo-workspace", "demo-call"))
                    else:
                        client.rows["demo-workspace", "demo-call"] = replacement

                client.before_update = change_row
                store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
                with self.assertRaises(StatusConflictError):
                    store.observe("demo-workspace", "demo-call", "in-progress",
                                  expected_revision=1)
                self.assertEqual(client.rows.get(("demo-workspace", "demo-call")), replacement)


if __name__ == "__main__":
    unittest.main()
