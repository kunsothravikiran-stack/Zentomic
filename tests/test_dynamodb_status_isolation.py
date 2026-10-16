"""Workspace/key isolation at the injected DynamoDB boundary, entirely offline."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import SyntheticClient, item
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore


class DynamoDBStatusIsolationTests(unittest.TestCase):
    def test_opaque_keys_never_alias_or_share_terminal_state(self):
        client = SyntheticClient()
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
        # Include delimiter collisions, whitespace and canonically equivalent
        # Unicode. These are distinct opaque identities, not normalized labels.
        keys = (
            ("workspace-a", "call"), ("workspace-b", "call"),
            ("workspace-a", "other-call"), ("a/b", "c"), ("a", "b/c"),
            (" workspace-a ", "call"), ("workspace-a", " call "),
            ("caf\u00e9", "call"), ("cafe\u0301", "call"),
            ("workspace-a", "caf\u00e9"), ("workspace-a", "cafe\u0301"),
        )
        for workspace, call in keys:
            self.assertEqual(store.load(workspace, call), CallStatusSnapshot())
            self.assertEqual(store.observe(workspace, call, "ringing",
                                           expected_revision=0),
                             CallStatusSnapshot("ringing", 1))
        self.assertEqual(set(client.rows), set(keys))

        # End one key at a time. A write may change only that key, even when
        # another row has the same call ID or an easily confused workspace ID.
        for index, (workspace, call) in enumerate(keys):
            before = deepcopy(client.rows)
            ended = store.observe(workspace, call, "completed", expected_revision=1)
            self.assertEqual(ended, CallStatusSnapshot("completed", 2))
            self.assertEqual({key: row for key, row in client.rows.items()
                              if key != (workspace, call)},
                             {key: row for key, row in before.items()
                              if key != (workspace, call)})
            for other_index, key in enumerate(keys):
                expected = (CallStatusSnapshot("completed", 2)
                            if other_index <= index else CallStatusSnapshot("ringing", 1))
                self.assertEqual(store.load(*key), expected)
            self.assertEqual(store.observe(workspace, call, "queued",
                                           expected_revision=2), ended)

    def test_utf8_byte_boundaries_survive_low_level_requests_exactly(self):
        for identity in ("a" * 256, "\u00e9" * 128, "\U0001f600" * 64):
            for key in ((identity, "call"), ("workspace", identity)):
                with self.subTest(key=key):
                    client = SyntheticClient()
                    store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
                    saved = store.observe(*key, "completed", expected_revision=0)
                    self.assertEqual(store.load(*key), saved)
                    self.assertEqual(set(client.rows), {key})
                    self.assertEqual(client.writes[0]["Key"], {
                        "workspace_id": {"S": key[0]}, "call_sid": {"S": key[1]},
                    })

    def test_invalid_keys_never_reach_admission_or_the_client(self):
        client = Mock(spec=["get_item", "update_item"])
        admission = Mock()
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status",
                                        before_request=admission)
        for identity in (None, True, 1, [], {}, "", " \t\n", "a" * 257,
                         "\u00e9" * 128 + "a", "\U0001f600" * 64 + "a", "private-\ud800"):
            for key in ((identity, "call"), ("workspace", identity)):
                for operation in (
                    lambda: store.load(*key),
                    lambda: store.observe(*key, "ringing", expected_revision=0),
                ):
                    with self.subTest(identity=repr(identity)), self.assertRaisesRegex(
                        ValueError,
                        "^status identifiers must be nonblank UTF-8 strings of at most 256 bytes$",
                    ):
                        operation()
                    admission.assert_not_called()
                    client.get_item.assert_not_called()
                    client.update_item.assert_not_called()

    def test_wrong_scope_read_fails_closed_before_any_write(self):
        for row in (
            item(workspace="other-workspace"), item(call="other-call"),
            item(workspace=" demo-workspace"), item(call="demo-call "),
        ):
            client = Mock(spec=["get_item", "update_item"])
            client.get_item.return_value = {"Item": row}
            admission = Mock()
            store = DynamoDBCallStatusStore(client=client, table_name="demo-status",
                                            before_request=admission)
            for operation in (
                lambda: store.load("demo-workspace", "demo-call"),
                lambda: store.observe("demo-workspace", "demo-call", "completed",
                                      expected_revision=1),
            ):
                with self.subTest(row=row), self.assertRaisesRegex(
                    ValueError, "^DynamoDB returned an invalid call status snapshot$",
                ):
                    operation()
            self.assertEqual(admission.call_count, 2)  # Reads only.
            self.assertEqual(client.get_item.call_count, 2)
            client.update_item.assert_not_called()


if __name__ == "__main__":
    unittest.main()
