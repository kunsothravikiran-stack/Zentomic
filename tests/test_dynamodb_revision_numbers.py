"""Exact numeric revision comparisons using synthetic storage only."""

import unittest
from copy import deepcopy
from decimal import localcontext

from tests.test_dynamodb_call_status_store import SyntheticClient, item
from zentomic.call_status_store import CallStatusSnapshot, StatusConflictError
from zentomic.dynamodb_call_status_store import (
    MAX_STATUS_REVISION, DynamoDBCallStatusStore,
)


class DynamoDBRevisionNumberTests(unittest.TestCase):
    def store(self, number, status="ringing"):
        client = SyntheticClient()
        client.rows["demo-workspace", "demo-call"] = item(status, number)
        store = DynamoDBCallStatusStore(client=client, table_name="demo-status")
        return client, store

    def observe(self, store, status, revision):
        return store.observe("demo-workspace", "demo-call", status,
                             expected_revision=revision)

    def test_equivalent_wire_numbers_allow_advances_and_noops(self):
        for number in ("1", "1.0", "1E+0", "10e-1", "+1", "01.000"):
            for incoming, status, revision in (
                ("completed", "completed", 2), ("queued", "ringing", 1),
            ):
                with self.subTest(number=number, incoming=incoming):
                    client, store = self.store(number)
                    self.assertEqual(self.observe(store, incoming, 1),
                                     CallStatusSnapshot(status, revision))
                    self.assertEqual(store.load("demo-workspace", "demo-call"),
                                     CallStatusSnapshot(status, revision))
                    self.assertEqual(client.writes[0]["ExpressionAttributeValues"]
                                     [":expected_revision"], {"N": "1"})

    def test_38_digit_revisions_do_not_round_through_decimal_context_or_float(self):
        # Adjacent large revisions must remain distinct even when ambient
        # Decimal precision is too small for ordinary decimal arithmetic.
        revision = MAX_STATUS_REVISION - 1
        number = str(revision) + ".0"
        with localcontext() as context:
            context.prec = 1
            client, store = self.store(number)
            self.assertEqual(self.observe(store, "completed", revision),
                             CallStatusSnapshot("completed", MAX_STATUS_REVISION))
            self.assertEqual(client.writes[0]["ExpressionAttributeValues"]
                             [":revision"], {"N": str(MAX_STATUS_REVISION)})
            self.assertEqual(self.observe(store, "ringing", MAX_STATUS_REVISION),
                             CallStatusSnapshot("completed", MAX_STATUS_REVISION))

    def test_distinct_adjacent_large_revision_is_a_real_write_conflict(self):
        revision = MAX_STATUS_REVISION - 1
        client, store = self.store(str(revision))
        client.before_update = lambda: client.rows.update({
            ("demo-workspace", "demo-call"): item("ringing", str(revision + 1)),
        })
        with self.assertRaises(StatusConflictError):
            self.observe(store, "completed", revision)
        self.assertEqual(store.load("demo-workspace", "demo-call"),
                         CallStatusSnapshot("ringing", revision + 1))
        self.assertEqual(len(client.writes), 1)

    def test_numeric_equivalence_does_not_mask_a_concurrent_status_change(self):
        client, store = self.store("1.00")
        client.before_update = lambda: client.rows.update({
            ("demo-workspace", "demo-call"): item("completed", "10e-1"),
        })
        with self.assertRaises(StatusConflictError):
            self.observe(store, "ringing", 1)
        self.assertEqual(store.load("demo-workspace", "demo-call"),
                         CallStatusSnapshot("completed", 1))

    def test_stale_expected_revision_never_reaches_the_write(self):
        client, store = self.store("1e2")
        original = deepcopy(client.rows)
        with self.assertRaises(StatusConflictError):
            self.observe(store, "completed", 99)
        self.assertEqual(client.rows, original)
        self.assertEqual(client.writes, [])


if __name__ == "__main__":
    unittest.main()
