"""Overlapping lifecycle writes against a synthetic atomic conditional client.

The lock models the service's compare-and-set, not real DynamoDB atomicity.
Only the first pair of updates waits at a bounded barrier; retries run freely.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Lock
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import ConditionalFailure, SyntheticClient, item
from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import CallStatusSnapshot, StatusConflictError
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore


class ConcurrentStatusClient(SyntheticClient):
    """Snapshot reads and conditional writes under one synthetic table lock."""

    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.barrier = Barrier(2, timeout=5)
        self.entered = 0
        self.conflicts = 0
        self.saved = []

    def get_item(self, **request):
        with self.lock:
            return super().get_item(**request)

    def update_item(self, **request):
        with self.lock:
            self.entered += 1
            synchronize = self.entered <= 2
        # Both observations have loaded their original state before either
        # conditional write may run. Do not hold the table lock while waiting.
        if synchronize:
            self.barrier.wait()
        with self.lock:
            try:
                result = super().update_item(**request)
            except ConditionalFailure:
                self.conflicts += 1
                raise
            self.saved.append(deepcopy(self.rows))
            return result


class DynamoDBStatusConcurrencyTests(unittest.TestCase):
    def store(self, client):
        # Distinct adapters share only the injected synthetic table.
        return DynamoDBCallStatusStore(client=client, table_name="demo-status")

    def overlap(self, *operations):
        def capture(operation):
            try:
                return operation()
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=len(operations)) as pool:
            futures = [pool.submit(capture, operation) for operation in operations]
            return [future.result(timeout=10) for future in futures]

    def invoke(self, client, status, validator, retry, *, version, encoded):
        return retry_call_status_event(
            callback({"CallStatus": status}, version, encoded),
            public_url="https://example.invalid/voice/status", validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id="workspace", store=self.store(client),
            max_conflict_retries=1, before_retry=retry,
        )

    def test_simultaneous_creation_has_one_winner_without_blind_overwrite(self):
        client = ConcurrentStatusClient()
        results = self.overlap(*[
            lambda status=status: self.store(client).observe(
                "workspace", CALL, status, expected_revision=0,
            ) for status in ("ringing", "in-progress")
        ])
        successes = [value for value in results if type(value) is CallStatusSnapshot]
        self.assertEqual(len(successes), 1)
        self.assertEqual(sum(isinstance(value, StatusConflictError) for value in results), 1)
        self.assertEqual(successes[0].revision, 1)
        self.assertEqual(self.store(client).load("workspace", CALL), successes[0])
        self.assertEqual((len(client.writes), client.conflicts, len(client.saved)), (2, 1, 1))
        self.assertTrue(all("attribute_not_exists" in request["ConditionExpression"]
                            for request in client.writes))

    def test_contending_callbacks_reauthenticate_and_preserve_first_terminal(self):
        # Either callback can win the initial condition. The invariant does
        # not depend on scheduler order, and the barrier guarantees contention.
        for statuses in (("ringing", "completed"), ("completed", "busy")):
            for version in ("1.0", "2.0"):
                for encoded in (False, True):
                    with self.subTest(statuses=statuses, version=version, encoded=encoded):
                        client = ConcurrentStatusClient()
                        client.rows["workspace", CALL] = {
                            **item("queued", "1", "workspace", CALL),
                            "unrelated": {"S": "synthetic-preserved"},
                        }
                        validators = [Mock(return_value=True), Mock(return_value=True)]
                        retries = [Mock(), Mock()]
                        results = self.overlap(*[
                            lambda index=index: self.invoke(
                                client, statuses[index], validators[index], retries[index],
                                version=version, encoded=encoded,
                            ) for index in range(2)
                        ])
                        self.assertTrue(all(type(value) is CallStatusSnapshot for value in results))
                        self.assertEqual(sorted(v.call_count for v in validators), [1, 2])
                        self.assertEqual(sorted(r.call_count for r in retries), [0, 1])
                        for retry in retries:
                            if retry.call_count:
                                retry.assert_called_once_with(1)
                        self.assertEqual((len(client.writes), client.conflicts), (3, 1))
                        first = client.saved[0]["workspace", CALL]["call_status"]["S"]
                        expected = first if first in ("completed", "busy") else "completed"
                        revision = 3 if first == "ringing" else 2
                        self.assertEqual(self.store(client).load("workspace", CALL),
                                         CallStatusSnapshot(expected, revision))
                        for snapshot in client.saved:
                            self.assertEqual(snapshot["workspace", CALL]["unrelated"],
                                             {"S": "synthetic-preserved"})
                        # The retry must compare against what the winner saved,
                        # not reuse the original queued/revision-one snapshot.
                        values = client.writes[-1]["ExpressionAttributeValues"]
                        self.assertEqual(values[":expected_status"], {"S": first})
                        self.assertEqual(values[":expected_revision"], {"N": "2"})

    def test_concurrent_terminal_noops_need_no_conflict_retry_or_revision_bump(self):
        client = ConcurrentStatusClient()
        client.rows["workspace", CALL] = item("completed", "2", "workspace", CALL)
        validators = [Mock(return_value=True), Mock(return_value=True)]
        retries = [Mock(), Mock()]
        results = self.overlap(*[
            lambda index=index: self.invoke(
                client, ("queued", "busy")[index], validators[index], retries[index],
                version="2.0", encoded=True,
            ) for index in range(2)
        ])
        self.assertEqual(results, [CallStatusSnapshot("completed", 2)] * 2)
        self.assertEqual((len(client.writes), client.conflicts), (2, 0))
        for validator, retry in zip(validators, retries):
            validator.assert_called_once()
            retry.assert_not_called()
        self.assertTrue(all(":expected_status" in request["ExpressionAttributeValues"]
                            for request in client.writes))

    def test_overlapping_distinct_workspace_or_call_keys_do_not_conflict(self):
        for other in (("other-workspace", CALL), ("workspace", "other-call")):
            with self.subTest(other=other):
                client = ConcurrentStatusClient()
                keys = (("workspace", CALL), other)
                results = self.overlap(*[
                    lambda key=key: self.store(client).observe(
                        *key, "completed", expected_revision=0,
                    ) for key in keys
                ])
                self.assertEqual(results, [CallStatusSnapshot("completed", 1)] * 2)
                self.assertEqual(set(client.rows), set(keys))
                self.assertEqual((len(client.writes), client.conflicts), (2, 0))
                for key in keys:
                    self.assertEqual(self.store(client).load(*key), results[0])


if __name__ == "__main__":
    unittest.main()
