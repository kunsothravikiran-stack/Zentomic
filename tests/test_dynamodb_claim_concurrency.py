"""Overlapping authenticated deliveries against a synthetic atomic claim table.

The lock models the service's conditional put, not real DynamoDB atomicity.
Barriers force every delivery to reach storage before any claim can win.
"""

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from threading import Barrier, Lock
from unittest.mock import patch

from tests.test_callback_claim_store import ACCOUNT, CALL, URL, callback
from tests.test_dynamodb_callback_claim_store import ClientError
from tests.test_dynamodb_claimed_voice_flow import CASES
from zentomic.callback_claim_store import CallbackReplayError
from zentomic.dynamodb_callback_claim_store import DynamoDBCallbackClaimStore


class ConcurrentClaimClient:
    """Synchronize contenders, then model one atomic conditional insertion."""

    def __init__(self, *, parties=2, lose_acknowledgment=False):
        self.barrier = Barrier(parties, timeout=5)
        self.lock = Lock()
        self.rows = {}
        self.requests = []
        self.lose_acknowledgment = lose_acknowledgment

    def put_item(self, **request):
        self.barrier.wait()
        with self.lock:
            self.requests.append(copy.deepcopy(request))
            item = request["Item"]
            key = (item["workspace_id"]["S"], item["callback_key"]["S"])
            if key in self.rows:
                raise ClientError("ConditionalCheckFailedException")
            self.rows[key] = copy.deepcopy(item)
        if self.lose_acknowledgment:
            raise TimeoutError("synthetic lost acknowledgment")
        return {}


class DynamoDBClaimConcurrencyTests(unittest.TestCase):
    def invoke(self, case, event, client, *, workspace="workspace", step="step"):
        # Each delivery owns a fresh store; only the injected table is shared.
        return case[0](
            event, **case[2], public_url=URL, validator=lambda *_: True,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id=workspace, step_id=step,
            claimer=DynamoDBCallbackClaimStore(client=client, table_name="demo-claims"),
            fallback_target="demo-reception",
        )

    def overlap(self, *operations):
        def capture(operation):
            try:
                return operation()
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=len(operations)) as pool:
            futures = [pool.submit(capture, operation) for operation in operations]
            return [future.result(timeout=10) for future in futures]

    def variants(self):
        return product(CASES, ("1.0", "2.0"), (False, True))

    def test_changed_concurrent_deliveries_enter_each_policy_only_once(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ConcurrentClaimClient()
                events = [callback({**case[3], "delivery": str(index)},
                                   version=version, encoded=encoded) for index in range(2)]
                with patch(case[1], return_value=case[4]) as policy:
                    results = self.overlap(*[
                        lambda event=event: self.invoke(case, event, client)
                        for event in events
                    ])
                    policy.assert_called_once()
                self.assertEqual(results.count(case[4]), 1)
                self.assertEqual(sum(isinstance(r, CallbackReplayError) for r in results), 1)
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(set(client.rows), {
                    ("workspace", f"{len(CALL.encode())}:{CALL}step"),
                })

    def test_lost_winner_acknowledgment_does_not_admit_either_delivery(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ConcurrentClaimClient(lose_acknowledgment=True)
                event = callback(case[3], version=version, encoded=encoded)
                with patch(case[1]) as policy:
                    results = self.overlap(*[
                        lambda: self.invoke(case, event, client) for _ in range(2)
                    ])
                    policy.assert_not_called()
                self.assertEqual(sum(isinstance(r, TimeoutError) for r in results), 1)
                self.assertEqual(sum(isinstance(r, CallbackReplayError) for r in results), 1)
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(len(client.rows), 1)

    def test_policy_failure_after_winning_cannot_admit_the_concurrent_loser(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ConcurrentClaimClient()
                event = callback(case[3], version=version, encoded=encoded)
                failure = RuntimeError("synthetic policy failure")
                with patch(case[1], side_effect=failure) as policy:
                    results = self.overlap(*[
                        lambda: self.invoke(case, event, client) for _ in range(2)
                    ])
                    policy.assert_called_once()
                self.assertEqual(sum(r is failure for r in results), 1)
                self.assertEqual(sum(isinstance(r, CallbackReplayError) for r in results), 1)
                self.assertEqual(len(client.rows), 1)

    def test_concurrent_workspaces_and_steps_remain_independent(self):
        keys = (("workspace", "step"), ("other", "step"),
                ("workspace", "other-step"), ("workspace ", "step"))
        for case in CASES:
            with self.subTest(helper=case[0].__name__):
                client = ConcurrentClaimClient(parties=len(keys))
                event = callback(case[3])
                with patch(case[1], return_value=case[4]) as policy:
                    results = self.overlap(*[
                        lambda workspace=workspace, step=step: self.invoke(
                            case, event, client, workspace=workspace, step=step,
                        ) for workspace, step in keys
                    ])
                    self.assertEqual(policy.call_count, len(keys))
                self.assertEqual(results, [case[4]] * len(keys))
                self.assertEqual(len(client.requests), len(keys))
                self.assertEqual(set(client.rows), {
                    (workspace, f"{len(CALL.encode())}:{CALL}{step}")
                    for workspace, step in keys
                })


if __name__ == "__main__":
    unittest.main()
