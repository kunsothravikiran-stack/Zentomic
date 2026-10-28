"""Offline claim adapter contracts; no SDK or real provider requests."""

import copy
import unittest
from unittest.mock import Mock

from tests.test_callback_claim_store import ACCOUNT, CALL, URL, callback
from zentomic.callback_claim_store import CallbackReplayError, validate_and_claim_call_event
from zentomic.dynamodb_callback_claim_store import DynamoDBCallbackClaimStore


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class ClaimClient:
    """Model conditional insertion only, not the DynamoDB service."""

    def __init__(self):
        self.requests = []
        self.rows = {}

    def put_item(self, **request):
        self.requests.append(copy.deepcopy(request))
        item = request["Item"]
        key = (item["workspace_id"]["S"], item["callback_key"]["S"])
        if key in self.rows:
            raise ClientError("ConditionalCheckFailedException")
        self.rows[key] = copy.deepcopy(item)
        return {}


class DynamoDBCallbackClaimTests(unittest.TestCase):
    def store(self, client, **kwargs):
        return DynamoDBCallbackClaimStore(client=client, table_name="demo-claims", **kwargs)

    def test_creation_and_replay_use_one_conditional_put_without_reads(self):
        client = ClaimClient()
        store = self.store(client)
        self.assertTrue(store.claim("workspace", "call", "step"))
        self.assertFalse(store.claim("workspace", "call", "step"))
        self.assertEqual(client.requests, [{
            "TableName": "demo-claims",
            "Item": {"workspace_id": {"S": "workspace"},
                     "callback_key": {"S": "4:callstep"}},
            "ConditionExpression": "attribute_not_exists(#workspace) AND attribute_not_exists(#callback)",
            "ExpressionAttributeNames": {"#workspace": "workspace_id", "#callback": "callback_key"},
            "ReturnValues": "NONE", "ReturnValuesOnConditionCheckFailure": "NONE",
        }] * 2)
        self.assertEqual(len(client.rows), 1)

    def test_exact_workspace_call_step_boundaries_do_not_alias(self):
        client = ClaimClient()
        store = self.store(client)
        keys = (
            ("a", "b:c", "d"), ("a", "b", "c:d"),
            ("a", "bc", "d"), ("a", "b", "cd"),
            ("other", "bc", "d"), ("a", " bc", "d"),
            ("a", "é", "step"), ("a", "e\u0301", "step"),
            ("😀" * 64, "😀" * 64, "😀" * 64),
        )
        for key in keys:
            self.assertTrue(store.claim(*key))
            self.assertFalse(store.claim(*key))
        self.assertEqual(len(client.rows), len(keys))
        self.assertIn(("a", "2:éstep"), client.rows)
        self.assertLessEqual(max(len(key[1].encode()) for key in client.rows), 516)

    def test_invalid_identifiers_precede_admission_and_storage(self):
        client, admission = Mock(), Mock()
        store = self.store(client, before_request=admission)
        for index in range(3):
            for value in (None, "", " ", [], "x" * 257, "😀" * 65, "\ud800"):
                key = ["workspace", "call", "step"]
                key[index] = value
                with self.subTest(index=index, value=repr(value)), self.assertRaises(ValueError):
                    store.claim(*key)
        admission.assert_not_called()
        client.put_item.assert_not_called()

    def test_configuration_validation_does_not_invoke_dependencies(self):
        client, admission = Mock(), Mock()
        self.store(client, before_request=admission)
        for overrides in ({"table_name": ""}, {"table_name": "ab"},
                          {"table_name": "x" * 256}, {"table_name": "arn:aws:demo"},
                          {"client": object()}, {"before_request": False}):
            config = dict(client=client, table_name="demo-claims", before_request=admission)
            config.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                DynamoDBCallbackClaimStore(**config)
        admission.assert_not_called()
        client.put_item.assert_not_called()

    def test_admission_is_fresh_and_outside_conditional_failure_translation(self):
        client = ClaimClient()
        denied = ClientError("ConditionalCheckFailedException")
        admission = Mock(side_effect=[None, denied])
        store = self.store(client, before_request=admission)
        self.assertTrue(store.claim("workspace", "call", "step"))
        with self.assertRaises(ClientError) as caught:
            store.claim("workspace", "call", "step")
        self.assertIs(caught.exception, denied)
        self.assertEqual(admission.call_count, 2)
        self.assertEqual(len(client.requests), 1)

    def test_nonconditional_errors_propagate_without_retry(self):
        for error in (TimeoutError(), ClientError("AccessDeniedException"),
                      ClientError("ProvisionedThroughputExceededException"), RuntimeError()):
            client = Mock()
            client.put_item.side_effect = error
            with self.subTest(error=type(error)), self.assertRaises(type(error)) as caught:
                self.store(client).claim("workspace", "call", "step")
            self.assertIs(caught.exception, error)
            client.put_item.assert_called_once()

    def test_admission_return_value_cannot_override_the_conditional_result(self):
        client = ClaimClient()
        store = self.store(client, before_request=Mock(return_value=False))
        self.assertTrue(store.claim("workspace", "call", "step"))
        self.assertFalse(store.claim("workspace", "call", "step"))

    def test_malformed_results_never_grant_admission(self):
        for result in (None, [], True, "synthetic-private-marker"):
            client = Mock()
            client.put_item.return_value = result
            with self.subTest(result=result), self.assertRaises(ValueError) as caught:
                self.store(client).claim("workspace", "call", "step")
            self.assertEqual(str(caught.exception),
                             "DynamoDB returned an invalid callback claim response")
            client.put_item.assert_called_once()

    def test_saved_claim_survives_uncertain_write_acknowledgment(self):
        for failure in (TimeoutError(), None, []):
            client = ClaimClient()
            put = client.put_item

            def uncertain(**request):
                put(**request)
                if isinstance(failure, Exception):
                    raise failure
                return failure

            client.put_item = uncertain
            store = self.store(client)
            with self.subTest(failure=failure), self.assertRaises((TimeoutError, ValueError)):
                store.claim("workspace", "call", "step")
            client.put_item = put
            self.assertFalse(store.claim("workspace", "call", "step"))
            self.assertEqual(len(client.rows), 1)

    def test_authenticated_composition_rejects_before_admission_then_denies_replay(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                client, admission = ClaimClient(), Mock()
                config = dict(
                    public_url=URL, validator=Mock(return_value=False),
                    expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                    workspace_id="trusted-workspace", step_id="trusted-step",
                    claimer=self.store(client, before_request=admission),
                )
                event = callback({"workspace_id": "untrusted", "step_id": "untrusted"},
                                 version=version, encoded=encoded)
                with self.assertRaises(ValueError):
                    validate_and_claim_call_event(event, **config)
                config["validator"] = Mock(return_value=True)
                for fields in ({"AccountSid": "wrong-account"}, {"CallSid": "wrong-call"}):
                    with self.assertRaises(ValueError):
                        validate_and_claim_call_event(
                            callback(fields, version=version, encoded=encoded), **config,
                        )
                admission.assert_not_called()
                self.assertEqual(client.requests, [])
                validate_and_claim_call_event(event, **config)
                with self.assertRaises(CallbackReplayError):
                    validate_and_claim_call_event(event, **config)
                self.assertEqual(set(client.rows), {
                    ("trusted-workspace", f"{len(CALL.encode())}:{CALL}trusted-step"),
                })


if __name__ == "__main__":
    unittest.main()
