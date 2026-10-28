"""Synthetic authentication-to-DynamoDB boundary, not SDK signature validation."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import SyntheticClient, item
from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore


URL = "https://example.invalid/voice/status?route=demo%2Fstatus&v=1"
WORKSPACE = "demo-workspace"


class DynamoDBCallbackAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.backend = SyntheticClient()
        self.client = Mock(wraps=self.backend, spec=["get_item", "update_item"])
        self.admit = Mock()
        self.store = DynamoDBCallStatusStore(
            client=self.client, table_name="demo-status", before_request=self.admit,
        )

    def observe(self, event, validator, **options):
        return retry_call_status_event(
            event, public_url=URL, validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id=WORKSPACE, store=self.store, **options,
        )

    def test_signed_extras_cannot_redirect_trusted_keys_or_expected_revision(self):
        fields = {
            "CallStatus": "completed", "workspace_id": "other-workspace",
            "call_sid": "other-call", "expected_revision": "999",
        }
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.setUp()
                    other = item("busy", "7", workspace="other-workspace", call="other-call")
                    self.backend.rows["other-workspace", "other-call"] = deepcopy(other)
                    event = callback(fields, version, encoded)
                    original = deepcopy(event)
                    validator = Mock(return_value=True)
                    self.assertEqual(self.observe(event, validator), CallStatusSnapshot("completed", 1))
                    validator.assert_called_once_with(
                        URL, {"AccountSid": ACCOUNT, "CallSid": CALL, **fields}, SIGNATURE,
                    )
                    self.assertEqual(event, original)
                    self.assertEqual(self.backend.rows["other-workspace", "other-call"], other)
                    self.assertEqual(len(self.backend.rows), 2)
                    for request in self.client.mock_calls:
                        self.assertEqual(request.kwargs["Key"], {
                            "workspace_id": {"S": WORKSPACE}, "call_sid": {"S": CALL},
                        })
                    self.assertEqual(self.admit.call_count, 3)

    def test_rejected_callbacks_never_reach_admission_reads_or_writes(self):
        cases = (
            ({"CallStatus": "ringing"}, False),
            ({"CallStatus": "ringing", "AccountSid": "other-account"}, True),
            ({"CallStatus": "ringing", "CallSid": "other-call"}, True),
            ({"CallStatus": "unknown-private-status"}, True),
            ({"DialCallStatus": "completed"}, True),
        )
        for initial in (None, item("completed", "2", call=CALL)):
            for version in ("1.0", "2.0"):
                for encoded in (False, True):
                    for fields, accepted in cases:
                        with self.subTest(initial=initial, version=version,
                                          encoded=encoded, fields=fields):
                            self.setUp()
                            if initial is not None:
                                self.backend.rows[WORKSPACE, CALL] = deepcopy(initial)
                            before = deepcopy(self.backend.rows)
                            validator, backoff = Mock(return_value=accepted), Mock()
                            with self.assertRaises(ValueError):
                                self.observe(callback(fields, version, encoded), validator,
                                             max_conflict_retries=2, before_retry=backoff)
                            validator.assert_called_once()
                            self.admit.assert_not_called()
                            self.client.get_item.assert_not_called()
                            self.client.update_item.assert_not_called()
                            backoff.assert_not_called()
                            self.assertEqual(self.backend.rows, before)

    def test_validator_cannot_mutate_the_status_that_will_be_persisted(self):
        event = callback({"CallStatus": "ringing"}, "2.0", True)
        original = deepcopy(event)

        def validate(url, fields, signature):
            self.assertEqual((url, signature), (URL, SIGNATURE))
            fields.update(CallStatus="completed", AccountSid="other-account", CallSid="other-call")
            return True

        self.assertEqual(self.observe(event, validate), CallStatusSnapshot("ringing", 1))
        self.assertEqual(event, original)
        self.assertEqual(self.backend.rows[WORKSPACE, CALL], item(call=CALL))
        self.assertEqual(len(self.backend.rows), 1)

    def test_conflict_retry_reauthenticates_changed_input_before_more_storage(self):
        for changed_fields, accepted in (
            ({"CallStatus": "ringing", "AccountSid": "other-account"}, True),
            ({"CallStatus": "ringing", "CallSid": "other-call"}, True),
            ({"CallStatus": "unknown-private-status"}, True),
            ({"CallStatus": "ringing"}, False),
        ):
            for version in ("1.0", "2.0"):
                for encoded in (False, True):
                    with self.subTest(fields=changed_fields, accepted=accepted,
                                      version=version, encoded=encoded):
                        self.setUp()
                        event = callback({"CallStatus": "ringing"}, version, encoded)
                        terminal = item("completed", "1", call=CALL)
                        self.backend.before_update = lambda: self.backend.rows.update({
                            (WORKSPACE, CALL): deepcopy(terminal),
                        })
                        validator = Mock(side_effect=[True, accepted])

                        def change_callback(retry_number):
                            self.assertEqual(retry_number, 1)
                            event.update(callback(changed_fields, version, encoded))

                        backoff = Mock(side_effect=change_callback)
                        with self.assertRaises(ValueError):
                            self.observe(event, validator, max_conflict_retries=2,
                                         before_retry=backoff)
                        self.assertEqual(validator.call_count, 2)
                        backoff.assert_called_once_with(1)
                        # Only the original authenticated attempt reaches the
                        # outer read, observe reload, and conflicting update.
                        self.assertEqual(self.admit.call_count, 3)
                        self.assertEqual(self.client.get_item.call_count, 2)
                        self.client.update_item.assert_called_once()
                        self.assertEqual(self.backend.rows, {(WORKSPACE, CALL): terminal})


if __name__ == "__main__":
    unittest.main()
