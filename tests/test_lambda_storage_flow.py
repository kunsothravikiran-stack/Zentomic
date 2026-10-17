"""Fixed-envelope admission composed with authenticated storage and backoff."""

import unittest
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import SyntheticClient, item
from tests.test_dynamodb_callback_claim_store import ClaimClient
from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import retry_call_status_event
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.callback_claim_store import validate_and_claim_call_event
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore
from zentomic.dynamodb_callback_claim_store import DynamoDBCallbackClaimStore
from zentomic.lambda_budget import (
    RequestBudgetExhaustedError, make_lambda_request_admission_hook,
    make_lambda_status_conflict_retry_hook,
)


WORKSPACE = "demo-workspace"
URL = "https://example.invalid/voice/status"


class LambdaStorageFlowTests(unittest.TestCase):
    def setUp(self):
        self.time = {"runtime": 5_000, "now": 0}
        self.context = Mock(spec=["get_remaining_time_in_millis"])
        self.context.get_remaining_time_in_millis.side_effect = lambda: self.time["runtime"]
        self.now = Mock(side_effect=lambda: self.time["now"])
        self.admit = Mock(wraps=make_lambda_request_admission_hook(
            context=self.context, deadline_ms=10_000, now_ms=self.now,
            required_request_ms=1_000, reserve_ms=500,
        ))
        self.backend = SyntheticClient()
        self.client = Mock(wraps=self.backend, spec=["get_item", "update_item"])
        self.store = DynamoDBCallStatusStore(
            client=self.client, table_name="demo-status", before_request=self.admit,
        )

        def wait(delay):
            self.time["runtime"] -= delay
            self.time["now"] += delay

        self.wait = Mock(side_effect=wait)
        self.sample = Mock(side_effect=lambda upper: upper - 1)
        self.backoff = Mock(wraps=make_lambda_status_conflict_retry_hook(
            context=self.context, deadline_ms=10_000, now_ms=self.now,
            wait_ms=self.wait, randbelow=self.sample, reserve_ms=500,
            minimum_retry_attempt_ms=1_000,
        ))
        # The first conditional create loses a race to a terminal observation.
        self.backend.before_update = lambda: self.backend.rows.update({
            (WORKSPACE, CALL): item("completed", "1", call=CALL),
        })

    def observe(self, event, validator):
        return retry_call_status_event(
            event, public_url=URL, validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id=WORKSPACE, store=self.store,
            before_retry=self.backoff, max_conflict_retries=2,
        )

    def validator(self, *, runtime, now, accepted=True):
        def validate(*args):
            if validate_mock.call_count == 2:
                self.time.update(runtime=runtime, now=now)
                return accepted
            return True

        validate_mock = Mock(side_effect=validate)
        return validate_mock

    def assert_conflict_only(self, validator, *, admissions, readings):
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(self.admit.call_count, admissions)
        self.assertEqual(self.context.get_remaining_time_in_millis.call_count, readings)
        self.assertEqual(self.now.call_count, readings)
        self.assertEqual(self.client.get_item.call_count, 2)
        self.client.update_item.assert_called_once()
        self.backoff.assert_called_once_with(1)
        self.wait.assert_called_once_with(25)
        self.sample.assert_called_once_with(26)
        self.assertEqual(self.backend.rows[WORKSPACE, CALL], item("completed", "1", call=CALL))

    def test_backoff_pass_does_not_admit_a_request_after_reauthentication_spends_time(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for runtime, now in ((1_499, 25), (4_975, 8_501)):
                    with self.subTest(version=version, encoded=encoded,
                                      runtime=runtime, now=now):
                        self.setUp()
                        validator = self.validator(runtime=runtime, now=now)
                        with self.assertRaises(RequestBudgetExhaustedError):
                            self.observe(callback({"CallStatus": "ringing"}, version, encoded),
                                         validator)
                        self.assert_conflict_only(validator, admissions=4, readings=6)

    def test_exact_remaining_envelope_allows_the_authenticated_terminal_noop_retry(self):
        validator = self.validator(runtime=1_500, now=8_500)
        self.assertEqual(
            self.observe(callback({"CallStatus": "ringing"}, "2.0", True), validator),
            CallStatusSnapshot("completed", 1),
        )
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(self.admit.call_count, 6)
        self.assertEqual(self.context.get_remaining_time_in_millis.call_count, 8)
        self.assertEqual(self.now.call_count, 8)
        self.assertEqual(self.client.get_item.call_count, 4)
        self.assertEqual(self.client.update_item.call_count, 2)
        self.backoff.assert_called_once_with(1)
        self.wait.assert_called_once_with(25)
        values = self.backend.writes[-1]["ExpressionAttributeValues"]
        self.assertEqual(values[":expected_status"], {"S": "completed"})
        self.assertEqual(values[":expected_revision"], {"N": "1"})
        self.assertEqual(self.backend.rows[WORKSPACE, CALL], item("completed", "1", call=CALL))

    def test_failed_reauthentication_precedes_readmission_even_when_budget_has_expired(self):
        validator = self.validator(runtime=0, now=10_000, accepted=False)
        with self.assertRaisesRegex(ValueError, "^webhook signature validation failed$"):
            self.observe(callback({"CallStatus": "ringing"}, "1.0", False), validator)
        self.assert_conflict_only(validator, admissions=3, readings=5)

    def test_one_invocation_hook_preserves_clock_history_across_different_stores(self):
        for runtime, now, message in (
            (5_001, 100, "invocation time must not increase"),
            (4_999, 99, "clock must not move backwards"),
        ):
            with self.subTest(runtime=runtime, now=now):
                self.setUp()
                self.time["now"] = 100
                claims = ClaimClient()
                claimer = DynamoDBCallbackClaimStore(
                    client=claims, table_name="demo-claims", before_request=self.admit,
                )
                self.assertTrue(claimer.claim(WORKSPACE, CALL, "step"))
                self.time.update(runtime=runtime, now=now)
                with self.assertRaisesRegex(ValueError, message):
                    self.store.load(WORKSPACE, CALL)
                self.client.get_item.assert_not_called()
                self.assertEqual(len(claims.requests), 1)
                self.assertEqual(len(claims.rows), 1)

    def test_status_read_admission_is_not_a_reservation_for_a_later_authenticated_claim(self):
        self.assertEqual(self.store.load(WORKSPACE, CALL), CallStatusSnapshot())
        self.time["runtime"] = 1_499
        claims = ClaimClient()
        config = dict(
            public_url=URL, validator=Mock(return_value=True),
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id=WORKSPACE, step_id="step",
            claimer=DynamoDBCallbackClaimStore(
                client=claims, table_name="demo-claims", before_request=self.admit,
            ),
        )
        event = callback({}, "2.0", True)
        with self.assertRaises(RequestBudgetExhaustedError):
            validate_and_claim_call_event(event, **config)
        config["validator"].assert_called_once()
        self.assertEqual(claims.requests, [])
        self.assertEqual(claims.rows, {})
        # A fresh invocation can claim the unconsumed step, using the same
        # persisted deadline rather than starting a new call budget.
        fresh_context = Mock(get_remaining_time_in_millis=Mock(return_value=30_000))
        config["claimer"] = DynamoDBCallbackClaimStore(
            client=claims, table_name="demo-claims",
            before_request=make_lambda_request_admission_hook(
                context=fresh_context, deadline_ms=10_000, now_ms=lambda: 8_500,
                required_request_ms=1_000, reserve_ms=500,
            ),
        )
        validate_and_claim_call_event(event, **config)
        self.assertEqual(len(claims.requests), 1)
        self.assertEqual(len(claims.rows), 1)


if __name__ == "__main__":
    unittest.main()
