"""Fixed request-envelope admission with synthetic clocks and clients only."""

import unittest
from unittest.mock import Mock

from tests.test_dynamodb_call_status_store import item
from tests.test_dynamodb_callback_claim_store import ClaimClient
from zentomic.call_status_store import CallStatusSnapshot
from zentomic.dynamodb_call_status_store import DynamoDBCallStatusStore
from zentomic.dynamodb_callback_claim_store import DynamoDBCallbackClaimStore
from zentomic.lambda_budget import (
    RequestBudgetExhaustedError, make_lambda_request_admission_hook,
)


class LambdaRequestAdmissionTests(unittest.TestCase):
    def hook(self, runtime, clock, **overrides):
        context = Mock(spec=["get_remaining_time_in_millis"])
        context.get_remaining_time_in_millis.side_effect = runtime
        config = dict(
            context=context, now_ms=clock, deadline_ms=10_000,
            required_request_ms=1_000, reserve_ms=500,
        )
        config.update(overrides)
        return make_lambda_request_admission_hook(**config), context

    def test_construction_validates_without_sampling_either_clock(self):
        clock = Mock()
        hook, context = self.hook((), clock)
        self.assertTrue(callable(hook))
        clock.assert_not_called()
        context.get_remaining_time_in_millis.assert_not_called()
        for overrides in (
            {"context": None}, {"now_ms": None}, {"deadline_ms": -1},
            {"deadline_ms": True}, {"required_request_ms": 0},
            {"required_request_ms": -1}, {"required_request_ms": True},
            {"required_request_ms": 1.5}, {"reserve_ms": -1},
            {"reserve_ms": False},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.hook((), clock, **overrides)
        clock.assert_not_called()

    def test_exact_envelope_plus_reserve_must_fit_both_budgets(self):
        for runtime, now, allowed in (
            (1_500, 8_500, True), (1_499, 8_500, False),
            (1_500, 8_501, False), (30_000, 10_000, False),
            (0, 0, False), (30_000, 0, True),
        ):
            with self.subTest(runtime=runtime, now=now):
                hook, context = self.hook((runtime,), lambda: now)
                if allowed:
                    self.assertIsNone(hook())
                else:
                    with self.assertRaises(RequestBudgetExhaustedError) as caught:
                        hook()
                    self.assertIsInstance(caught.exception, TimeoutError)
                    self.assertEqual(str(caught.exception),
                                     "request admission budget exhausted")
                context.get_remaining_time_in_millis.assert_called_once_with()

    def test_each_admission_refreshes_time_instead_of_reusing_a_pass(self):
        clock = Mock(side_effect=(7_000, 8_500, 8_501))
        hook, context = self.hook((3_000, 2_000, 2_000), clock)
        self.assertIsNone(hook())
        self.assertIsNone(hook())
        with self.assertRaises(RequestBudgetExhaustedError):
            hook()
        self.assertEqual(clock.call_count, 3)
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 3)

    def test_clock_resets_fail_even_when_the_other_budget_hides_them(self):
        for runtime, times, message in (
            ((30_000, 29_000), (7_000, 6_999), "clock must not move backwards"),
            ((2_000, 2_001), (7_000, 8_000), "invocation time must not increase"),
        ):
            with self.subTest(message=message):
                hook, _ = self.hook(runtime, Mock(side_effect=times))
                self.assertIsNone(hook())
                with self.assertRaisesRegex(ValueError, message):
                    hook()

    def test_invalid_or_unavailable_clocks_never_allow_a_request(self):
        for invalid in (None, True, -1, 1.5, "1000"):
            for runtime_invalid in (True, False):
                with self.subTest(invalid=invalid, runtime_invalid=runtime_invalid):
                    hook, _ = self.hook(
                        (invalid if runtime_invalid else 3_000,),
                        lambda: 7_000 if runtime_invalid else invalid,
                    )
                    with self.assertRaises(ValueError):
                        hook()
        for runtime_failure in (True, False):
            marker = "synthetic-private-clock"
            hook, _ = self.hook(
                (RuntimeError(marker) if runtime_failure else 3_000,),
                Mock(side_effect=RuntimeError(marker)) if not runtime_failure
                else lambda: 7_000,
            )
            with self.assertRaises(ValueError) as caught:
                hook()
            self.assertNotIn(marker, str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_fresh_invocation_does_not_reset_the_persisted_call_deadline(self):
        first, _ = self.hook((2_000,), lambda: 8_000)
        self.assertIsNone(first())
        second, _ = self.hook((30_000,), lambda: 8_501)
        with self.assertRaises(RequestBudgetExhaustedError):
            second()

    def test_status_read_can_succeed_then_noop_write_is_denied(self):
        for runtime, times in (
            ((2_000, 1_499), (7_000, 7_000)),
            ((2_000, 2_000), (7_000, 8_501)),
        ):
            with self.subTest(runtime=runtime, times=times):
                hook, _ = self.hook(runtime, Mock(side_effect=times))
                client = Mock(spec=["get_item", "update_item"])
                client.get_item.return_value = {"Item": item("completed", "2")}
                store = DynamoDBCallStatusStore(
                    client=client, table_name="demo-status", before_request=hook,
                )
                with self.assertRaises(RequestBudgetExhaustedError):
                    store.observe("demo-workspace", "demo-call", "ringing",
                                  expected_revision=2)
                client.get_item.assert_called_once()
                client.update_item.assert_not_called()

    def test_admitted_lifecycle_request_preserves_existing_snapshot_contract(self):
        hook, context = self.hook((2_000, 1_500), lambda: 8_500)
        client = Mock(spec=["get_item", "update_item"])
        client.get_item.return_value = {}
        client.update_item.return_value = {"Attributes": item("ringing", "1")}
        store = DynamoDBCallStatusStore(
            client=client, table_name="demo-status", before_request=hook,
        )
        self.assertEqual(store.observe("demo-workspace", "demo-call", "ringing",
                                       expected_revision=0), CallStatusSnapshot("ringing", 1))
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 2)
        client.get_item.assert_called_once()
        client.update_item.assert_called_once()

    def test_denied_claim_never_sends_a_put_or_consumes_the_step(self):
        hook, _ = self.hook((1_499,), lambda: 0)
        client = ClaimClient()
        store = DynamoDBCallbackClaimStore(
            client=client, table_name="demo-claims", before_request=hook,
        )
        with self.assertRaises(RequestBudgetExhaustedError):
            store.claim("workspace", "call", "step")
        self.assertEqual(client.requests, [])
        self.assertEqual(client.rows, {})
        fresh_hook, _ = self.hook((2_000, 1_500, 1_499), lambda: 0)
        fresh_store = DynamoDBCallbackClaimStore(
            client=client, table_name="demo-claims", before_request=fresh_hook,
        )
        self.assertTrue(fresh_store.claim("workspace", "call", "step"))
        self.assertFalse(fresh_store.claim("workspace", "call", "step"))
        with self.assertRaises(RequestBudgetExhaustedError):
            fresh_store.claim("workspace", "call", "step")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(client.rows), 1)


if __name__ == "__main__":
    unittest.main()
