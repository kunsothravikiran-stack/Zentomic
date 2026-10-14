"""Lambda context timeout admission with local synthetic runtime readings."""

import unittest
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import (
    StatusRetryBudgetExhaustedError, retry_call_status_event,
)
from zentomic.call_status_store import (
    CallStatusSnapshot, InMemoryCallStatusStore, StatusConflictError,
)
from zentomic.lambda_budget import (
    make_lambda_status_conflict_retry_hook, resolve_lambda_operation_timeout,
)


class LambdaOperationTimeoutTests(unittest.TestCase):
    def resolve(self, context, **overrides):
        config = dict(
            context=context, deadline_ms=10_000, now_ms=7_000,
            maximum_ms=5_000, minimum_ms=1_000, reserve_ms=500,
        )
        config.update(overrides)
        return resolve_lambda_operation_timeout(**config)

    def context(self, remaining_ms):
        context = Mock(spec=["get_remaining_time_in_millis"])
        context.get_remaining_time_in_millis.return_value = remaining_ms
        return context

    def test_caps_both_deadlines_before_reserve_maximum_and_rounding(self):
        for remaining_ms, overrides, expected in (
            (2_000, {}, 1_500),
            (9_000, {}, 2_500),
            (9_000, {"maximum_ms": 1_200}, 1_200),
            (2_000, {"granularity_ms": 1_000}, 1_000),
            (9_000, {"deadline_ms": 7_000}, None),
        ):
            with self.subTest(remaining_ms=remaining_ms, overrides=overrides):
                context = self.context(remaining_ms)
                self.assertEqual(self.resolve(context, **overrides), expected)
                context.get_remaining_time_in_millis.assert_called_once_with()

    def test_invocation_exhaustion_and_exact_minimum_boundary(self):
        for remaining_ms in (0, 499, 500, 1_499, 1_500):
            with self.subTest(remaining_ms=remaining_ms):
                expected = 1_000 if remaining_ms == 1_500 else None
                self.assertEqual(
                    self.resolve(self.context(remaining_ms)), expected,
                )

    def test_repeated_decisions_sample_runtime_again_without_resetting_call(self):
        context = self.context(0)
        context.get_remaining_time_in_millis.side_effect = (2_500, 1_500, 500)

        self.assertEqual(
            [self.resolve(context) for _ in range(3)], [2_000, 1_000, None],
        )
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 3)
        # A new invocation's larger budget must still respect the old deadline.
        self.assertIsNone(self.resolve(
            self.context(30_000), now_ms=10_000,
        ))

    def test_missing_or_noncallable_runtime_method_fails_closed(self):
        class NoncallableContext:
            get_remaining_time_in_millis = 30_000

        for context in (None, object(), {}, NoncallableContext()):
            with self.subTest(context_type=type(context).__name__):
                with self.assertRaisesRegex(
                    ValueError, "^context must provide a trusted callable",
                ):
                    self.resolve(context)

    def test_invalid_runtime_readings_never_disable_the_invocation_bound(self):
        for value in (None, True, False, -1, 1.5, "1000", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "^Lambda invocation time must be a nonnegative integer$",
            ):
                self.resolve(self.context(value))

    def test_unavailable_context_or_method_has_a_safe_diagnostic(self):
        class UnreadableContext:
            @property
            def get_remaining_time_in_millis(self):
                raise RuntimeError("synthetic-private-context")

        failing_method = self.context(0)
        failing_method.get_remaining_time_in_millis.side_effect = RuntimeError(
            "synthetic-private-context"
        )
        for context in (UnreadableContext(), failing_method):
            with self.subTest(context_type=type(context).__name__):
                with self.assertRaisesRegex(
                    ValueError, "^Lambda invocation time is unavailable$",
                ) as raised:
                    self.resolve(context)
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertNotIn("synthetic-private-context", str(raised.exception))

    def test_invalid_timeout_policy_is_rejected_even_when_invocation_is_expired(self):
        for overrides in (
            {"deadline_ms": -1}, {"now_ms": True}, {"maximum_ms": 0},
            {"minimum_ms": 0}, {"minimum_ms": 6_000},
            {"reserve_ms": -1}, {"granularity_ms": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(self.context(0), **overrides)


class LambdaStatusRetryTests(unittest.TestCase):
    def context(self, *readings):
        context = Mock(spec=["get_remaining_time_in_millis"])
        context.get_remaining_time_in_millis.side_effect = readings
        return context

    def hook(self, context, **overrides):
        config = dict(
            context=context, wait_ms=Mock(), randbelow=lambda upper: upper - 1,
            reserve_ms=500, minimum_retry_attempt_ms=100,
        )
        config.update(overrides)
        return make_lambda_status_conflict_retry_hook(**config)

    def test_construction_checks_policy_without_sampling_runtime_or_jitter(self):
        context = self.context()
        sampler = Mock()
        for overrides in (
            {"wait_ms": None}, {"randbelow": None}, {"reserve_ms": -1},
            {"minimum_retry_attempt_ms": 0}, {"base_delay_ms": 0},
            {"max_delay_ms": 10_001},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.hook(context, **{"randbelow": sampler, **overrides})
        self.hook(context, randbelow=sampler)
        context.get_remaining_time_in_millis.assert_not_called()
        sampler.assert_not_called()
        with self.assertRaisesRegex(ValueError, "trusted callable"):
            self.hook(None)

    def test_each_retry_samples_before_and_after_wait_with_shared_history(self):
        context = self.context(1_000, 975, 900, 850)
        waiter = Mock()
        hook = self.hook(context, wait_ms=waiter)
        hook(1)
        hook(2)
        self.assertEqual([call.args[0] for call in waiter.call_args_list], [25, 50])
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 4)

    def test_exhaustion_stops_before_wait_or_after_runtime_overhead(self):
        for readings, waits in (((599,), 0), ((625, 599), 1)):
            with self.subTest(readings=readings):
                waiter = Mock()
                with self.assertRaises(StatusRetryBudgetExhaustedError):
                    self.hook(self.context(*readings), wait_ms=waiter)(1)
                self.assertEqual(waiter.call_count, waits)

    def test_runtime_failure_after_wait_keeps_context_details_private(self):
        context = self.context(1_000, RuntimeError("synthetic-private-context"))
        with self.assertRaisesRegex(
            ValueError, "^Lambda invocation time is unavailable$",
        ) as raised:
            self.hook(context)(1)
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)

    def test_invalid_or_restored_runtime_readings_fail_closed(self):
        for readings in ((True,), (-1,), (1_000, None), (1_000, 1_001),
                         (1_000, 1_000), (1_000, 975, 976)):
            with self.subTest(readings=readings), self.assertRaises(ValueError):
                hook = self.hook(self.context(*readings))
                hook(1)
                if len(readings) == 3:
                    hook(2)

    def observe(self, store, validator, hook):
        return retry_call_status_event(
            callback({"CallStatus": "completed"}, "2.0", False),
            public_url="https://example.invalid/voice/status",
            validator=validator, expected_account_sid=ACCOUNT,
            expected_call_sid=CALL, workspace_id="synthetic-workspace",
            store=store, before_retry=hook,
        )

    def test_successful_status_observation_does_not_read_runtime(self):
        context = self.context()
        result = self.observe(
            InMemoryCallStatusStore(), Mock(return_value=True), self.hook(context),
        )
        self.assertEqual(result, CallStatusSnapshot("completed", 1))
        context.get_remaining_time_in_millis.assert_not_called()

    def test_exhausted_lambda_prevents_reauthentication_and_second_write(self):
        store = Mock(spec=["load", "observe"])
        store.load.return_value = CallStatusSnapshot(None, 0)
        store.observe.side_effect = StatusConflictError("synthetic conflict")
        validator = Mock(return_value=True)
        with self.assertRaises(StatusRetryBudgetExhaustedError):
            self.observe(store, validator, self.hook(self.context(599)))
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(store.load.call_count, 1)
        self.assertEqual(store.observe.call_count, 1)

    def test_budgeted_conflict_retries_the_complete_authenticated_observation(self):
        store = InMemoryCallStatusStore()
        observe = store.observe

        def conflict_once(*args, **kwargs):
            if store.observe.call_count == 1:
                raise StatusConflictError("synthetic conflict")
            return observe(*args, **kwargs)

        store.observe = Mock(side_effect=conflict_once)
        validator = Mock(return_value=True)
        context = self.context(1_000, 975)
        waiter = Mock()
        result = self.observe(store, validator, self.hook(context, wait_ms=waiter))
        self.assertEqual(result, CallStatusSnapshot("completed", 1))
        self.assertEqual(validator.call_count, 2)
        self.assertEqual(store.observe.call_count, 2)
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 2)
        waiter.assert_called_once_with(25)


if __name__ == "__main__":
    unittest.main()
