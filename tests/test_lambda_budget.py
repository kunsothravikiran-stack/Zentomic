"""Lambda context timeout admission with local synthetic runtime readings."""

import unittest
from itertools import product
from unittest.mock import Mock

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.call_status_event import (
    StatusRetryBudgetExhaustedError, retry_call_status_event,
)
from zentomic.call_status_store import (
    CallStatusSnapshot, InMemoryCallStatusStore, StatusConflictError,
)
from zentomic.lambda_budget import (
    make_lambda_operation_timeout_resolver, make_lambda_status_conflict_retry_hook,
    resolve_lambda_operation_timeout,
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


class LambdaOperationResolverTests(unittest.TestCase):
    def resolver(self, runtime, clock, **overrides):
        context = Mock(spec=["get_remaining_time_in_millis"])
        context.get_remaining_time_in_millis.side_effect = runtime
        config = dict(
            context=context, now_ms=clock, deadline_ms=10_000,
            maximum_ms=5_000, minimum_ms=1_000, reserve_ms=500,
        )
        config.update(overrides)
        return make_lambda_operation_timeout_resolver(**config), context

    def test_construction_validates_every_policy_without_sampling_clocks(self):
        clock = Mock()
        for overrides in (
            {"context": None}, {"deadline_ms": -1}, {"deadline_ms": True},
            {"now_ms": None}, {"maximum_ms": 0}, {"minimum_ms": 0},
            {"minimum_ms": 6_000}, {"reserve_ms": -1},
            {"granularity_ms": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolver((), clock, **overrides)
        _, context = self.resolver((), clock)
        clock.assert_not_called()
        context.get_remaining_time_in_millis.assert_not_called()

    def test_refreshes_both_clocks_and_preserves_reserve_and_rounding(self):
        clock = Mock(side_effect=(7_000, 7_500, 8_500))
        resolve, context = self.resolver(
            (9_000, 2_250, 1_499), clock, granularity_ms=1_000,
        )
        self.assertEqual([resolve() for _ in range(3)], [2_000, 1_000, None])
        self.assertEqual(clock.call_count, 3)
        self.assertEqual(context.get_remaining_time_in_millis.call_count, 3)

    def test_call_expiry_and_runtime_exhaustion_deny_first_attempt(self):
        for runtime, now, expected in (
            (30_000, 10_000, None), (0, 7_000, None),
            (1_499, 7_000, None), (1_500, 7_000, 1_000),
            (30_000, 8_500, 1_000), (30_000, 8_501, None),
        ):
            with self.subTest(runtime=runtime, now=now):
                resolve, _ = self.resolver((runtime,), lambda: now)
                self.assertEqual(resolve(), expected)

    def test_independent_history_rejects_resets_hidden_by_other_clock(self):
        for runtime, times, diagnostic in (
            ((30_000, 29_000), (7_000, 6_999), "clock must not move backwards"),
            ((2_000, 2_001), (7_000, 9_000), "invocation time must not increase"),
        ):
            with self.subTest(diagnostic=diagnostic):
                resolve, _ = self.resolver(runtime, Mock(side_effect=times))
                self.assertIsNotNone(resolve())
                with self.assertRaisesRegex(ValueError, diagnostic):
                    resolve()

    def test_new_invocation_resets_history_but_not_persisted_call_deadline(self):
        first, _ = self.resolver((2_000,), lambda: 7_000)
        second, _ = self.resolver((30_000,), lambda: 10_000)
        self.assertEqual(first(), 1_500)
        self.assertIsNone(second())

    def test_invalid_clock_or_runtime_reading_fails_closed(self):
        for invalid in (None, True, False, -1, 1.5, "1000", [], {}):
            for invalid_runtime in (True, False):
                with self.subTest(value=invalid, runtime=invalid_runtime):
                    runtime = (invalid,) if invalid_runtime else (30_000,)
                    clock = (lambda: 7_000) if invalid_runtime else (lambda: invalid)
                    resolve, _ = self.resolver(runtime, clock)
                    with self.assertRaises(ValueError):
                        resolve()

    def test_runtime_and_clock_failures_have_private_safe_diagnostics(self):
        for failing_runtime, diagnostic in (
            (True, "Lambda invocation time is unavailable"),
            (False, "call budget clock is unavailable"),
        ):
            with self.subTest(runtime=failing_runtime):
                failure = RuntimeError("synthetic-private-clock")
                runtime = (failure,) if failing_runtime else (30_000,)
                clock = Mock(side_effect=(7_000,) if failing_runtime else (failure,))
                resolve, _ = self.resolver(runtime, clock)
                with self.assertRaisesRegex(ValueError, "^" + diagnostic + "$") as raised:
                    resolve()
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertNotIn("synthetic-private-clock", str(raised.exception))

    def test_repeated_equal_readings_are_allowed_without_refreshing_deadline(self):
        resolve, _ = self.resolver((2_000, 2_000), lambda: 7_000)
        self.assertEqual([resolve(), resolve()], [1_500, 1_500])

    def test_bound_policy_matches_single_decision_helper(self):
        for runtime, now, maximum, minimum, reserve, granularity in product(
            (0, 1_500, 30_000), (0, 7_000, 10_000), (1_000, 5_000),
            (1, 1_000), (0, 500), (1, 1_000),
        ):
            with self.subTest(
                runtime=runtime, now=now, maximum=maximum, minimum=minimum,
                reserve=reserve, granularity=granularity,
            ):
                policy = dict(
                    maximum_ms=maximum, minimum_ms=minimum,
                    reserve_ms=reserve, granularity_ms=granularity,
                )
                resolve, context = self.resolver((runtime, runtime), lambda: now, **policy)
                self.assertEqual(resolve(), resolve_lambda_operation_timeout(
                    context=context, deadline_ms=10_000, now_ms=now, **policy,
                ))


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

    def test_call_deadline_configuration_is_checked_without_reading_clocks(self):
        context = self.context()
        clock = Mock()
        for overrides in (
            {"deadline_ms": 1_000}, {"now_ms": clock},
            {"deadline_ms": -1, "now_ms": clock},
            {"deadline_ms": True, "now_ms": clock},
            {"deadline_ms": 1_000, "now_ms": 100},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.hook(context, **overrides)
        self.hook(context, deadline_ms=1_000, now_ms=clock)
        context.get_remaining_time_in_millis.assert_not_called()
        clock.assert_not_called()

    def test_expired_call_prevents_wait_even_with_invocation_headroom(self):
        for now in (9_401, 10_000, 10_001):
            with self.subTest(now=now):
                waiter, sampler = Mock(), Mock()
                hook = self.hook(
                    self.context(30_000), deadline_ms=10_000,
                    now_ms=lambda: now, wait_ms=waiter, randbelow=sampler,
                )
                with self.assertRaises(StatusRetryBudgetExhaustedError):
                    hook(1)
                waiter.assert_not_called()
                sampler.assert_not_called()

    def test_call_time_caps_jitter_before_reserving_cleanup_and_next_attempt(self):
        context = self.context(30_000, 29_999)
        clock = Mock(side_effect=(9_399, 9_400))
        waiter, sampler = Mock(), Mock(side_effect=lambda upper: upper - 1)
        self.hook(
            context, deadline_ms=10_000, now_ms=clock,
            wait_ms=waiter, randbelow=sampler,
        )(1)
        sampler.assert_called_once_with(2)
        waiter.assert_called_once_with(1)
        self.assertEqual(clock.call_count, 2)

    def test_optional_call_deadline_preserves_tighter_runtime_and_exact_boundary(self):
        for runtime, now, expected_delay in ((601, 0, 1), (30_000, 9_400, 0)):
            with self.subTest(runtime=runtime, now=now):
                waiter = Mock()
                self.hook(
                    self.context(runtime, runtime - expected_delay),
                    deadline_ms=10_000,
                    now_ms=Mock(side_effect=(now, now + expected_delay)),
                    wait_ms=waiter,
                )(1)
                waiter.assert_called_once_with(expected_delay)

    def test_call_deadline_is_rechecked_after_wait_and_between_retries(self):
        for readings, actions in (((9_375, 9_401), 1),
                                  ((9_000, 9_025, 9_401), 2)):
            with self.subTest(readings=readings):
                waiter = Mock()
                hook = self.hook(
                    self.context(30_000, 29_975, 29_900),
                    deadline_ms=10_000, now_ms=Mock(side_effect=readings),
                    wait_ms=waiter,
                )
                with self.assertRaises(StatusRetryBudgetExhaustedError):
                    for number in range(1, actions + 1):
                        hook(number)
                self.assertEqual(waiter.call_count, 1)

    def test_call_clock_failures_and_invalid_readings_have_safe_diagnostics(self):
        for value in (None, True, -1, 1.5, "1000",
                      RuntimeError("synthetic-private-clock")):
            with self.subTest(value=value):
                hook = self.hook(
                    self.context(30_000), deadline_ms=10_000,
                    now_ms=Mock(side_effect=[value]),
                )
                with self.assertRaisesRegex(ValueError, "^call budget clock") as raised:
                    hook(1)
                self.assertIsNone(raised.exception.__cause__)
                self.assertNotIn("synthetic-private-clock", str(raised.exception))
                if isinstance(value, Exception):
                    self.assertTrue(raised.exception.__suppress_context__)

    def test_clock_rollback_and_runtime_reset_cannot_hide_behind_the_other_bound(self):
        for runtime, clock, error in (
            ((1_000, 975), (7_000, 6_999), "clock must not move backwards"),
            ((30_000, 30_001), (9_000, 9_025), "invocation time must not increase"),
            ((1_000, 975, 900), (7_000, 7_025, 7_024),
             "clock must not move backwards"),
            ((30_000, 29_975, 29_976), (9_000, 9_025, 9_050),
             "invocation time must not increase"),
        ):
            with self.subTest(runtime=runtime):
                hook = self.hook(
                    self.context(*runtime), deadline_ms=10_000,
                    now_ms=Mock(side_effect=clock),
                )
                with self.assertRaisesRegex(ValueError, error):
                    hook(1)
                    if len(runtime) == 3:
                        hook(2)

    def test_first_success_does_not_sample_optional_call_clock(self):
        clock = Mock()
        result = self.observe(
            InMemoryCallStatusStore(), Mock(return_value=True),
            self.hook(self.context(), deadline_ms=10_000, now_ms=clock),
        )
        self.assertEqual(result, CallStatusSnapshot("completed", 1))
        clock.assert_not_called()

    def test_call_expiry_after_wait_prevents_second_authenticated_write(self):
        store = Mock(spec=["load", "observe"])
        store.load.return_value = CallStatusSnapshot(None, 0)
        store.observe.side_effect = StatusConflictError("synthetic conflict")
        validator, waiter = Mock(return_value=True), Mock()
        hook = self.hook(
            self.context(30_000, 29_975), deadline_ms=10_000,
            now_ms=Mock(side_effect=(9_375, 9_401)), wait_ms=waiter,
        )
        with self.assertRaises(StatusRetryBudgetExhaustedError):
            self.observe(store, validator, hook)
        waiter.assert_called_once_with(25)
        self.assertEqual(validator.call_count, 1)
        self.assertEqual(store.load.call_count, 1)
        self.assertEqual(store.observe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
