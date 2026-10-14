"""Lambda context timeout admission with local synthetic runtime readings."""

import unittest
from unittest.mock import Mock

from zentomic.lambda_budget import resolve_lambda_operation_timeout


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


if __name__ == "__main__":
    unittest.main()
