"""Exhaustive small-domain checks against enumerated legal timeouts."""

from itertools import product
import unittest

from zentomic.budget import resolve_operation_timeout


class OperationTimeoutInvariantTests(unittest.TestCase):
    def test_timeout_is_the_largest_legal_duration_across_combined_limits(self):
        # Enumerate legal candidates independently of the implementation's
        # cap/reserve/rounding order. Include expired calls, absent/zero runtime
        # budgets, infeasible quantization, and exact admission boundaries.
        limits = [(minimum, maximum)
                  for maximum in range(1, 5)
                  for minimum in range(1, maximum + 1)]
        for remaining, invocation, bounds, reserve, quantum in product(
            range(-1, 6), (None, *range(6)), limits, range(3), range(1, 5),
        ):
            minimum, maximum = bounds
            legal = [duration for duration in range(quantum, maximum + 1, quantum)
                     if duration >= minimum
                     and duration + reserve <= remaining
                     and (invocation is None or duration + reserve <= invocation)]
            expected = max(legal) if legal else None
            with self.subTest(remaining=remaining, invocation=invocation,
                              bounds=bounds, reserve=reserve, quantum=quantum):
                self.assertEqual(resolve_operation_timeout(
                    deadline_ms=remaining + 1, now_ms=1,
                    minimum_ms=minimum, maximum_ms=maximum, reserve_ms=reserve,
                    granularity_ms=quantum, invocation_remaining_ms=invocation,
                ), expected)

    def test_units_and_large_clock_origins_preserve_the_candidate_oracle(self):
        # Exercise realistic second granularity and integers above float's exact
        # range. A clock-origin shift must not change a relative timeout.
        for origin, scale, remaining, invocation in product(
            (0, 10**20), (1, 1000), range(-1, 10), (None, 0, 4, 7, 10),
        ):
            legal = [duration * scale for duration in (2, 4, 6)
                     if duration >= 3 and duration + 1 <= remaining
                     and (invocation is None or duration + 1 <= invocation)]
            with self.subTest(origin=origin, scale=scale,
                              remaining=remaining, invocation=invocation):
                now = origin + scale
                self.assertEqual(resolve_operation_timeout(
                    deadline_ms=now + remaining * scale, now_ms=now,
                    minimum_ms=3 * scale, maximum_ms=7 * scale,
                    reserve_ms=scale, granularity_ms=2 * scale,
                    invocation_remaining_ms=None if invocation is None else invocation * scale,
                ), max(legal) if legal else None)


if __name__ == "__main__":
    unittest.main()
