"""Offline callback claim tests using synthetic identifiers only."""

import unittest
from concurrent.futures import ThreadPoolExecutor

from zentomic.callback_claim_store import (
    MAX_CLAIM_IDENTIFIER_BYTES,
    CallbackClaimCapacityError,
    InMemoryCallbackClaimStore,
)


class CallbackClaimStoreTests(unittest.TestCase):
    def test_first_claim_wins_and_exact_replay_is_denied(self):
        store = InMemoryCallbackClaimStore()

        self.assertIs(store.claim("workspace-a", "call-a", "menu-attempt-1"), True)
        self.assertIs(store.claim("workspace-a", "call-a", "menu-attempt-1"), False)

    def test_claims_are_isolated_by_workspace_call_and_step(self):
        store = InMemoryCallbackClaimStore()
        keys = (
            ("workspace-a", "call-a", "step-a"),
            ("workspace-b", "call-a", "step-a"),
            ("workspace-a", "call-b", "step-a"),
            ("workspace-a", "call-a", "step-b"),
        )

        self.assertEqual([store.claim(*key) for key in keys], [True] * len(keys))
        self.assertEqual([store.claim(*key) for key in keys], [False] * len(keys))

    def test_identifier_boundaries_preserve_exact_values(self):
        store = InMemoryCallbackClaimStore()
        ascii_boundary = "a" * MAX_CLAIM_IDENTIFIER_BYTES
        unicode_boundary = "😀" * (MAX_CLAIM_IDENTIFIER_BYTES // 4)

        self.assertTrue(store.claim(ascii_boundary, unicode_boundary, " step "))
        self.assertFalse(store.claim(ascii_boundary, unicode_boundary, " step "))
        self.assertTrue(store.claim(ascii_boundary, unicode_boundary, "step"))

    def test_invalid_identifiers_do_not_consume_capacity(self):
        store = InMemoryCallbackClaimStore(max_entries=1)
        invalid = (
            None, "", "   ", [],
            "a" * (MAX_CLAIM_IDENTIFIER_BYTES + 1),
            "😀" * (MAX_CLAIM_IDENTIFIER_BYTES // 4) + "a",
            "\ud800",
        )

        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaisesRegex(
                ValueError, "^claim identifiers must be nonblank UTF-8 strings",
            ):
                store.claim("workspace", "call", value)
        self.assertTrue(store.claim("workspace", "call", "valid-step"))

    def test_capacity_rejects_new_keys_but_preserves_replay_detection(self):
        store = InMemoryCallbackClaimStore(max_entries=1)
        self.assertTrue(store.claim("workspace", "call", "step-1"))

        with self.assertRaisesRegex(
            CallbackClaimCapacityError, "^callback claim capacity reached$",
        ):
            store.claim("workspace", "call", "step-2")
        self.assertFalse(store.claim("workspace", "call", "step-1"))

    def test_max_entries_is_a_strict_positive_integer(self):
        for value in (None, True, False, 0, -1, 1.0, "1", []):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "^max_entries must be a positive integer$",
            ):
                InMemoryCallbackClaimStore(max_entries=value)

    def test_competing_threads_admit_exactly_one_claim(self):
        store = InMemoryCallbackClaimStore()
        key = ("workspace", "call", "step")

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(lambda _: store.claim(*key), range(64)))

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 63)


if __name__ == "__main__":
    unittest.main()
