"""Local conditional lifecycle writes, using synthetic identities only."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier

from zentomic.call_status_store import (
    InMemoryCallStatusStore, StatusCapacityError, StatusConflictError,
)


class CallStatusStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryCallStatusStore()

    def observe(self, status, revision=0, workspace="workspace-a", call="call-a"):
        return self.store.observe(workspace, call, status, expected_revision=revision)

    def test_new_state_is_immutable_and_stores_are_independent(self):
        initial = self.store.load("workspace-a", "call-a")
        self.assertEqual((initial.status, initial.revision), (None, 0))
        with self.assertRaises(FrozenInstanceError):
            initial.revision = 9
        saved = self.observe("ringing")
        self.assertEqual((saved.status, saved.revision), ("ringing", 1))
        self.assertEqual(initial.status, None)
        self.assertEqual(InMemoryCallStatusStore().load("workspace-a", "call-a"), initial)

    def test_workspace_and_call_keys_do_not_collide(self):
        self.observe("completed")
        self.observe("ringing", workspace="workspace-b")
        self.observe("queued", call="call-b")
        self.observe("busy", workspace="a/b", call="c")
        self.observe("failed", workspace="a", call="b/c")
        for workspace, call, expected in (
            ("workspace-a", "call-a", "completed"),
            ("workspace-b", "call-a", "ringing"),
            ("workspace-a", "call-b", "queued"),
            ("a/b", "c", "busy"), ("a", "b/c", "failed"),
        ):
            self.assertEqual(self.store.load(workspace, call).status, expected)

    def test_stale_write_conflicts_then_reload_preserves_terminal_state(self):
        stale = self.store.load("workspace-a", "call-a")
        ended = self.observe("completed", stale.revision)
        with self.assertRaisesRegex(StatusConflictError, "^call status revision conflict$"):
            self.observe("ringing", stale.revision)
        latest = self.store.load("workspace-a", "call-a")
        self.assertEqual(self.observe("ringing", latest.revision), ended)

    def test_noop_observations_do_not_spend_revisions(self):
        active = self.observe("in-progress")
        for status in ("queued", "ringing", "in-progress"):
            self.assertEqual(self.observe(status, active.revision), active)
        ended = self.observe("busy", active.revision)
        self.assertEqual(ended.revision, 2)
        for status in ("completed", "busy", "failed", "no-answer", "canceled", "queued"):
            self.assertEqual(self.observe(status, ended.revision), ended)
        # Even a no-op requires the current revision.
        with self.assertRaises(StatusConflictError):
            self.observe("busy", active.revision)

    def test_two_writers_using_one_revision_cannot_both_succeed(self):
        barrier = Barrier(2)

        def write(status):
            snapshot = self.store.load("workspace-a", "call-a")
            barrier.wait(timeout=5)
            try:
                return self.observe(status, snapshot.revision)
            except StatusConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, ("ringing", "completed")))
        saved = [result for result in results if result is not None]
        self.assertEqual(len(saved), 1)
        self.assertEqual(self.store.load("workspace-a", "call-a"), saved[0])
        self.assertEqual(saved[0].revision, 1)

    def test_capacity_rejects_new_keys_without_evicting_terminal_state(self):
        store = InMemoryCallStatusStore(max_entries=1)
        saved = store.observe("workspace-a", "call-a", "completed", expected_revision=0)
        with self.assertRaisesRegex(StatusCapacityError, "^call status capacity reached$"):
            store.observe("private-workspace", "private-call", "ringing", expected_revision=0)
        self.assertEqual(store.load("workspace-a", "call-a"), saved)
        self.assertEqual(store.load("private-workspace", "private-call").revision, 0)
        self.assertEqual(store.observe("workspace-a", "call-a", "ringing",
                                      expected_revision=1), saved)

    def test_reads_do_not_spend_capacity_and_existing_keys_can_advance(self):
        store = InMemoryCallStatusStore(max_entries=1)
        for index in range(3):
            self.assertEqual(store.load("workspace", str(index)).revision, 0)
        active = store.observe("workspace", "call", "ringing", expected_revision=0)
        ended = store.observe("workspace", "call", "completed",
                              expected_revision=active.revision)
        self.assertEqual((ended.status, ended.revision), ("completed", 2))
        with self.assertRaises(StatusConflictError):
            store.observe("workspace", "call", "completed", expected_revision=0)

    def test_capacity_must_be_a_positive_integer(self):
        for limit in (None, True, False, 0, -1, 1.0, "1", []):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                InMemoryCallStatusStore(max_entries=limit)

    def test_identifier_byte_limit_accepts_exact_boundaries_without_normalizing(self):
        # Each identity is bounded separately, in UTF-8 bytes, not characters.
        for identity in ("a" * 256, "é" * 128, "😀" * 64, " a "):
            for workspace, call in ((identity, "call"), ("workspace", identity)):
                with self.subTest(workspace=workspace, call=call):
                    saved = self.store.observe(workspace, call, "completed",
                                              expected_revision=0)
                    self.assertEqual(self.store.load(workspace, call), saved)
        self.assertEqual(self.store.load("a", "call").revision, 0)
        self.assertEqual(self.store.load("workspace", "a").revision, 0)

    def test_oversized_identifiers_are_rejected_without_spending_capacity(self):
        store = InMemoryCallStatusStore(max_entries=1)
        for identity in ("a" * 257, "é" * 128 + "a", "😀" * 64 + "a",
                         "private-" + "x" * 10000):
            for workspace, call in ((identity, "call"), ("workspace", identity)):
                for operation in (
                    lambda: store.load(workspace, call),
                    lambda: store.observe(workspace, call, "ringing",
                                          expected_revision=0),
                ):
                    with self.subTest(identity=identity[:8]), self.assertRaisesRegex(
                        ValueError,
                        "^status identifiers must be nonblank UTF-8 strings of at most 256 bytes$",
                    ):
                        operation()
        saved = store.observe("workspace", "call", "completed", expected_revision=0)
        self.assertEqual((saved.status, saved.revision), ("completed", 1))
        self.assertEqual(store.load("workspace", "call"), saved)

    def test_two_new_keys_cannot_exceed_capacity(self):
        store = InMemoryCallStatusStore(max_entries=1)
        barrier = Barrier(2)

        def write(call):
            barrier.wait(timeout=5)
            try:
                store.observe("workspace", call, "ringing", expected_revision=0)
                return call
            except StatusCapacityError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, ("call-a", "call-b")))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(sum(store.load("workspace", call).revision
                             for call in ("call-a", "call-b")), 1)

    def test_invalid_input_never_changes_state_or_reflects_identifiers(self):
        saved = self.observe("completed")
        for revision in (-1, True, 1.0, "1", None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.observe("ringing", revision)
        for status in (None, "private-invalid-status", [], True):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.observe(status, saved.revision)
        for identity in (None, [], "", " ", "private-\ud800"):
            for workspace, call in ((identity, "call-a"), ("workspace-a", identity)):
                for operation in (
                    lambda: self.store.load(workspace, call),
                    lambda: self.observe("ringing", workspace=workspace, call=call),
                ):
                    with self.assertRaises(ValueError) as caught:
                        operation()
                    self.assertNotIn("private-", str(caught.exception))
        self.assertEqual(self.store.load("workspace-a", "call-a"), saved)


if __name__ == "__main__":
    unittest.main()
