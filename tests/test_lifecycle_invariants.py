"""Bounded exhaustive checks of local status storage and voice admission.

Enumerate synthetic observation orderings, not provider delivery guarantees.
No live adapters, distributed writes, automatic retries, or cleanup are used.
"""

import unittest
from itertools import product

from zentomic.budget import resolve_voice_step_budget
from zentomic.call_status_store import InMemoryCallStatusStore, StatusConflictError


ACTIVE = ("queued", "ringing", "in-progress")
TERMINAL = ("completed", "busy", "failed", "no-answer", "canceled")


def expected_status(history):
    # Independent prefix oracle: first terminal wins, otherwise furthest active.
    # Do not call the production transition policy to compute expected results.
    ended = [status for status in history if status in TERMINAL]
    return ended[0] if ended else max(history, key=ACTIVE.index)


class LifecycleInvariantTests(unittest.TestCase):
    def test_all_four_observation_sequences_preserve_storage_and_admission(self):
        # 8**4 orderings include duplicates, skipped/regressive active states,
        # conflicting terminal outcomes, and callbacks delayed after termination.
        for sequence in product(ACTIVE + TERMINAL, repeat=4):
            with self.subTest(sequence=sequence):
                store = InMemoryCallStatusStore(max_entries=1)
                history = []
                snapshots = []
                previous_status = None
                revision = 0
                for incoming in sequence:
                    history.append(incoming)
                    expected = expected_status(history)
                    revision += int(expected != previous_status)
                    loaded = store.load("workspace", "call")
                    saved = store.observe("workspace", "call", incoming,
                                          expected_revision=loaded.revision)
                    self.assertEqual((saved.status, saved.revision), (expected, revision))
                    self.assertEqual(store.load("workspace", "call"), saved)
                    snapshots.append((saved, expected, revision))

                    # Even a duplicate must reject a stale revision. Failure
                    # cannot replace a snapshot, resurrect a call, or spend one.
                    with self.assertRaises(StatusConflictError):
                        store.observe("workspace", "call", incoming,
                                      expected_revision=revision - 1)
                    self.assertEqual(store.load("workspace", "call"), saved)

                    # Plenty of time/count headroom cannot admit a terminal
                    # call, nor spend a transition on that denial.
                    decision = resolve_voice_step_budget(
                        call_status=saved.status, deadline_ms=10_000, now_ms=100,
                        transitions=2, max_transitions=10,
                    )
                    terminal = expected in TERMINAL
                    self.assertEqual(
                        (decision.action, decision.transitions, decision.remaining_ms),
                        ("hangup" if terminal else "continue", 2 if terminal else 3, 9900),
                    )
                    previous_status = expected

                # Later writes must not mutate objects held by earlier readers.
                for snapshot, status, version in snapshots:
                    self.assertEqual((snapshot.status, snapshot.revision), (status, version))


if __name__ == "__main__":
    unittest.main()
