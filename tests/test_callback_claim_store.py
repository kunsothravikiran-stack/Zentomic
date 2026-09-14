"""Offline callback claim tests using synthetic identifiers only."""

import copy
import unittest
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from urllib.parse import urlencode

from zentomic.callback_claim_store import (
    MAX_CLAIM_IDENTIFIER_BYTES,
    CallbackClaimCapacityError,
    CallbackReplayError,
    InMemoryCallbackClaimStore,
    validate_and_claim_call_event,
)


ACCOUNT = "synthetic-account"
CALL = "synthetic-call"
URL = "https://example.invalid/voice/menu"
SIGNATURE = "A" * 27 + "="


def callback(fields=None, *, version="2.0", encoded=False):
    body = urlencode({"AccountSid": ACCOUNT, "CallSid": CALL, **(fields or {})})
    event = {
        "version": version,
        "body": body,
        "isBase64Encoded": encoded,
        "headers": {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": SIGNATURE,
        },
    }
    if encoded:
        event["body"] = b64encode(body.encode()).decode()
    if version == "1.0":
        event["httpMethod"] = "POST"
    else:
        event["requestContext"] = {"http": {"method": "POST"}}
    return event


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


class CallbackClaimEventTests(unittest.TestCase):
    def claim(self, event, claimer, validator=None, **overrides):
        config = dict(
            public_url=URL,
            validator=validator or Mock(return_value=True),
            expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
            workspace_id="synthetic-workspace",
            step_id="menu-attempt-1",
            claimer=claimer,
        )
        config.update(overrides)
        return validate_and_claim_call_event(event, **config)

    def test_authenticated_event_is_claimed_once_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    store = InMemoryCallbackClaimStore()
                    event = callback(
                        {"Digits": "1", "workspace_id": "untrusted",
                         "step_id": "untrusted"},
                        version=version, encoded=encoded,
                    )
                    original = copy.deepcopy(event)
                    fields = self.claim(event, store)
                    self.assertEqual(fields["Digits"], "1")
                    self.assertEqual(event, original)
                    with self.assertRaisesRegex(
                        CallbackReplayError, "^callback step already claimed$",
                    ):
                        self.claim(event, store)

                    # Signed fields cannot choose a different workspace or step.
                    self.assertFalse(store.claim(
                        "synthetic-workspace", CALL, "menu-attempt-1",
                    ))
                    self.assertTrue(store.claim("untrusted", CALL, "untrusted"))

    def test_rejected_callback_does_not_poison_the_trusted_claim(self):
        store = InMemoryCallbackClaimStore()
        cases = (
            (callback(), Mock(return_value=False)),
            (callback({"AccountSid": "synthetic-other"}), Mock(return_value=True)),
            ({}, Mock(return_value=True)),
        )
        for event, validator in cases:
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    self.claim(event, store, validator)
                self.assertTrue(store.claim(
                    "synthetic-workspace", CALL, "menu-attempt-1",
                ))
                store = InMemoryCallbackClaimStore()

    def test_invalid_trusted_claim_configuration_precedes_authentication(self):
        for overrides in (
            {"workspace_id": ""},
            {"step_id": " "},
            {"expected_call_sid": "x" * (MAX_CLAIM_IDENTIFIER_BYTES + 1)},
        ):
            validator = Mock(return_value=True)
            claimer = Mock()
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.claim(callback(), claimer, validator, **overrides)
            validator.assert_not_called()
            claimer.claim.assert_not_called()

        validator = Mock(return_value=True)
        with self.assertRaisesRegex(
            ValueError, "^claimer must provide a trusted callable claim method$",
        ):
            self.claim(callback(), object(), validator)
        validator.assert_not_called()

    def test_claimer_must_return_an_exact_boolean(self):
        for value in (None, 0, 1, "true", [], {}):
            claimer = Mock()
            claimer.claim.return_value = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "^claimer must return exactly True or False$",
            ):
                self.claim(callback(), claimer)

    def test_capacity_failures_propagate_without_claiming_another_step(self):
        store = InMemoryCallbackClaimStore(max_entries=1)
        self.assertTrue(store.claim("synthetic-workspace", CALL, "existing"))
        with self.assertRaisesRegex(
            CallbackClaimCapacityError, "^callback claim capacity reached$",
        ):
            self.claim(callback(), store)
        self.assertFalse(store.claim("synthetic-workspace", CALL, "existing"))

    def test_concurrent_authenticated_replays_admit_exactly_one(self):
        store = InMemoryCallbackClaimStore()
        event = callback({"Digits": "1"})

        def attempt(_):
            try:
                self.claim(event, store, validator=lambda *_: True)
            except CallbackReplayError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(attempt, range(64)))

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 63)


if __name__ == "__main__":
    unittest.main()
