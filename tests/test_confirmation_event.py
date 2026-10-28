"""Offline confirmation boundary tests with synthetic call identities."""

import copy
import unittest
from unittest.mock import Mock, patch

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.callback_claim_store import (
    CallbackReplayError,
    InMemoryCallbackClaimStore,
)
from zentomic.confirmation_event import (
    resolve_claimed_confirmation_event,
    resolve_confirmation_event,
)
from zentomic.gather import GatherDecision
from tests.test_intent import OversizedIntentRoutes


class ConfirmationEventTests(unittest.TestCase):
    def resolve(self, event, validator, **overrides):
        config = dict(public_url="https://example.invalid/voice/confirm",
                      expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                      pending_intent="support", routes={"support": "demo-support"},
                      fallback_target="demo-reception", attempts=0, max_attempts=3,
                      hangup_digit="9")
        config.update(overrides)
        return resolve_confirmation_event(event, validator=validator, **config)

    def test_authenticated_choice_obeys_trusted_pending_intent_and_budget(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for digits, pending, attempts, expected in (
                    ("1", "support", 0, GatherDecision("route", "demo-support", 1)),
                    ("2", "support", 0, GatherDecision("fallback", "demo-reception", 1)),
                    ("9", "support", 0, GatherDecision("hangup", None, 1)),
                    (None, "support", 0, GatherDecision("retry", None, 1)),
                    ("yes", "support", 1, GatherDecision("retry", None, 2)),
                    ("1", "support", 2, GatherDecision("route", "demo-support", 3)),
                    ("1", "support", 3, GatherDecision("fallback", "demo-reception", 3)),
                    ("1", None, 0, GatherDecision("fallback", "demo-reception", 1)),
                    ("1", "unknown", 0, GatherDecision("fallback", "demo-reception", 1)),
                ):
                    with self.subTest(version=version, encoded=encoded, digits=digits,
                                      pending=pending, attempts=attempts):
                        fields = dict(intent="support", pending_intent="support",
                                      confirmed="true", attempts="0", max_attempts="999",
                                      fallback_target="untrusted", SpeechResult="yes")
                        if digits is not None:
                            fields["Digits"] = digits
                        event = callback(fields, version, encoded)
                        original = copy.deepcopy(event)
                        validator = Mock(return_value=True)
                        self.assertEqual(self.resolve(event, validator,
                                                      pending_intent=pending, attempts=attempts),
                                         expected)
                        validator.assert_called_once()
                        self.assertEqual(event, original)

    def test_rejected_identity_or_signature_never_reaches_policy(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, valid in (({}, False), ({}, 1),
                                      ({"AccountSid": "other-account"}, True),
                                      ({"CallSid": "other-call"}, True)):
                    with self.subTest(version=version, encoded=encoded, fields=fields,
                                      valid=valid), patch(
                        "zentomic.confirmation_event.resolve_intent_confirmation",
                    ) as policy:
                        with self.assertRaises(ValueError):
                            self.resolve(callback(dict(fields, Digits="1"), version, encoded),
                                         Mock(return_value=valid))
                        policy.assert_not_called()

    def test_validator_cannot_change_choice_or_routes(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, expected in (
                    ({"Digits": "1"}, GatherDecision("route", "demo-support", 1)),
                    ({"Digits": "2"}, GatherDecision("fallback", "demo-reception", 1)),
                    ({}, GatherDecision("retry", None, 1)),
                ):
                    with self.subTest(version=version, encoded=encoded, fields=fields):
                        routes = {"support": "demo-support"}

                        def mutate(url, values, signature):
                            values["Digits"] = "1"
                            routes["support"] = "injected-target"
                            return True

                        event = callback(fields, version, encoded)
                        original = copy.deepcopy(event)
                        self.assertEqual(self.resolve(event, mutate, routes=routes), expected)
                        self.assertEqual(event, original)

    def test_invalid_transport_never_reaches_validator_or_policy(self):
        duplicate = callback({}, "2.0", False)
        duplicate["body"] += "&Digits=1&Digits=2"
        for event in ({}, duplicate):
            validator = Mock(return_value=True)
            with self.subTest(event=event), patch(
                "zentomic.confirmation_event.resolve_intent_confirmation",
            ) as policy:
                with self.assertRaises(ValueError):
                    self.resolve(event, validator)
                validator.assert_not_called()
                policy.assert_not_called()

    def test_invalid_trusted_configuration_raises(self):
        for overrides in ({"routes": []}, {"routes": {"support": ""}},
                          {"attempts": True}, {"attempts": -1}, {"max_attempts": 0},
                          {"fallback_target": ""}, {"hangup_digit": "1"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(callback({"Digits": "1"}, "2.0", False),
                             Mock(return_value=True), **overrides)

    def test_oversized_routes_are_rejected_before_authentication(self):
        validator = Mock(return_value=True)
        with self.assertRaisesRegex(
            ValueError, "^routes must contain at most 128 entries$",
        ):
            self.resolve(callback({"Digits": "1"}, "2.0", False), validator,
                         routes=OversizedIntentRoutes())
        validator.assert_not_called()


class ClaimedConfirmationEventTests(unittest.TestCase):
    def resolve(self, event, claimer, validator=None, **overrides):
        config = dict(
            pending_intent="support",
            routes={"support": "demo-support"},
            public_url="https://example.invalid/voice/confirm",
            validator=validator or Mock(return_value=True),
            expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
            workspace_id="synthetic-workspace",
            step_id="confirmation-attempt-1",
            claimer=claimer,
            fallback_target="demo-reception",
            attempts=0,
            max_attempts=3,
            hangup_digit="9",
        )
        config.update(overrides)
        return resolve_claimed_confirmation_event(event, **config)

    def test_authenticated_step_is_confirmed_once_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    store = InMemoryCallbackClaimStore()
                    event = callback(
                        {"Digits": "1", "pending_intent": "untrusted",
                         "workspace_id": "untrusted", "step_id": "untrusted",
                         "attempts": "99", "confirmed": "true"},
                        version, encoded,
                    )
                    original = copy.deepcopy(event)
                    self.assertEqual(
                        self.resolve(event, store),
                        GatherDecision("route", "demo-support", 1),
                    )
                    self.assertEqual(event, original)
                    with self.assertRaisesRegex(
                        CallbackReplayError, "^callback step already claimed$",
                    ), patch(
                        "zentomic.confirmation_event._resolve_validated_intent_confirmation",
                    ) as policy:
                        self.resolve(event, store)
                    policy.assert_not_called()
                    self.assertFalse(store.claim(
                        "synthetic-workspace", CALL, "confirmation-attempt-1",
                    ))
                    self.assertTrue(store.claim("untrusted", CALL, "untrusted"))

    def test_rejected_callback_does_not_consume_the_step(self):
        for fields, valid in (
            ({}, False),
            ({"AccountSid": "other-account"}, True),
            ({"CallSid": "other-call"}, True),
        ):
            store = InMemoryCallbackClaimStore()
            with self.subTest(fields=fields, valid=valid), self.assertRaises(ValueError):
                self.resolve(callback(fields, "2.0", False), store, Mock(return_value=valid))
            self.assertTrue(store.claim(
                "synthetic-workspace", CALL, "confirmation-attempt-1",
            ))

    def test_invalid_policy_does_not_authenticate_or_claim(self):
        for overrides in (
            {"routes": {"support": ""}},
            {"fallback_target": ""},
            {"attempts": True},
            {"max_attempts": 0},
            {"hangup_digit": "1"},
        ):
            validator = Mock(return_value=True)
            store = InMemoryCallbackClaimStore()
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(callback({"Digits": "1"}, "2.0", False), store,
                             validator, **overrides)
            validator.assert_not_called()
            self.assertTrue(store.claim(
                "synthetic-workspace", CALL, "confirmation-attempt-1",
            ))

    def test_claim_cannot_mutate_or_revalidate_the_policy_snapshot(self):
        class SingleValidationTarget(str):
            checks = 0

            def encode(self, *args, **kwargs):
                self.checks += 1
                if self.checks > 1:
                    raise AssertionError("target was validated after the claim")
                return super().encode(*args, **kwargs)

        target = SingleValidationTarget("demo-support")
        routes = {"support": target}
        claimer = Mock()

        def claim(*_):
            routes["support"] = "untrusted-target"
            return True

        claimer.claim.side_effect = claim
        self.assertEqual(
            self.resolve(callback({"Digits": "1"}, "2.0", False), claimer,
                         routes=routes),
            GatherDecision("route", target, 1),
        )
        self.assertEqual(target.checks, 1)
        claimer.claim.assert_called_once_with(
            "synthetic-workspace", CALL, "confirmation-attempt-1",
        )


if __name__ == "__main__":
    unittest.main()
