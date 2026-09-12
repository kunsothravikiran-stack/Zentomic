"""Offline confirmation boundary tests with synthetic call identities."""

import copy
import unittest
from unittest.mock import Mock, patch

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.confirmation_event import resolve_confirmation_event
from zentomic.gather import GatherDecision


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


if __name__ == "__main__":
    unittest.main()
