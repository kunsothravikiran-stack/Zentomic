"""Offline keypad boundary contracts with synthetic, noncryptographic fixtures."""

import copy
import unittest
from unittest.mock import Mock, patch

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.gather import GatherDecision
from zentomic.gather_event import resolve_gather_event


class GatherEventTests(unittest.TestCase):
    def resolve(self, event, validator, routes=None, **overrides):
        config = dict(public_url="https://example.invalid/voice/menu",
                      expected_account_sid=ACCOUNT, expected_call_sid=CALL,
                      fallback_target="demo-reception", attempts=0,
                      max_attempts=3, hangup_digit="9")
        config.update(overrides)
        return resolve_gather_event(
            event, {"1": "demo-sales"} if routes is None else routes,
            validator=validator, **config,
        )

    def test_authenticated_inputs_share_existing_policy_in_all_transports(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for fields, attempts, expected in (
                    ({"Digits": "1"}, 0, GatherDecision("route", "demo-sales", 1)),
                    ({"Digits": "9"}, 0, GatherDecision("hangup", None, 1)),
                    ({}, 0, GatherDecision("retry", None, 1)),
                    ({"Digits": ""}, 1, GatherDecision("retry", None, 2)),
                    ({"Digits": "11"}, 2, GatherDecision("fallback", "demo-reception", 3)),
                    ({"Digits": "1"}, 3, GatherDecision("fallback", "demo-reception", 3)),
                ):
                    with self.subTest(version=version, encoded=encoded,
                                      fields=fields, attempts=attempts):
                        event = callback(dict(fields, attempts="0", max_attempts="999",
                                              fallback_target="untrusted", hangup_digit="1"),
                                         version, encoded)
                        original = copy.deepcopy(event)
                        validator = Mock(return_value=True)
                        self.assertEqual(self.resolve(event, validator, attempts=attempts), expected)
                        validator.assert_called_once()
                        self.assertEqual(event, original)

    def test_failed_authentication_or_identity_never_consumes_policy(self):
        for fields, valid in (({}, False), ({}, 1),
                              ({"AccountSid": "other-account"}, True),
                              ({"CallSid": "other-call"}, True)):
            with self.subTest(fields=fields, valid=valid), patch(
                "zentomic.gather_event.resolve_gather",
            ) as policy:
                with self.assertRaises(ValueError):
                    self.resolve(callback(fields, "2.0", False), Mock(return_value=valid))
                policy.assert_not_called()

    def test_validator_cannot_rewrite_digits_or_live_route_configuration(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    routes = {"1": "demo-sales"}

                    def mutate(url, fields, signature):
                        fields["Digits"] = "9"
                        routes["1"] = "untrusted"
                        return True

                    event = callback({"Digits": "1"}, version, encoded)
                    original = copy.deepcopy(event)
                    self.assertEqual(self.resolve(event, mutate, routes),
                                     GatherDecision("route", "demo-sales", 1))
                    self.assertEqual(event, original)

    def test_invalid_policy_configuration_is_not_hidden_by_valid_input(self):
        for overrides in ({"attempts": True}, {"max_attempts": 0},
                          {"fallback_target": ""}, {"hangup_digit": "1"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(callback({"Digits": "1"}, "2.0", False),
                             Mock(return_value=True), **overrides)
        with self.assertRaises(ValueError):
            self.resolve(callback({"Digits": "1"}, "2.0", False),
                         Mock(return_value=True), {"11": "demo-sales"})

    def test_invalid_transport_and_route_container_fail_before_validator(self):
        for event, routes in (({}, {}), (callback({}, "2.0", False), [])):
            validator = Mock(return_value=True)
            with self.subTest(event=event, routes=routes), self.assertRaises(ValueError):
                self.resolve(event, validator, routes)
            validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
