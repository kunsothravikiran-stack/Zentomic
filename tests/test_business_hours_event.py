"""Synthetic claimed hours decisions; no real signatures, clocks or services."""

import copy
from datetime import date, datetime, timezone
import unittest
from unittest.mock import Mock, patch

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.business_hours import resolve_business_hours_target_at
from zentomic.business_hours_event import resolve_claimed_business_hours_event
from zentomic.callback_claim_store import CallbackReplayError, InMemoryCallbackClaimStore


class ClaimedBusinessHoursEventTests(unittest.TestCase):
    def resolve(self, event, claimer, **overrides):
        config = dict(
            hours={0: [(540, 1020)]},
            instant=datetime(2026, 1, 5, 14, tzinfo=timezone.utc),
            timezone_name="America/New_York",
            open_target="demo-team", closed_target="demo-voicemail",
            public_url="https://example.invalid/voice/hours",
            validator=Mock(return_value=True),
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id="synthetic-workspace", step_id="hours-step",
        )
        config.update(overrides)
        hours = config.pop("hours")
        return resolve_claimed_business_hours_event(event, hours, claimer=claimer, **config)

    def test_authentication_precedes_claim_in_all_proxy_transports(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    order = []
                    validator = Mock(side_effect=lambda *_: order.append("auth") or True)
                    claimer = Mock()
                    claimer.claim.side_effect = lambda *_: order.append("claim") or True
                    event = callback({"instant": "private-time", "timezone_name": "UTC",
                                      "target": "untrusted", "workspace_id": "untrusted",
                                      "step_id": "untrusted", "Digits": "9"}, version, encoded)
                    before = copy.deepcopy(event)
                    self.assertEqual(self.resolve(event, claimer, validator=validator), "demo-team")
                    self.assertEqual(order, ["auth", "claim"])
                    claimer.claim.assert_called_once_with("synthetic-workspace", CALL, "hours-step")
                    self.assertEqual(event, before)

    def test_closed_boundary_and_local_date_override_use_existing_policy(self):
        event = callback({}, "2.0", False)
        cases = (
            ({"instant": datetime(2026, 1, 5, 13, 59, 59, tzinfo=timezone.utc)}, "demo-voicemail"),
            ({"instant": datetime(2026, 1, 5, 22, tzinfo=timezone.utc)}, "demo-voicemail"),
            ({"date_overrides": {date(2026, 1, 5): []}}, "demo-voicemail"),
            ({"hours": {}, "date_overrides": {date(2026, 1, 5): [(540, 541)]}}, "demo-team"),
            ({"instant": datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
              "hours": {2: [(0, 1440)]},
              "date_overrides": {date(2025, 12, 31): []}}, "demo-voicemail"),
        )
        for config, expected in cases:
            with self.subTest(config=config):
                store = InMemoryCallbackClaimStore()
                self.assertEqual(self.resolve(event, store, **config), expected)
                self.assertFalse(store.claim("synthetic-workspace", CALL, "hours-step"))

    def test_invalid_complete_policy_does_not_authenticate_or_claim(self):
        cases = (
            {"hours": {0: [(540, 1020)], 6: [(3, 2)]}},
            {"date_overrides": {date(2026, 1, 6): [(0, 0)]}},
            {"instant": datetime(2026, 1, 5, 14)},
            {"timezone_name": "Private/Missing-Zone"},
            {"closed_target": ""}, {"open_target": "\ud800"},
        )
        for config in cases:
            with self.subTest(config=config):
                validator, claimer = Mock(return_value=True), Mock()
                with self.assertRaises(ValueError):
                    self.resolve(callback({}, "2.0", False), claimer, validator=validator, **config)
                validator.assert_not_called()
                claimer.claim.assert_not_called()

    def test_invalid_transport_signature_or_identity_never_claims(self):
        for event, valid in (({}, True), (callback({}, "2.0", False), False),
                             (callback({}, "2.0", False), 1),
                             (callback({"AccountSid": "other-account"}, "1.0", True), True),
                             (callback({"CallSid": "other-call"}, "2.0", False), True)):
            with self.subTest(event=event, valid=valid):
                store = InMemoryCallbackClaimStore()
                with self.assertRaises(ValueError):
                    self.resolve(event, store, validator=Mock(return_value=valid))
                self.assertTrue(store.claim("synthetic-workspace", CALL, "hours-step"))
        claimer = Mock()
        with self.assertRaises(ValueError):
            self.resolve(callback({}, "2.0", False), claimer,
                         validator=Mock(side_effect=RuntimeError("private-validator")))
        claimer.claim.assert_not_called()

    def test_unreadable_zone_data_does_not_authenticate_or_consume_claim(self):
        event = callback({}, "2.0", False)
        store = InMemoryCallbackClaimStore()
        validator = Mock(return_value=True)
        with patch("zentomic.business_hours.ZoneInfo",
                   side_effect=PermissionError("private-zone-path")):
            with self.assertRaisesRegex(
                ValueError, "^timezone_name must identify an available IANA timezone$",
            ):
                self.resolve(event, store, validator=validator)
        validator.assert_not_called()
        # The same callback can succeed once local zone data is available.
        self.assertEqual(self.resolve(event, store, validator=validator), "demo-team")
        validator.assert_called_once()
        with self.assertRaises(CallbackReplayError):
            self.resolve(event, store)

    def test_validator_and_claimer_cannot_rewrite_the_policy_snapshot(self):
        hours = {0: [[540, 1020]]}
        overrides = {date(2026, 1, 6): []}
        order = []

        def authenticate(*_):
            hours[0][0][:] = [0, 1]
            overrides[date(2026, 1, 5)] = []
            order.append("auth")
            return True

        def claim(*_):
            hours.clear()
            overrides.clear()
            order.append("claim")
            return True

        claimer = Mock()
        claimer.claim.side_effect = claim
        with patch("zentomic.business_hours_event.resolve_business_hours_target_at",
                   wraps=resolve_business_hours_target_at) as policy:
            self.assertEqual(self.resolve(callback({}, "2.0", False), claimer,
                                          validator=authenticate, hours=hours,
                                          date_overrides=overrides), "demo-team")
            policy.assert_called_once()
        self.assertEqual(order, ["auth", "claim"])

    def test_replay_returns_no_target_and_keys_are_workspace_call_step_scoped(self):
        store = InMemoryCallbackClaimStore()
        event = callback({}, "2.0", False)
        self.assertEqual(self.resolve(event, store), "demo-team")
        with self.assertRaises(CallbackReplayError):
            self.resolve(event, store, hours={})
        for config in ({"workspace_id": "other-workspace"}, {"step_id": "other-step"},
                       {"expected_call_sid": "other-call"}):
            with self.subTest(config=config):
                fields = {"CallSid": config.get("expected_call_sid", CALL)}
                self.assertEqual(self.resolve(callback(fields, "2.0", False), store,
                                              **config), "demo-team")

    def test_storage_failure_and_nonboolean_claims_fail_closed(self):
        event = callback({}, "2.0", False)
        for result in (None, 0, 1, "true"):
            claimer = Mock()
            claimer.claim.return_value = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.resolve(event, claimer)
        claimer = Mock()
        claimer.claim.side_effect = RuntimeError("synthetic storage unavailable")
        with self.assertRaisesRegex(RuntimeError, "^synthetic storage unavailable$"):
            self.resolve(event, claimer)

    def test_bad_claim_configuration_does_not_authenticate(self):
        for config in ({"workspace_id": ""}, {"step_id": ""},
                       {"expected_call_sid": ""}):
            validator, claimer = Mock(return_value=True), Mock()
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.resolve(callback({}, "2.0", False), claimer, validator=validator, **config)
            validator.assert_not_called()
            claimer.claim.assert_not_called()

    def test_exact_targets_and_inputs_are_preserved_without_output(self):
        hours, overrides = {0: [[540, 1020]]}, {date(2026, 1, 6): []}
        before = copy.deepcopy((hours, overrides))
        with patch("builtins.print") as output:
            self.assertEqual(self.resolve(callback({}, "2.0", False), InMemoryCallbackClaimStore(),
                                          hours=hours, date_overrides=overrides,
                                          open_target=" équipe "), " équipe ")
            self.assertEqual(self.resolve(callback({}, "2.0", False), InMemoryCallbackClaimStore(),
                                          hours={}, closed_target=" 音声 "), " 音声 ")
            output.assert_not_called()
        self.assertEqual((hours, overrides), before)


if __name__ == "__main__":
    unittest.main()
