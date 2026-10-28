"""Authenticated Number action boundaries using only synthetic fixtures."""

import base64
import copy
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from zentomic.callback_claim_store import (
    CallbackReplayError,
    InMemoryCallbackClaimStore,
)
from zentomic.dial import DialDecision
from zentomic.dial_event import (
    resolve_claimed_dial_result_event,
    resolve_dial_result_event,
)


class DialEventTests(unittest.TestCase):
    def event(self, version="2.0", encoded=False, **overrides):
        fields = {"AccountSid": "synthetic-account", "CallSid": "synthetic-parent",
                  "DialCallSid": "synthetic-child", "DialCallStatus": "busy",
                  "CallStatus": "in-progress", **overrides}
        body = urlencode({key: value for key, value in fields.items() if value is not None})
        event = {"version": version, "body": body, "isBase64Encoded": encoded,
                 "headers": {"Content-Type": "application/x-www-form-urlencoded",
                             "X-Twilio-Signature": "A" * 27 + "="}}
        if encoded:
            event["body"] = base64.b64encode(body.encode()).decode()
        if version == "1.0":
            event["httpMethod"] = "POST"
        else:
            event["requestContext"] = {"http": {"method": "POST"}}
        return event

    def resolve(self, event, validator, **overrides):
        config = {"public_url": "https://example.com/voice/dial-result",
                  "expected_account_sid": "synthetic-account",
                  "expected_call_sid": "synthetic-parent",
                  "expected_dial_call_sid": "synthetic-child",
                  "fallback_target": "demo-reception", "fallback_used": False}
        config.update(overrides)
        return resolve_dial_result_event(event, validator=validator, **config)

    def test_authenticated_outcomes_in_both_transport_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for status in ("busy", "no-answer", "failed", "completed", "canceled"):
                    for used in (False, True):
                        with self.subTest(version=version, encoded=encoded, status=status, used=used):
                            event = self.event(version, encoded, DialCallStatus=status,
                                               fallback_used="false", fallback_target="untrusted")
                            original = copy.deepcopy(event)
                            validator = Mock(return_value=True)
                            decision = self.resolve(event, validator, fallback_used=used)
                            expected = (DialDecision("fallback", "demo-reception")
                                        if status in ("busy", "no-answer", "failed") and not used
                                        else DialDecision("hangup", None))
                            self.assertEqual(decision, expected)
                            validator.assert_called_once()
                            self.assertEqual(event, original)

    def test_invalid_trusted_state_fails_before_authentication(self):
        for key in ("fallback_target", "expected_dial_call_sid"):
            for value in (None, "", " \t", [], True, "\ud800", "private-" + "x" * 10000):
                validator = Mock(return_value=True)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.resolve(self.event(), validator, **{key: value})
                validator.assert_not_called()
        for value in (None, 0, 1, "false", []):
            validator = Mock(return_value=True)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(self.event(), validator, fallback_used=value)
            validator.assert_not_called()

    def test_wrong_or_missing_identity_never_reaches_policy(self):
        for name in ("AccountSid", "CallSid", "DialCallSid"):
            for value in (None, "other-leg", "synthetic-child "):
                with self.subTest(name=name, value=value), patch(
                    "zentomic.dial_event.resolve_dial_result",
                ) as policy:
                    validator = Mock(return_value=True)
                    with self.assertRaises(ValueError):
                        self.resolve(self.event(**{name: value}), validator)
                    validator.assert_called_once()
                    policy.assert_not_called()

    def test_failed_authentication_never_reaches_policy(self):
        for validator in (Mock(return_value=False), Mock(return_value=1),
                          Mock(side_effect=RuntimeError("synthetic-private-error"))):
            with patch("zentomic.dial_event.resolve_dial_result") as policy:
                with self.assertRaises(ValueError) as caught:
                    self.resolve(self.event(), validator)
                self.assertNotIn("synthetic-private-error", str(caught.exception))
                policy.assert_not_called()

    def test_call_status_cannot_replace_invalid_dial_status_even_after_fallback(self):
        for status in (None, "", "answered", "in-progress", "BUSY"):
            for used in (False, True):
                with self.subTest(status=status, used=used), self.assertRaises(ValueError):
                    self.resolve(self.event(DialCallStatus=status, CallStatus="completed"),
                                 Mock(return_value=True), fallback_used=used)

    def test_validator_cannot_rewrite_outcome_or_repair_child_identity(self):
        def mutate(url, fields, signature):
            fields.update(DialCallStatus="completed", DialCallSid="synthetic-child")
            return True

        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    event = self.event(version, encoded)
                    original = copy.deepcopy(event)
                    self.assertEqual(self.resolve(event, mutate), DialDecision("fallback", "demo-reception"))
                    self.assertEqual(event, original)
                    with self.assertRaises(ValueError):
                        self.resolve(self.event(version, encoded, DialCallSid="previous-child"), mutate)

    def test_invalid_transport_fails_before_validator(self):
        event = self.event()
        event["requestContext"]["http"]["method"] = "GET"
        validator = Mock(return_value=True)
        with self.assertRaises(ValueError):
            self.resolve(event, validator)
        validator.assert_not_called()

    def test_identifiers_and_targets_are_not_normalized(self):
        child = " synthetic-child-\u00e9 "
        decision = self.resolve(self.event(DialCallSid=child), Mock(return_value=True),
                                expected_dial_call_sid=child, fallback_target=" reception-\u00e9 ")
        self.assertEqual(decision, DialDecision("fallback", " reception-\u00e9 "))


class ClaimedDialEventTests(unittest.TestCase):
    event = DialEventTests.event

    def resolve_claimed(self, event, claimer, validator=None, **overrides):
        config = {
            "public_url": "https://example.com/voice/dial-result",
            "validator": validator or Mock(return_value=True),
            "expected_account_sid": "synthetic-account",
            "expected_call_sid": "synthetic-parent",
            "expected_dial_call_sid": "synthetic-child",
            "workspace_id": "synthetic-workspace",
            "step_id": "dial-attempt-1",
            "claimer": claimer,
            "fallback_target": "demo-reception",
            "fallback_used": False,
        }
        config.update(overrides)
        return resolve_claimed_dial_result_event(event, **config)

    def test_authenticated_child_result_is_claimed_once_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    store = InMemoryCallbackClaimStore()
                    event = self.event(
                        version, encoded, workspace_id="untrusted",
                        step_id="untrusted", fallback_target="untrusted",
                    )
                    original = copy.deepcopy(event)
                    self.assertEqual(
                        self.resolve_claimed(event, store),
                        DialDecision("fallback", "demo-reception"),
                    )
                    self.assertEqual(event, original)
                    with self.assertRaisesRegex(
                        CallbackReplayError, "^callback step already claimed$",
                    ), patch(
                        "zentomic.dial_event._resolve_validated_dial_result",
                    ) as policy:
                        self.resolve_claimed(event, store)
                    policy.assert_not_called()
                    self.assertFalse(store.claim(
                        "synthetic-workspace", "synthetic-parent", "dial-attempt-1",
                    ))
                    self.assertTrue(store.claim(
                        "untrusted", "synthetic-parent", "untrusted",
                    ))

    def test_rejected_or_stale_child_callback_does_not_consume_the_step(self):
        cases = (
            ({}, Mock(return_value=False)),
            ({"AccountSid": "other-account"}, Mock(return_value=True)),
            ({"CallSid": "other-parent"}, Mock(return_value=True)),
            ({"DialCallSid": "previous-child"}, Mock(return_value=True)),
        )
        for fields, validator in cases:
            store = InMemoryCallbackClaimStore()
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.resolve_claimed(self.event(**fields), store, validator)
            self.assertTrue(store.claim(
                "synthetic-workspace", "synthetic-parent", "dial-attempt-1",
            ))

    def test_invalid_trusted_state_does_not_authenticate_or_claim(self):
        for overrides in (
            {"expected_dial_call_sid": ""},
            {"fallback_target": ""},
            {"fallback_used": "false"},
            {"workspace_id": ""},
            {"step_id": " "},
            {"claimer": object()},
        ):
            validator = Mock(return_value=True)
            claimer = Mock()
            call_overrides = dict(overrides)
            supplied_claimer = call_overrides.pop("claimer", claimer)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve_claimed(
                    self.event(), supplied_claimer, validator, **call_overrides,
                )
            validator.assert_not_called()
            claimer.claim.assert_not_called()

    def test_claim_cannot_trigger_revalidation_or_replace_trusted_fallback(self):
        class SingleValidationTarget(str):
            checks = 0

            def encode(self, *args, **kwargs):
                self.checks += 1
                if self.checks > 1:
                    raise AssertionError("fallback was validated after the claim")
                return super().encode(*args, **kwargs)

        target = SingleValidationTarget("demo-reception")
        claimer = Mock()
        claimer.claim.return_value = True
        self.assertEqual(
            self.resolve_claimed(
                self.event(), claimer, fallback_target=target,
            ),
            DialDecision("fallback", target),
        )
        self.assertEqual(target.checks, 1)
        claimer.claim.assert_called_once_with(
            "synthetic-workspace", "synthetic-parent", "dial-attempt-1",
        )


if __name__ == "__main__":
    unittest.main()
