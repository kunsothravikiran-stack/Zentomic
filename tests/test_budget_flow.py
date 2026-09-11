"""Offline deadline/classifier integration example, not a production adapter.

The signature validator and classifier are mocks. Test-owned deadlines do not
provide durable session state, cancellation, workspace isolation, or replay
protection. No provider timeout or live hangup is exercised here.
"""

import unittest
from unittest.mock import Mock, patch
from xml.etree.ElementTree import fromstring

from tests.test_voice_flow import ACCOUNT, CALL, FALLBACK, ROUTES, callback
from zentomic.authentication import validate_call_event
from zentomic.budget import resolve_call_budget, resolve_operation_timeout
from zentomic.classification import parse_intent_response
from zentomic.speech import resolve_speech_gather
from zentomic.twiml import render_hangup


class BudgetFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.validator = Mock(return_value=True)
        self.classifier = Mock(return_value='{"intent":"support"}')
        self.clock = Mock(return_value=1_000)
        self.invocation_remaining = Mock(return_value=60_000)
        self.deadline_ms = 5_000

    def collect(self, fields, version="2.0", encoded=False):
        """Show authentication, fresh call/runtime budgets, and late-result gating.

        Return (pending label, terminal XML). Nonclassifiable speech and invalid
        model output return (None, None); retry/fallback orchestration is outside
        this example. timeout_ms is a mock interface, not an SDK keyword.
        """
        fields = validate_call_event(
            callback(fields, version, encoded),
            public_url="https://example.invalid/voice/speech-result",
            validator=self.validator, expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
        )
        timeout_ms = resolve_operation_timeout(
            deadline_ms=self.deadline_ms, now_ms=self.clock(),
            maximum_ms=2_000, minimum_ms=500, reserve_ms=250,
            granularity_ms=100,
            invocation_remaining_ms=self.invocation_remaining(),
        )
        if timeout_ms is None:
            return None, render_hangup()
        decision = resolve_speech_gather(
            fields.get("SpeechResult"), fallback_target=FALLBACK, attempts=0,
        )
        if decision.action != "classify":
            return None, None
        output = self.classifier(decision.transcript, timeout_ms=timeout_ms)
        # A dependency can return late despite its requested timeout. Never
        # reuse the admission timestamp to authorize a subsequent step.
        if resolve_call_budget(
            deadline_ms=self.deadline_ms, now_ms=self.clock(),
        ).action == "hangup":
            return None, render_hangup()
        return parse_intent_response(output, allowed_intents=ROUTES.keys()), None

    def assert_hangup(self, result):
        pending, xml = result
        self.assertIsNone(pending)
        self.assertEqual([node.tag for node in fromstring(xml)], ["Hangup"])

    def test_timeout_reaches_classifier_after_authentication_in_all_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.classifier.reset_mock()
                    self.clock.side_effect = [3_501, 4_999]
                    result = self.collect({"SpeechResult": "support"}, version, encoded)
                    self.assertEqual(result, ("support", None))
                    # 1499 remaining - 250 reserve, rounded down to 100 ms.
                    self.classifier.assert_called_once_with("support", timeout_ms=1_200)

    def test_insufficient_budget_never_starts_classification(self):
        for now in (4_251, 4_999, 5_000, 5_001):
            with self.subTest(now=now):
                self.clock.return_value = now
                self.assert_hangup(self.collect({"SpeechResult": "support"}))
        self.classifier.assert_not_called()

    def test_invocation_budget_caps_timeout_in_all_proxy_formats(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for remaining, expected in ((1_750, 1_500), (1_000, 700), (750, 500)):
                    with self.subTest(version=version, encoded=encoded, remaining=remaining):
                        self.classifier.reset_mock()
                        self.invocation_remaining.return_value = remaining
                        self.assertEqual(
                            self.collect({"SpeechResult": "support"}, version, encoded),
                            ("support", None),
                        )
                        self.classifier.assert_called_once_with("support", timeout_ms=expected)

    def test_short_invocation_does_not_start_or_parse_classifier_work(self):
        for remaining in (0, 250, 749):
            with self.subTest(remaining=remaining):
                self.invocation_remaining.return_value = remaining
                with patch(__name__ + ".parse_intent_response") as parse:
                    self.assert_hangup(self.collect({"SpeechResult": "support"}))
                    parse.assert_not_called()
        self.classifier.assert_not_called()

    def test_invocation_budget_is_refreshed_and_not_taken_from_callback(self):
        fields = {"SpeechResult": "support", "invocation_remaining_ms": "999999999"}
        self.invocation_remaining.side_effect = [2_250, 750, 749]
        self.assertEqual(self.collect(fields), ("support", None))
        self.assertEqual(self.collect(fields), ("support", None))
        self.assert_hangup(self.collect(fields))
        self.assertEqual(self.invocation_remaining.call_count, 3)
        self.assertEqual([call.kwargs["timeout_ms"] for call in self.classifier.call_args_list],
                         [2_000, 500])
        self.assertEqual(self.deadline_ms, 5_000)

    def test_larger_new_invocation_does_not_extend_call_deadline(self):
        self.invocation_remaining.side_effect = [750, 60_000]
        self.clock.side_effect = [1_000, 2_000, 5_000]
        self.assertEqual(self.collect({"SpeechResult": "support"}), ("support", None))
        self.assert_hangup(self.collect({"SpeechResult": "support"}))
        self.classifier.assert_called_once_with("support", timeout_ms=500)

    def test_late_results_are_discarded_before_parsing(self):
        for finished in (5_000, 5_001):
            with self.subTest(finished=finished):
                self.classifier.reset_mock()
                self.clock.side_effect = [4_250, finished]
                with patch(__name__ + ".parse_intent_response") as parse:
                    self.assert_hangup(self.collect({"SpeechResult": "support"}))
                    parse.assert_not_called()
                self.classifier.assert_called_once_with("support", timeout_ms=500)

    def test_callbacks_share_deadline_and_cannot_supply_timing_configuration(self):
        fields = {"SpeechResult": "support", "deadline_ms": "999999999",
                  "now_ms": "0", "timeout_ms": "999999999", "reserve_ms": "0"}
        self.clock.side_effect = [1_000, 2_000, 4_250, 4_999, 5_000]
        self.assertEqual(self.collect(fields), ("support", None))
        self.assertEqual(self.collect(fields), ("support", None))
        self.assert_hangup(self.collect(fields))
        self.assertEqual([call.kwargs["timeout_ms"] for call in self.classifier.call_args_list],
                         [2_000, 500])
        self.assertEqual(self.deadline_ms, 5_000)

    def test_rejected_callbacks_do_not_read_clock_or_invoke_classifier(self):
        for overrides, valid in (({}, False), ({"AccountSid": "other-account"}, True),
                                 ({"CallSid": "other-call"}, True)):
            with self.subTest(overrides=overrides, valid=valid):
                self.validator.return_value = valid
                with self.assertRaises(ValueError):
                    self.collect({"SpeechResult": "support", **overrides})
        self.clock.assert_not_called()
        self.invocation_remaining.assert_not_called()
        self.classifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
