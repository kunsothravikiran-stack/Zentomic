"""Synthetic dial outcome decisions without telephony or persistence."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.dial import DialDecision, resolve_dial_result
from zentomic.twiml import render_hangup
from zentomic.webhook import parse_form_body


class DialResultTests(unittest.TestCase):
    def resolve(self, status, **kwargs):
        config = {"fallback_target": "reception"}
        config.update(kwargs)
        return resolve_dial_result(status, **config)

    def test_unsuccessful_attempts_select_fallback(self):
        for status in ("busy", "no-answer", "failed"):
            with self.subTest(status=status):
                self.assertEqual(self.resolve(status), DialDecision("fallback", "reception"))

    def test_completed_and_canceled_calls_do_not_redial(self):
        for status in ("completed", "canceled"):
            for used in (False, True):
                with self.subTest(status=status, used=used):
                    self.assertEqual(self.resolve(status, fallback_used=used), DialDecision("hangup", None))

    def test_failed_fallback_ends_instead_of_looping(self):
        for status in ("busy", "no-answer", "failed"):
            with self.subTest(status=status):
                self.assertEqual(self.resolve(status, fallback_used=True), DialDecision("hangup", None))

    def test_missing_unknown_and_wrong_callback_statuses_are_rejected(self):
        for status in (None, "", "answered", "queued", "initiated", "ringing", "in-progress",
                       "cancelled", "future-status", True, 1, [], {}, b"busy"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.resolve(status)

    def test_status_is_not_normalized(self):
        for status in ("BUSY", " busy", "busy ", "no_answer", "completed\n"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.resolve(status)

    def test_configuration_is_validated_even_for_terminal_results(self):
        for target in (None, "", " \t", True, 1, [], {}):
            for status in ("completed", "busy"):
                with self.subTest(target=target, status=status), self.assertRaises(ValueError):
                    self.resolve(status, fallback_target=target)

    def test_fallback_state_requires_a_real_boolean(self):
        for used in (None, 0, 1, "false", "true", [], {}):
            with self.subTest(used=used), self.assertRaises(ValueError):
                self.resolve("completed", fallback_used=used)

    def test_targets_remain_opaque_and_decisions_are_immutable(self):
        decision = self.resolve("busy", fallback_target=" reception ")
        self.assertEqual(decision.target, " reception ")
        with self.assertRaises(FrozenInstanceError):
            decision.target = "sales"

    def test_form_result_uses_dial_status_and_terminal_renderer(self):
        # Transport parsing is not authentication; this is only offline composition.
        fields = parse_form_body("DialCallStatus=no-answer&CallStatus=in-progress")
        first = self.resolve(fields["DialCallStatus"])
        self.assertEqual(first.action, "fallback")
        second = self.resolve(fields["DialCallStatus"], fallback_used=True)
        self.assertEqual(second.action, "hangup")
        self.assertEqual([node.tag for node in fromstring(render_hangup())], ["Hangup"])

    def test_errors_do_not_reflect_callback_contents(self):
        with self.assertRaises(ValueError) as caught:
            self.resolve("SyntheticPrivateStatus")
        self.assertNotIn("SyntheticPrivateStatus", str(caught.exception))

    def test_policy_is_deterministic_offline_and_silent(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ), patch("builtins.print") as output:
            self.assertEqual(self.resolve("busy"), self.resolve("busy"))
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
