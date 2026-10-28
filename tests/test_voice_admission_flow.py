"""Offline authenticated redirect admission, not a production adapter.

State is test-owned and sequential. No durable writes, deduplication, provider
cryptography, concurrent callbacks, operation timeouts, or live calls are tested.
"""

import unittest
from unittest.mock import Mock, patch
from xml.etree.ElementTree import fromstring

from tests.test_voice_flow import ACCOUNT, CALL, callback
from zentomic.authentication import validate_call_event
from zentomic.budget import resolve_voice_step_budget
from zentomic.call_status import is_terminal_call_status
from zentomic.response import twiml_response
from zentomic.twiml import render_hangup, render_redirect


class VoiceAdmissionFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.validator = Mock(return_value=True)
        self.clock = Mock(return_value=100)
        self.state = dict(status="in-progress", deadline_ms=1_000, transitions=0)

    def transition(self, fields, path, version="2.0", encoded=False):
        """Compose existing helpers using trusted, test-owned state and paths.

        Authentication precedes state admission. Real adapters additionally need
        step binding, deduplication, and atomic count/next-step persistence before
        returning work. This sequential assignment does not implement those.
        """
        validate_call_event(
            callback(fields, version, encoded),
            public_url="https://example.invalid/voice/next",
            validator=self.validator, expected_account_sid=ACCOUNT,
            expected_call_sid=CALL,
        )
        if is_terminal_call_status(self.state["status"]):
            return twiml_response(render_hangup())
        decision = resolve_voice_step_budget(
            deadline_ms=self.state["deadline_ms"], now_ms=self.clock(),
            transitions=self.state["transitions"], max_transitions=2,
            minimum_remaining_ms=100,
        )
        self.state["transitions"] = decision.transitions
        if decision.action == "hangup":
            self.state["status"] = "completed"
            return twiml_response(render_hangup())
        return twiml_response(render_redirect(action_path=path))

    def assert_response(self, response, verb, path=None):
        self.assertEqual(response["statusCode"], 200)
        self.assertFalse(response["isBase64Encoded"])
        root = fromstring(response["body"])
        self.assertEqual([node.tag for node in root], [verb])
        if path is not None:
            self.assertEqual(root[0].text, path)
            self.assertEqual(root[0].attrib, {"method": "POST"})

    def test_distinct_steps_share_count_and_last_admission_still_redirects(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    self.state.update(status="in-progress", transitions=0)
                    for path, count in (("/voice/speech", 1), ("/voice/confirm", 2)):
                        self.assert_response(
                            self.transition({}, path, version, encoded), "Redirect", path,
                        )
                        self.assertEqual(self.state["transitions"], count)
                    self.assert_response(
                        self.transition({}, "/voice/dial", version, encoded), "Hangup",
                    )
                    self.assertEqual(self.state["transitions"], 2)
                    self.assertEqual(self.state["deadline_ms"], 1_000)

    def test_fresh_clock_denial_does_not_spend_the_remaining_transition(self):
        self.clock.side_effect = [900, 901]
        self.assert_response(self.transition({}, "/voice/confirm"), "Redirect", "/voice/confirm")
        self.assert_response(self.transition({}, "/voice/dial"), "Hangup")
        self.assertEqual(self.state["transitions"], 1)
        self.assertEqual(self.clock.call_count, 2)

    def test_caller_budget_and_path_fields_cannot_reopen_or_redirect_work(self):
        fields = dict(deadline_ms="999999999", now_ms="0", transitions="0",
                      max_transitions="999999999", minimum_remaining_ms="0",
                      action_path="https://example.invalid/untrusted", status="in-progress")
        self.assert_response(
            self.transition(fields, "/voice/confirm"), "Redirect", "/voice/confirm",
        )
        self.state["transitions"] = 2
        self.assert_response(self.transition(fields, "/voice/dial"), "Hangup")
        self.assertEqual(self.state["transitions"], 2)
        self.assertEqual(self.state["deadline_ms"], 1_000)

    def test_failed_authentication_never_reads_clock_or_consumes_admission(self):
        for fields, valid in (({}, False), ({"CallSid": "other-call"}, True),
                              ({"AccountSid": "other-account"}, True)):
            with self.subTest(fields=fields, valid=valid):
                self.validator.return_value = valid
                before = dict(self.state)
                with self.assertRaises(ValueError):
                    self.transition(fields, "/voice/confirm")
                self.assertEqual(self.state, before)
        self.clock.assert_not_called()

    def test_terminal_state_blocks_admission_even_when_both_budgets_have_room(self):
        for status in ("completed", "busy", "failed", "no-answer", "canceled"):
            with self.subTest(status=status):
                self.state["status"] = status
                before = dict(self.state)
                self.assert_response(self.transition({}, "/voice/confirm"), "Hangup")
                self.assertEqual(self.state, before)
        self.clock.assert_not_called()
        self.assertEqual(self.validator.call_count, 5)

    def test_terminal_denial_does_not_reopen_after_clock_rollback(self):
        self.clock.return_value = 1_000
        self.assert_response(self.transition({}, "/voice/confirm"), "Hangup")
        self.clock.return_value = 100
        self.assert_response(self.transition({}, "/voice/confirm"), "Hangup")
        self.assertEqual(self.state["transitions"], 0)
        self.assertEqual(self.clock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
