"""Offline multi-step contract tests, not a deployable voice adapter.

The test owns session state and injects a fake signature validator. No SDK
cryptography, persistence, replay protection, or workspace loading is exercised.
"""

import base64
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode
from xml.etree.ElementTree import fromstring

from zentomic.authentication import validate_call_event
from zentomic.dial import resolve_dial_result
from zentomic.intent import resolve_intent_confirmation
from zentomic.twiml import render_dial, render_dtmf_gather, render_hangup


ACCOUNT = "synthetic-account"
CALL = "synthetic-call"
SIGNATURE = "A" * 27 + "="  # Shape only, not a real signature.
ROUTES = {"support": "team/Support"}
DESTINATIONS = {
    "team/Support": ("+12025550101", "+12025550102"),
    "team/Reception": ("+12025550103",),
}
FALLBACK = "team/Reception"


def callback(fields, version, encoded):
    body = urlencode({"AccountSid": ACCOUNT, "CallSid": CALL, **fields})
    event = {
        "version": version,
        "headers": {"Content-Type": "application/x-www-form-urlencoded",
                    "X-Twilio-Signature": SIGNATURE},
        "body": base64.b64encode(body.encode()).decode() if encoded else body,
        "isBase64Encoded": encoded,
    }
    if version == "2.0":
        event["requestContext"] = {"http": {"method": "POST"}}
    else:
        event["httpMethod"] = "POST"
    return event


class VoiceFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))
        self.validator = Mock(return_value=True)

    def authenticated(self, fields, version, encoded, path):
        return validate_call_event(
            callback(fields, version, encoded),
            public_url="https://example.invalid" + path,
            validator=self.validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
        )

    def confirmation(self, fields, version, encoded, attempts):
        fields = self.authenticated(fields, version, encoded, "/voice/confirm")
        return resolve_intent_confirmation(
            fields.get("Digits"), "support", ROUTES,
            fallback_target=FALLBACK, attempts=attempts,
        )

    def dial_result(self, fields, version, encoded, fallback_used=False):
        fields = self.authenticated(fields, version, encoded, "/voice/dial-result")
        return resolve_dial_result(
            fields.get("DialCallStatus"), fallback_target=FALLBACK,
            fallback_used=fallback_used,
        )

    def test_silence_confirmation_forwarding_and_completion(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    retry = self.confirmation({}, version, encoded, attempts=0)
                    self.assertEqual((retry.action, retry.target, retry.attempts), ("retry", None, 1))
                    menu = fromstring(render_dtmf_gather(
                        "Press 1 for Support, or 2 for reception.", action_path="/voice/confirm",
                    ))
                    self.assertEqual([child.tag for child in menu], ["Gather"])
                    confirmed = self.confirmation({"Digits": "1"}, version, encoded, retry.attempts)
                    self.assertEqual((confirmed.action, confirmed.attempts), ("route", 2))
                    dial = fromstring(render_dial(
                        DESTINATIONS[confirmed.target], action_path="/voice/dial-result", time_limit=600,
                    ))
                    self.assertEqual([node.text for node in dial.findall("Dial/Number")],
                                     list(DESTINATIONS["team/Support"]))
                    self.assertEqual(dial.find("Dial").attrib["timeLimit"], "600")
                    terminal = self.dial_result({"DialCallStatus": "completed"}, version, encoded)
                    self.assertEqual((terminal.action, terminal.target), ("hangup", None))
                    self.assertEqual([child.tag for child in fromstring(render_hangup())], ["Hangup"])

    def test_failed_forwarding_cannot_repeat_the_fallback(self):
        for status in ("busy", "no-answer", "failed"):
            with self.subTest(status=status):
                result = self.dial_result({"DialCallStatus": status}, "2.0", True)
                self.assertEqual((result.action, result.target), ("fallback", FALLBACK))
                xml = fromstring(render_dial(DESTINATIONS[result.target], action_path="/voice/dial-result"))
                self.assertEqual([node.text for node in xml.findall("Dial/Number")],
                                 list(DESTINATIONS[FALLBACK]))
                # The adapter must persist this flag before executing the fallback.
                terminal = self.dial_result({"DialCallStatus": status}, "2.0", True, fallback_used=True)
                self.assertEqual((terminal.action, terminal.target), ("hangup", None))

    def test_signed_extra_fields_do_not_replace_trusted_state(self):
        forged = {"Digits": "1", "attempts": "0", "max_attempts": "999",
                  "intent": "sales", "confirmed": "true", "target": "untrusted-target"}
        result = self.confirmation(forged, "1.0", False, attempts=3)
        self.assertEqual((result.action, result.target, result.attempts), ("fallback", FALLBACK, 3))
        terminal = self.dial_result(
            {"DialCallStatus": "failed", "fallback_used": "false"}, "2.0", False, fallback_used=True,
        )
        self.assertEqual((terminal.action, terminal.target), ("hangup", None))

    def test_rejected_callbacks_never_reach_confirmation_policy(self):
        for version in ("1.0", "2.0"):
            for overrides, valid in (({}, False), ({"AccountSid": "other-account"}, True),
                                     ({"CallSid": "other-call"}, True)):
                with self.subTest(version=version, overrides=overrides, valid=valid):
                    self.validator.return_value = valid
                    with patch(__name__ + ".resolve_intent_confirmation") as policy:
                        with self.assertRaises(ValueError):
                            self.confirmation({"Digits": "1", **overrides}, version, False, attempts=0)
                        policy.assert_not_called()

    def test_parent_call_status_cannot_substitute_for_dial_outcome(self):
        with self.assertRaises(ValueError):
            self.dial_result({"CallStatus": "completed"}, "1.0", False)
        result = self.dial_result(
            {"CallStatus": "in-progress", "DialCallStatus": "no-answer"}, "1.0", False,
        )
        self.assertEqual((result.action, result.target), ("fallback", FALLBACK))


if __name__ == "__main__":
    unittest.main()
