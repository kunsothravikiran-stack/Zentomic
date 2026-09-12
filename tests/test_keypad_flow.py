"""Offline authenticated keypad-to-TwiML contracts with test-owned state.

The validator is noncryptographic and attempt persistence is sequential only.
No durable step claims, replay protection, authorization, or live calls exist
in this harness. Repeated invocations below test budget policy, not deduplication.
"""

import copy
import unittest
from unittest.mock import Mock, patch
from xml.etree.ElementTree import fromstring

from tests.test_voice_flow import ACCOUNT, CALL, SIGNATURE, callback
from zentomic.gather_event import resolve_gather_event
from zentomic.response import twiml_response
from zentomic.twiml import render_dial, render_dtmf_gather, render_hangup


URL = "https://example.invalid/voice/menu"
DESTINATIONS = {"demo-support": ("+12025550101",),
                "demo-reception": ("+12025550102",)}


class KeypadFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.enterContext(patch("socket.socket", side_effect=AssertionError("Network forbidden")))

    def collect(self, fields, version, encoded, state, routes, validator):
        event = callback(fields, version, encoded)
        original = copy.deepcopy(event)
        decision = resolve_gather_event(
            event, routes, public_url=URL, validator=validator,
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            fallback_target="demo-reception", attempts=state["attempts"],
            max_attempts=3, hangup_digit="9",
        )
        self.assertEqual(event, original)
        validator.assert_called_with(
            URL, {"AccountSid": ACCOUNT, "CallSid": CALL, **fields}, SIGNATURE,
        )
        # This assignment is not an atomic database write or a step claim.
        state["attempts"] = decision.attempts
        if decision.action == "retry":
            xml = render_dtmf_gather("Please select a department.", action_path="/voice/menu")
        elif decision.action == "hangup":
            xml = render_hangup()
        else:
            xml = render_dial(DESTINATIONS[decision.target], action_path="/voice/dial-result")
        response = twiml_response(xml)
        self.assertEqual(response["statusCode"], 200)
        self.assertIs(response["isBase64Encoded"], False)
        self.assertEqual(response["headers"], {
            "Content-Type": "application/xml; charset=utf-8", "Cache-Control": "no-store",
        })
        return decision, fromstring(response["body"])

    def test_silence_invalid_input_and_exhaustion_cannot_reopen_collection(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    state, routes, validator = {"attempts": 0}, {"1": "demo-support"}, Mock(return_value=True)
                    for index, fields in enumerate(({}, {"Digits": "11"}, {"Digits": "8"}), 1):
                        decision, root = self.collect(fields, version, encoded, state, routes, validator)
                        self.assertEqual(state["attempts"], index)
                        self.assertEqual(decision.action, "retry" if index < 3 else "fallback")
                        self.assertEqual([node.tag for node in root], ["Gather" if index < 3 else "Dial"])
                        if index < 3:
                            self.assertEqual(root[0].get("action"), "/voice/menu")
                            self.assertEqual(root[0].get("actionOnEmptyResult"), "true")
                    for digit in ("1", "9"):
                        fields = dict(Digits=digit, attempts="0", max_attempts="999",
                                      target="demo-support", fallback_target="demo-support")
                        decision, root = self.collect(fields, version, encoded, state, routes, validator)
                        self.assertEqual((decision.action, state["attempts"]), ("fallback", 3))
                        self.assertEqual([node.tag for node in root], ["Dial"])
                        self.assertEqual([node.text for node in root.findall("Dial/Number")],
                                         list(DESTINATIONS["demo-reception"]))
                    self.assertEqual(validator.call_count, 5)

    def test_removed_route_after_retry_cannot_be_restored_by_signed_fields(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                with self.subTest(version=version, encoded=encoded):
                    state, routes, validator = {"attempts": 0}, {"1": "demo-support"}, Mock(return_value=True)
                    self.collect({}, version, encoded, state, routes, validator)
                    routes.clear()  # Trusted configuration changed between invocations.
                    fields = {"Digits": "1", "target": "demo-support", "routes": "1=demo-support"}
                    for action, count in (("retry", 2), ("fallback", 3)):
                        decision, root = self.collect(fields, version, encoded, state, routes, validator)
                        self.assertEqual((decision.action, state["attempts"]), (action, count))
                        self.assertNotIn(DESTINATIONS["demo-support"][0],
                                         [node.text for node in root.findall("Dial/Number")])
                    self.assertEqual([node.text for node in root.findall("Dial/Number")],
                                     list(DESTINATIONS["demo-reception"]))

    def test_rejected_callbacks_preserve_last_attempt_for_route_or_hangup(self):
        for version in ("1.0", "2.0"):
            for encoded in (False, True):
                for digit, action, verb in (("1", "route", "Dial"), ("9", "hangup", "Hangup")):
                    with self.subTest(version=version, encoded=encoded, digit=digit):
                        state, routes, validator = {"attempts": 0}, {"1": "demo-support"}, Mock(return_value=True)
                        for _ in range(2):
                            self.collect({}, version, encoded, state, routes, validator)
                        for extra, valid in (({}, False), ({"AccountSid": "other"}, True),
                                             ({"CallSid": "other"}, True)):
                            validator.return_value = valid
                            with patch("zentomic.gather_event.resolve_gather") as policy:
                                with self.assertRaises(ValueError):
                                    self.collect({"Digits": digit, **extra}, version, encoded,
                                                 state, routes, validator)
                                policy.assert_not_called()
                            self.assertEqual(state, {"attempts": 2})
                        validator.return_value = True
                        decision, root = self.collect({"Digits": digit}, version, encoded,
                                                      state, routes, validator)
                        self.assertEqual((decision.action, state["attempts"]), (action, 3))
                        self.assertEqual([node.tag for node in root], [verb])
                        self.assertEqual([node.text for node in root.findall("Dial/Number")],
                                         list(DESTINATIONS["demo-support"]) if digit == "1" else [])


if __name__ == "__main__":
    unittest.main()
