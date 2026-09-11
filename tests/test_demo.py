"""Exercise the runnable demo using synthetic input and guarded execution."""

import io
import json
import unittest
from contextlib import redirect_stdout
from xml.etree.ElementTree import fromstring

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_keypad


class KeypadDemoTests(unittest.TestCase):
    def test_default_cli_demonstrates_retry_then_route(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["mode"], "offline-demo")
        steps = result["steps"]
        self.assertEqual([step["action"] for step in steps], ["gather", "retry", "route"])
        self.assertEqual([step["attempts"] for step in steps], [0, 1, 2])
        self.assertEqual(steps[-1]["target"], "demo-sales")
        for step in steps[:-1]:
            gather = fromstring(step["twiml"]).find("Gather")
            self.assertEqual(gather.get("action"), "/voice/menu")
            self.assertEqual(gather.get("actionOnEmptyResult"), "true")
        self.assertNotIn("twiml", steps[-1])

    def test_support_cli_is_offline(self):
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(["2"]), 0)
        self.assertEqual(json.loads(output.getvalue())["steps"][-1], {
            "action": "route", "attempts": 1, "target": "demo-support",
        })

    def test_silence_and_invalid_input_have_a_finite_fallback(self):
        for inputs in ([], [""], ["8", "8", "8", "1"]):
            with self.subTest(inputs=inputs), offline_guard():
                steps = simulate_keypad(inputs)
            self.assertEqual([step["action"] for step in steps],
                             ["gather", "retry", "retry", "fallback"])
            self.assertEqual(steps[-1], {
                "action": "fallback", "attempts": 3, "target": "demo-reception",
            })

    def test_last_attempt_can_still_route_or_hang_up(self):
        for digit, action in (("1", "route"), ("9", "hangup")):
            with self.subTest(digit=digit), offline_guard():
                steps = simulate_keypad(["8", "", digit])
            self.assertEqual((steps[-1]["action"], steps[-1]["attempts"]), (action, 3))

    def test_explicit_exit_emits_hangup_without_a_target(self):
        with offline_guard():
            steps = simulate_keypad(["9"])
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[-1]["action"], "hangup")
        self.assertNotIn("target", steps[-1])
        self.assertEqual([node.tag for node in fromstring(steps[-1]["twiml"])], ["Hangup"])

    def test_does_not_consume_inputs_after_terminal_decision(self):
        for digit in ("1", "2", "9"):
            def inputs():
                yield digit
                raise AssertionError("demo consumed input after terminal decision")

            with self.subTest(digit=digit), offline_guard():
                self.assertEqual(len(simulate_keypad(inputs())), 2)

    def test_unrecognized_input_is_not_reflected_or_normalized(self):
        for value in ("synthetic-private-text", " 1", "１", "12"):
            with self.subTest(value=value), offline_guard():
                steps = simulate_keypad([value, "2"])
                self.assertEqual(steps, simulate_keypad(["8", "2"]))
            self.assertEqual(steps[1]["action"], "retry")
            self.assertEqual(steps[-1]["target"], "demo-support")


if __name__ == "__main__":
    unittest.main()
