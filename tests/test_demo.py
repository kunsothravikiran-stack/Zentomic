"""Exercise the runnable demo using synthetic input and guarded execution."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from xml.etree.ElementTree import fromstring

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_classification, simulate_confirmation, simulate_keypad


class KeypadDemoTests(unittest.TestCase):
    def test_all_modes_stop_consuming_at_the_retry_budget(self):
        for run in (simulate_keypad,
                    lambda digits: simulate_confirmation("sales", digits),
                    lambda digits: simulate_classification('{"intent":"sales"}', digits)):
            consumed = []

            def inputs():
                for digit in ("8", "", "8"):
                    consumed.append(digit)
                    yield digit
                raise AssertionError("consumed input beyond the retry budget")

            with self.subTest(run=run), offline_guard():
                steps = run(inputs())
            self.assertEqual(consumed, ["8", "", "8"])
            self.assertEqual(steps[-1], {
                "action": "fallback", "attempts": 3, "target": "demo-reception",
            })

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


class ConfirmationDemoTests(unittest.TestCase):
    def test_cli_requires_explicit_confirmation(self):
        for label in ("sales", "support"):
            for digits, action, target in ((["1"], "route", "demo-" + label),
                                            (["2"], "fallback", "demo-reception"),
                                            ([], "fallback", "demo-reception")):
                output = io.StringIO()
                with self.subTest(label=label, digits=digits), offline_guard(), \
                        redirect_stdout(output):
                    self.assertEqual(main(["--intent", label, *digits]), 0)
                steps = json.loads(output.getvalue())["steps"]
                self.assertEqual(steps[-1]["action"], action)
                self.assertEqual(steps[-1]["target"], target)
                self.assertNotIn("target", steps[0])
                gather = fromstring(steps[0]["twiml"]).find("Gather")
                self.assertEqual(gather.get("action"), "/voice/confirm")
                self.assertIn(label, gather.findtext("Say"))

    def test_bounded_retry_and_explicit_hangup(self):
        for digit, action in (("1", "route"), ("2", "fallback"), ("9", "hangup")):
            with self.subTest(digit=digit), offline_guard():
                steps = simulate_confirmation("sales", ["8", "", digit])
            self.assertEqual([step["action"] for step in steps],
                             ["gather", "retry", "retry", action])
            self.assertEqual([step["attempts"] for step in steps], [0, 1, 2, 3])
            if action == "hangup":
                self.assertNotIn("target", steps[-1])
                self.assertIsNotNone(fromstring(steps[-1]["twiml"]).find("Hangup"))
        with offline_guard():
            self.assertEqual(simulate_confirmation("sales", ["8"] * 3 + ["1"])[-1],
                             {"action": "fallback", "attempts": 3, "target": "demo-reception"})

    def test_stops_consuming_after_terminal_decision(self):
        for digit in ("1", "2", "9"):
            def inputs():
                yield digit
                raise AssertionError("consumed input after terminal decision")

            with self.subTest(digit=digit), offline_guard():
                self.assertEqual(len(simulate_confirmation("support", inputs())), 2)

    def test_unknown_label_is_rejected_without_reflection_or_consumption(self):
        def inputs():
            raise AssertionError("consumed input with invalid configuration")
            yield

        for label in ("synthetic-private", " sales", "SALES", None, []):
            with self.subTest(label=label), offline_guard():
                with self.assertRaisesRegex(ValueError, "^intent must be sales or support$"):
                    simulate_confirmation(label, inputs())

    def test_invalid_digits_are_not_reflected(self):
        with offline_guard():
            expected = simulate_confirmation("sales", ["8", "1"])
            for digit in ("synthetic-private", " 1", "１", "12"):
                with self.subTest(digit=digit):
                    self.assertEqual(simulate_confirmation("sales", [digit, "1"]), expected)


class ClassificationDemoTests(unittest.TestCase):
    def test_admitted_label_uses_existing_confirmation_policy(self):
        for label in ("sales", "support"):
            for digits in ([], ["1"], ["2"], ["9"], ["8", "", "1"], ["8"] * 4):
                with self.subTest(label=label, digits=digits), offline_guard():
                    self.assertEqual(
                        simulate_classification(json.dumps({"intent": label}), digits),
                        simulate_confirmation(label, digits),
                    )

    def test_rejected_response_falls_back_without_consuming_or_reflecting_input(self):
        def inputs():
            raise AssertionError("consumed confirmation input without a pending intent")
            yield

        for response in (None, "", "synthetic-private", "{}", "null", "[]",
                         '{"intent":"unknown"}', '{"intent":"Sales"}',
                         '{"intent":"sales","confirmed":true}',
                         '{"intent":"sales","intent":"support"}',
                         '{"intent":"sales"}' + " " * 4096):
            with self.subTest(response=response), offline_guard():
                self.assertEqual(simulate_classification(response, inputs()), [
                    {"action": "fallback", "attempts": 0, "target": "demo-reception"},
                ])

    def test_cli_routes_only_after_confirmation_and_rejects_mixed_modes(self):
        for response, digits, expected in (
            ('{"intent":"support"}', ["1"], "route"),
            ('{"intent":"support"}', [], "fallback"),
            ("synthetic-private", ["1"], "fallback"),
        ):
            output = io.StringIO()
            with self.subTest(response=response, digits=digits), offline_guard(), \
                    redirect_stdout(output):
                self.assertEqual(main(["--classifier-response", response, *digits]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["mode"], "offline-demo")
            self.assertEqual(result["steps"][-1]["action"], expected)
            self.assertNotIn(response, output.getvalue())
        with offline_guard(), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--intent", "sales", "--classifier-response", "{}"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
