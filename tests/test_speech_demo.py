"""Offline speech admission through separate intent confirmation."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from tests.__main__ import offline_guard
from zentomic.demo import main, simulate_classification, simulate_speech


def unused_inputs():
    raise AssertionError("consumed inputs after this stage ended")
    yield


class SpeechDemoTests(unittest.TestCase):
    def test_admitted_speech_requires_separate_confirmation(self):
        response = '{"intent":"support"}'
        for digits in ([], ["1"], ["2"], ["9"], ["8", "", "1"]):
            with self.subTest(digits=digits), offline_guard():
                steps = simulate_speech(["synthetic-private-text"], response, digits)
                expected = simulate_classification(response, digits)
            self.assertEqual([step["action"] for step in steps[:2]], ["gather", "classify"])
            self.assertEqual([step["attempts"] for step in steps[:2]], [0, 1])
            self.assertEqual([step["stage"] for step in steps[:2]], ["speech"] * 2)
            self.assertEqual(steps[2:], [dict(step, stage="confirmation") for step in expected])
            gather = fromstring(steps[0]["twiml"]).find("Gather")
            self.assertEqual(gather.get("input"), "speech")
            self.assertEqual(gather.get("action"), "/voice/speech")
            self.assertNotIn("synthetic-private", json.dumps(steps))
            self.assertNotIn("transcript", json.dumps(steps))

    def test_rejected_speech_never_reaches_classifier_or_confirmation(self):
        for speech in ([], [None], [""], [" "], ["x" * 2001], ["\ud800"], [42]):
            with self.subTest(speech=speech), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                steps = simulate_speech(speech, "synthetic-private-response", unused_inputs())
            classify.assert_not_called()
            self.assertEqual([step["action"] for step in steps],
                             ["gather", "retry", "retry", "fallback"])
            self.assertEqual(steps[-1], {
                "stage": "speech", "action": "fallback", "attempts": 3,
                "target": "demo-reception",
            })
            self.assertNotIn("synthetic-private", json.dumps(steps))

    def test_speech_consumption_stops_on_admission_or_exhaustion(self):
        for prefix in (["synthetic"], ["", "", "synthetic"], ["", "", ""]):
            def speeches():
                yield from prefix
                yield from unused_inputs()

            with self.subTest(prefix=prefix), offline_guard():
                steps = simulate_speech(speeches(), "invalid", unused_inputs())
            self.assertEqual(steps[-1]["action"], "fallback")
            self.assertEqual(steps[-1]["stage"],
                             "confirmation" if "synthetic" in prefix else "speech")

    def test_cli_repeated_speech_and_existing_classification_are_composed(self):
        output = io.StringIO()
        args = ["--speech", "", "--speech", "synthetic-private",
                "--classifier-response", '{"intent":"sales"}', "1"]
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(args), 0)
        steps = json.loads(output.getvalue())["steps"]
        self.assertEqual([step["action"] for step in steps],
                         ["gather", "retry", "classify", "gather", "route"])
        self.assertEqual(steps[-1]["target"], "demo-sales")
        self.assertNotIn("synthetic-private", output.getvalue())

    def test_last_speech_attempt_can_use_full_separate_confirmation_budget(self):
        def speeches():
            yield from ["", "", "synthetic"]
            yield from unused_inputs()

        def digits():
            yield from ["8", "", "1"]
            yield from unused_inputs()

        with offline_guard():
            steps = simulate_speech(speeches(), '{"intent":"sales"}', digits())
        self.assertEqual([step["attempts"] for step in steps], [0, 1, 2, 3] * 2)
        self.assertEqual(steps[-1], {
            "stage": "confirmation", "action": "route", "attempts": 3,
            "target": "demo-sales",
        })

    def test_cli_speech_requires_classifier_fixture(self):
        for args in (["--speech", "synthetic"],
                     ["--speech", "synthetic", "--intent", "sales"]):
            with self.subTest(args=args), offline_guard(), redirect_stderr(io.StringIO()), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            simulate.assert_not_called()

    def test_empty_hints_preserve_default_output(self):
        expected = simulate_speech(["", "synthetic"], '{"intent":"sales"}', ["1"])
        for hints in (None, [], ()):
            with self.subTest(hints=hints):
                self.assertEqual(simulate_speech(
                    ["", "synthetic"], '{"intent":"sales"}', ["1"], hints=hints,
                ), expected)

    def test_vocabulary_is_only_in_speech_xml_and_never_selects_the_intent(self):
        phrases = ['sales & <help> "desk"', "café", "sales"]
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for hints in (phrases, tuple(phrases)):
            with self.subTest(hints=type(hints).__name__), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"], hints=hints,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                self.assertEqual({k: v for k, v in step.items() if k != "twiml"},
                                 {k: v for k, v in original.items() if k != "twiml"})
                if step["stage"] == "speech" and "twiml" in step:
                    root = fromstring(step["twiml"])
                    self.assertEqual([node.tag for node in root.iter()],
                                     ["Response", "Gather", "Say"])
                    self.assertEqual(root[0].get("hints"), ",".join(phrases))
                else:
                    self.assertEqual(step, original)
            self.assertEqual(steps[0]["twiml"], steps[1]["twiml"])

    def test_hints_are_snapshotted_once_for_all_retries(self):
        class ChangingHints(list):
            def __iter__(self):
                yield "sales"
                self[:] = ["support"]

        hints = ChangingHints(["sales"])
        steps = simulate_speech([], None, unused_inputs(), hints=hints)
        self.assertEqual(hints, ["support"])
        xml = [step["twiml"] for step in steps if "twiml" in step]
        self.assertEqual(len(xml), 3)
        self.assertTrue(all(fromstring(value)[0].get("hints") == "sales" for value in xml))

    def test_invalid_hints_fail_before_consuming_or_classifying_inputs(self):
        for hints in ([""], ["synthetic-private,phrase"], ["\ud800"], ["x" * 101],
                      ["x"] * 501, "sales", [42]):
            with self.subTest(hints=type(hints).__name__), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError):
                    simulate_speech(unused_inputs(), None, unused_inputs(), hints=hints)
            classify.assert_not_called()

    def test_cli_hints_preserve_order_and_require_explicit_confirmation(self):
        for digits, target in (([], "demo-reception"), (["1"], "demo-support")):
            output = io.StringIO()
            args = ["--speech-hint", "sales", "--speech-hint", "technical support",
                    "--speech-hint", "sales", "--speech", "synthetic-private",
                    "--classifier-response", '{"intent":"support"}', *digits]
            with offline_guard(), redirect_stdout(output):
                self.assertEqual(main(args), 0)
            steps = json.loads(output.getvalue())["steps"]
            self.assertEqual(fromstring(steps[0]["twiml"])[0].get("hints"),
                             "sales,technical support,sales")
            self.assertEqual(steps[-1]["target"], target)
            self.assertNotIn("synthetic-private", output.getvalue())

    def test_cli_hint_limits_and_private_errors_without_partial_json(self):
        base = ["--speech", "synthetic", "--classifier-response", '{"intent":"sales"}']
        output = io.StringIO()
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(base + ["--speech-hint", "é" * 100] * 500), 0)
        gather = fromstring(json.loads(output.getvalue())["steps"][0]["twiml"])[0]
        self.assertEqual(gather.get("hints"), ",".join(["é" * 100] * 500))
        for hint_args in (["--speech-hint", "synthetic-private,phrase"],
                          ["--speech-hint", "synthetic-private\nphrase"],
                          ["--speech-hint", ""], ["--speech-hint", "x" * 101],
                          ["--speech-hint", "synthetic-private"] * 501):
            output, errors = io.StringIO(), io.StringIO()
            with offline_guard(), redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(base + hint_args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech hints", errors.getvalue())
            self.assertNotIn("synthetic-private", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())

    def test_cli_hints_cannot_silently_configure_other_modes(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-hint", "synthetic-private", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-hint requires --speech", errors.getvalue())
            self.assertNotIn("synthetic-private", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
