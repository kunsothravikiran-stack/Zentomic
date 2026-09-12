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


if __name__ == "__main__":
    unittest.main()
