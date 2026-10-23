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

    def test_confidence_threshold_retries_explicit_low_scores(self):
        with offline_guard():
            steps = simulate_speech(
                ["synthetic-private-low", "synthetic-private-high"],
                '{"intent":"sales"}', ["1"],
                confidences=["0.799", "0.80"],
                minimum_confidence_percent=80,
            )
        self.assertEqual([step["action"] for step in steps],
                         ["gather", "retry", "classify", "gather", "route"])
        self.assertEqual(steps[-1]["target"], "demo-sales")
        self.assertNotIn("synthetic-private", json.dumps(steps))
        self.assertNotIn("0.799", json.dumps(steps))

    def test_missing_confidence_is_admitted_but_malformed_score_retries(self):
        for confidences, expected in ((None, "classify"), ([], "classify"),
                                      ([None], "classify"), (["private-score"], "retry")):
            with self.subTest(confidences=confidences), offline_guard():
                steps = simulate_speech(
                    ["synthetic"], '{"intent":"sales"}', [],
                    confidences=confidences, minimum_confidence_percent=80,
                )
            self.assertEqual(steps[1]["action"], expected)
            self.assertNotIn("private-score", json.dumps(steps))

    def test_invalid_confidence_policy_fails_before_other_inputs(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid policy must fail before hint iteration")

        for threshold in (-1, 101, True, 80.0, "80"):
            with self.subTest(threshold=repr(threshold)), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError):
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), confidences=unused_inputs(),
                        minimum_confidence_percent=threshold,
                    )
            classify.assert_not_called()

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

    def test_cli_confidence_threshold_is_aligned_and_private(self):
        output = io.StringIO()
        args = [
            "--speech-minimum-confidence", "80",
            "--speech", "synthetic-private-low", "--speech-confidence", "private-score",
            "--speech", "synthetic-private-high", "--speech-confidence", "0.80",
            "--classifier-response", '{"intent":"sales"}', "1",
        ]
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(args), 0)
        self.assertEqual(
            [step["action"] for step in json.loads(output.getvalue())["steps"]],
            ["gather", "retry", "classify", "gather", "route"],
        )
        self.assertNotIn("synthetic-private", output.getvalue())
        self.assertNotIn("private-score", output.getvalue())

    def test_cli_confidence_options_are_strict_and_mode_scoped(self):
        cases = (
            (["--speech-minimum-confidence", "80"],
             "--speech-minimum-confidence requires --speech"),
            (["--speech-confidence", "0.8", "--speech", "synthetic",
              "--classifier-response", '{"intent":"sales"}'],
             "--speech-confidence requires --speech-minimum-confidence"),
            (["--speech-minimum-confidence", "80", "--speech", "synthetic",
              "--speech-confidence", "0.8", "--speech", "synthetic",
              "--classifier-response", '{"intent":"sales"}'],
             "provide one --speech-confidence for each --speech"),
        )
        for args, message in cases:
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(args=args), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn(message, errors.getvalue())

        for value in ("-1", "00", "01", "101", "+1", "1.0", "private-threshold"):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-minimum-confidence", value, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(value=value), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech minimum confidence", errors.getvalue())
            self.assertNotIn("private-threshold", errors.getvalue())
            simulate.assert_not_called()

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

    def test_explicit_language_changes_only_speech_collection_xml(self):
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for language in ("en-US", "en-GB", "en-IN", "es-US", "es-ES", "hi-IN"):
            with self.subTest(language=language), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"],
                    language=language,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                if step["stage"] == "speech" and "twiml" in step:
                    actual = fromstring(step["twiml"])
                    expected = fromstring(original["twiml"])
                    self.assertEqual(actual[0].get("language"), language)
                    actual[0].set("language", "en-US")
                    self.assertEqual(actual.attrib, expected.attrib)
                    self.assertEqual(actual[0].attrib, expected[0].attrib)
                    self.assertEqual(actual[0][0].text, expected[0][0].text)
                else:
                    self.assertEqual(step, original)

    def test_invalid_language_fails_before_hints_inputs_or_classifier(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid language must fail before hint iteration")

        for language in (None, True, "auto", "private-locale", "en-us"):
            with self.subTest(language=language), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError) as error:
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), language=language,
                    )
            classify.assert_not_called()
            self.assertNotIn("private-locale", str(error.exception))

    def test_cli_explicit_language_is_reused_only_for_speech_retries(self):
        output = io.StringIO()
        args = ["--speech-language", "en-IN", "--speech", "", "--speech", "synthetic",
                "--classifier-response", '{"intent":"sales"}', "1"]
        with offline_guard(), redirect_stdout(output):
            self.assertEqual(main(args), 0)
        steps = json.loads(output.getvalue())["steps"]
        speech_xml = [fromstring(step["twiml"])[0] for step in steps
                      if step["stage"] == "speech" and "twiml" in step]
        self.assertEqual(len(speech_xml), 2)
        self.assertTrue(all(gather.get("language") == "en-IN" for gather in speech_xml))
        confirmation = [fromstring(step["twiml"])[0] for step in steps
                        if step["stage"] == "confirmation" and "twiml" in step]
        self.assertTrue(confirmation)
        self.assertTrue(all(gather.get("language") is None for gather in confirmation))
        self.assertEqual(steps[-1]["target"], "demo-sales")

    def test_cli_invalid_language_is_private_and_prints_no_partial_json(self):
        for language in ("auto", "private-locale", "en-us", "en-US "):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-language", language, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(language=language), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech language", errors.getvalue())
            self.assertNotIn("private-locale", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            simulate.assert_not_called()

    def test_cli_language_requires_speech_and_cannot_configure_other_modes(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-language", "en-IN", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-language requires --speech", errors.getvalue())

    def test_explicit_input_timeout_changes_only_speech_collection_xml(self):
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for input_timeout in (1, 60):
            with self.subTest(input_timeout=input_timeout), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"],
                    input_timeout=input_timeout,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                if step["stage"] == "speech" and "twiml" in step:
                    actual = fromstring(step["twiml"])
                    expected = fromstring(original["twiml"])
                    self.assertEqual(actual[0].get("timeout"), str(input_timeout))
                    actual[0].set("timeout", "5")
                    self.assertEqual(actual.attrib, expected.attrib)
                    self.assertEqual(actual[0].attrib, expected[0].attrib)
                    self.assertEqual(actual[0][0].text, expected[0][0].text)
                else:
                    self.assertEqual(step, original)

    def test_invalid_input_timeout_fails_before_hints_inputs_or_classifier(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid timeout must fail before hint iteration")

        for input_timeout in (None, True, 0, 61, "5", "auto", "private-timeout"):
            with self.subTest(input_timeout=input_timeout), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError) as error:
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), input_timeout=input_timeout,
                    )
            classify.assert_not_called()
            self.assertNotIn("private-timeout", str(error.exception))

    def test_cli_input_timeout_is_reused_only_for_speech_retries(self):
        for value in ("1", "60"):
            output = io.StringIO()
            args = ["--speech-input-timeout", value, "--speech", "",
                    "--speech", "synthetic", "--classifier-response",
                    '{"intent":"sales"}', "1"]
            with self.subTest(value=value), offline_guard(), redirect_stdout(output):
                self.assertEqual(main(args), 0)
            steps = json.loads(output.getvalue())["steps"]
            speech_xml = [fromstring(step["twiml"])[0] for step in steps
                          if step["stage"] == "speech" and "twiml" in step]
            self.assertEqual(len(speech_xml), 2)
            self.assertTrue(all(gather.get("timeout") == value for gather in speech_xml))
            confirmation = [fromstring(step["twiml"])[0] for step in steps
                            if step["stage"] == "confirmation" and "twiml" in step]
            self.assertTrue(confirmation)
            self.assertTrue(all(gather.get("timeout") == "5" for gather in confirmation))
            self.assertEqual(steps[-1]["target"], "demo-sales")

    def test_cli_invalid_input_timeout_is_private_and_prints_no_partial_json(self):
        for value in ("0", "00", "01", "61", "+1", "1.0", "auto",
                      "private-timeout", "9" * 10000):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-input-timeout", value, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(value=value[:20]), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech input timeout", errors.getvalue())
            self.assertNotIn("private-timeout", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            simulate.assert_not_called()

    def test_cli_input_timeout_requires_speech_and_cannot_configure_other_modes(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-input-timeout", "5", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-input-timeout requires --speech", errors.getvalue())

    def test_explicit_profanity_filter_changes_only_speech_collection_xml(self):
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for value, rendered in ((True, "true"), (False, "false")):
            with self.subTest(value=value), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"],
                    profanity_filter=value,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                if step["stage"] == "speech" and "twiml" in step:
                    actual = fromstring(step["twiml"])
                    expected = fromstring(original["twiml"])
                    self.assertEqual(actual[0].get("profanityFilter"), rendered)
                    del actual[0].attrib["profanityFilter"]
                    self.assertEqual(actual[0].attrib, expected[0].attrib)
                    self.assertEqual(actual[0][0].text, expected[0][0].text)
                else:
                    self.assertEqual(step, original)

    def test_invalid_profanity_filter_fails_before_hints_inputs_or_classifier(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid filter must fail before hint iteration")

        for value in (0, 1, "true", "false", "private-filter", [], {}):
            with self.subTest(value=repr(value)[:30]), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError) as error:
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), profanity_filter=value,
                    )
            classify.assert_not_called()
            self.assertNotIn("private-filter", str(error.exception))

    def test_cli_profanity_filter_is_reused_only_for_speech_retries(self):
        for value in ("true", "false"):
            output = io.StringIO()
            args = ["--speech-profanity-filter", value, "--speech", "",
                    "--speech", "synthetic", "--classifier-response",
                    '{"intent":"sales"}', "1"]
            with self.subTest(value=value), offline_guard(), redirect_stdout(output):
                self.assertEqual(main(args), 0)
            steps = json.loads(output.getvalue())["steps"]
            speech_xml = [fromstring(step["twiml"])[0] for step in steps
                          if step["stage"] == "speech" and "twiml" in step]
            self.assertEqual(len(speech_xml), 2)
            self.assertTrue(all(gather.get("profanityFilter") == value
                                for gather in speech_xml))
            confirmation = [fromstring(step["twiml"])[0] for step in steps
                            if step["stage"] == "confirmation" and "twiml" in step]
            self.assertTrue(confirmation)
            self.assertTrue(all(gather.get("profanityFilter") is None
                                for gather in confirmation))
            self.assertEqual(steps[-1]["target"], "demo-sales")

    def test_cli_profanity_filter_requires_speech_and_exact_lowercase_value(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-profanity-filter", "false", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-profanity-filter requires --speech", errors.getvalue())

        for value in ("TRUE", "False", "0", "1", "private-filter"):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-profanity-filter", value, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(value=value), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech profanity filter", errors.getvalue())
            self.assertNotIn("private-filter", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            simulate.assert_not_called()

    def test_explicit_speech_model_changes_only_speech_collection_xml(self):
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for model in ("default", "numbers_and_commands", "phone_call"):
            with self.subTest(model=model), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"],
                    speech_model=model,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                if step["stage"] == "speech" and "twiml" in step:
                    actual = fromstring(step["twiml"])
                    expected = fromstring(original["twiml"])
                    self.assertEqual(actual[0].get("speechModel"), model)
                    del actual[0].attrib["speechModel"]
                    self.assertEqual(actual[0].attrib, expected[0].attrib)
                    self.assertEqual(actual[0][0].text, expected[0][0].text)
                else:
                    self.assertEqual(step, original)

    def test_invalid_speech_model_fails_before_hints_inputs_or_classifier(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid model must fail before hint iteration")

        for model in ("", "DEFAULT", "experimental_utterances", "private-model", 1, []):
            with self.subTest(model=repr(model)[:30]), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError) as error:
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), speech_model=model,
                    )
            classify.assert_not_called()
            self.assertNotIn("private-model", str(error.exception))

    def test_cli_speech_model_requires_speech_and_uses_exact_allowlist(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-model", "phone_call", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-model requires --speech", errors.getvalue())

        for model in ("DEFAULT", "experimental_utterances", "private-model"):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-model", model, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(model=model), offline_guard(), redirect_stdout(output), \
                    redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech model", errors.getvalue())
            self.assertNotIn("private-model", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            simulate.assert_not_called()

    def test_explicit_speech_timeout_changes_only_speech_collection_xml(self):
        baseline = simulate_speech(["", "synthetic"], '{"intent":"support"}', ["1"])
        for speech_timeout in (1, 60, "auto"):
            with self.subTest(speech_timeout=speech_timeout), offline_guard():
                steps = simulate_speech(
                    ["", "synthetic"], '{"intent":"support"}', ["1"],
                    speech_timeout=speech_timeout,
                )
            self.assertEqual(steps[-1]["target"], "demo-support")
            for step, original in zip(steps, baseline, strict=True):
                if step["stage"] == "speech" and "twiml" in step:
                    actual = fromstring(step["twiml"])
                    expected = fromstring(original["twiml"])
                    self.assertEqual(actual[0].get("speechTimeout"), str(speech_timeout))
                    actual[0].set("speechTimeout", "2")
                    self.assertEqual(actual.attrib, expected.attrib)
                    self.assertEqual(actual[0].attrib, expected[0].attrib)
                    self.assertEqual(actual[0][0].text, expected[0][0].text)
                else:
                    self.assertEqual(step, original)

    def test_invalid_speech_timeout_fails_before_hints_inputs_or_classifier(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid timeout must fail before hint iteration")

        for speech_timeout in (None, True, 0, 61, "2", "AUTO", "private-timeout"):
            with self.subTest(speech_timeout=speech_timeout), offline_guard(), \
                    patch("zentomic.demo.parse_intent_response") as classify:
                with self.assertRaises(ValueError) as error:
                    simulate_speech(
                        unused_inputs(), None, unused_inputs(),
                        hints=UnreadHints(["sales"]), speech_timeout=speech_timeout,
                    )
            classify.assert_not_called()
            self.assertNotIn("private-timeout", str(error.exception))

    def test_cli_speech_timeout_is_reused_only_for_speech_retries(self):
        for value, expected_value in (("1", "1"), ("60", "60"), ("auto", "auto")):
            output = io.StringIO()
            args = ["--speech-timeout", value, "--speech", "", "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}', "1"]
            with self.subTest(value=value), offline_guard(), redirect_stdout(output):
                self.assertEqual(main(args), 0)
            steps = json.loads(output.getvalue())["steps"]
            speech_xml = [fromstring(step["twiml"])[0] for step in steps
                          if step["stage"] == "speech" and "twiml" in step]
            self.assertEqual(len(speech_xml), 2)
            self.assertTrue(all(gather.get("speechTimeout") == expected_value
                                for gather in speech_xml))
            confirmation = [fromstring(step["twiml"])[0] for step in steps
                            if step["stage"] == "confirmation" and "twiml" in step]
            self.assertTrue(confirmation)
            self.assertTrue(all(gather.get("speechTimeout") is None
                                for gather in confirmation))
            self.assertEqual(steps[-1]["target"], "demo-sales")

    def test_cli_invalid_speech_timeout_is_private_and_prints_no_partial_json(self):
        for value in ("0", "00", "01", "61", "+1", "1.0", "AUTO",
                      "private-timeout", "9" * 10000):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech-timeout", value, "--speech", "synthetic",
                    "--classifier-response", '{"intent":"sales"}']
            with self.subTest(value=value[:20]), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors), \
                    patch("zentomic.demo.simulate_speech") as simulate:
                with self.assertRaises(SystemExit) as error:
                    main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("invalid synthetic speech timeout", errors.getvalue())
            self.assertNotIn("private-timeout", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            simulate.assert_not_called()

    def test_cli_speech_timeout_requires_speech_and_cannot_configure_other_modes(self):
        for mode in ([], ["1"], ["--intent", "sales"],
                     ["--classifier-response", '{"intent":"sales"}'],
                     ["--dial-result", "busy"], ["--call-status", "completed"],
                     ["--business-hours-at", "2026-01-05T14:00:00Z"]):
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(mode=mode), offline_guard(), \
                    redirect_stdout(output), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as error:
                    main(["--speech-timeout", "auto", *mode])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("--speech-timeout requires --speech", errors.getvalue())

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

    def test_unicode_multiline_hints_fail_before_consumption_or_classification(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=ord(separator)):
                with offline_guard(), patch("zentomic.demo.parse_intent_response") as classify:
                    with self.assertRaises(ValueError) as error:
                        simulate_speech(
                            unused_inputs(), None, unused_inputs(),
                            hints=["synthetic-private" + separator + "support"],
                        )
                classify.assert_not_called()
                self.assertNotIn("synthetic-private", str(error.exception))

    def test_cli_unicode_multiline_hints_are_private_input_errors(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            output, errors = io.StringIO(), io.StringIO()
            args = ["--speech", "synthetic", "--classifier-response", '{"intent":"sales"}',
                    "--speech-hint", "synthetic-private" + separator + "support"]
            with self.subTest(separator=ord(separator)):
                with offline_guard(), redirect_stdout(output), redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as error:
                        main(args)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("invalid synthetic speech hints", errors.getvalue())
                self.assertNotIn("synthetic-private", errors.getvalue())
                self.assertNotIn("Traceback", errors.getvalue())

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
