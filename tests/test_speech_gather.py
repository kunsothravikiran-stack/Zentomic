"""Speech collection XML contracts, with no transcription or provider traffic."""

import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.twiml import render_speech_gather


class SpeechGatherTests(unittest.TestCase):
    def render(self, prompt="How can we help?", **kwargs):
        return render_speech_gather(
            prompt, **{"action_path": "/voice/speech-result", **kwargs},
        )

    def test_speech_only_collection_posts_even_on_silence(self):
        root = fromstring(self.render())
        self.assertEqual(root.tag, "Response")
        self.assertEqual([child.tag for child in root], ["Gather"])
        self.assertEqual(root[0].attrib, {
            "input": "speech", "language": "en-US",
            "action": "/voice/speech-result", "method": "POST",
            "actionOnEmptyResult": "true", "timeout": "5", "speechTimeout": "2",
        })
        self.assertEqual([child.tag for child in root[0]], ["Say"])
        self.assertEqual(root[0][0].text, "How can we help?")

    def test_prompt_markup_and_unicode_are_preserved_as_text(self):
        prompt = 'Café తెలుగు 😀 & <Dial>synthetic</Dial> "help"\t\n'
        root = fromstring(self.render(prompt))
        self.assertEqual(root[0][0].text, prompt)
        self.assertEqual(len(root[0][0]), 0)
        self.assertEqual(root.findall(".//Dial"), [])

    def test_prompt_validation_uses_shared_limits(self):
        for prompt in (None, True, 1, [], {}, "", " \t\n", "x" * 1001,
                       "a\x00", "a\ud800", "a\uffff"):
            with self.subTest(prompt=repr(prompt)[:30]), self.assertRaises(ValueError):
                self.render(prompt)
        self.assertEqual(fromstring(self.render("x" * 1000))[0][0].text, "x" * 1000)

    def test_timeouts_are_independent_and_include_boundaries(self):
        for timeout, speech_timeout in ((1, 60), (60, 1), (8, 3)):
            root = fromstring(self.render(timeout=timeout, speech_timeout=speech_timeout))
            self.assertEqual(root[0].get("timeout"), str(timeout))
            self.assertEqual(root[0].get("speechTimeout"), str(speech_timeout))

    def test_timeouts_reject_nonintegers_and_out_of_range_values(self):
        for name in ("timeout", "speech_timeout"):
            for value in (0, -1, 61, True, False, 1.5, "2", "auto", None, [], {}):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.render(**{name: value})

    def test_callback_paths_use_the_existing_hosted_path_allowlist(self):
        for path in ("/speech", "/voice/speech-result", "/v1/speech_result/"):
            self.assertEqual(fromstring(self.render(action_path=path))[0].get("action"), path)
        for path in (None, 1, [], "", "/", "speech", "//example.invalid/speech",
                     "https://example.invalid/speech", "/speech?x=1", "/speech#x",
                     "/speech/../other", "/speech//other", "/speech%2fother",
                     "/speech\\other", "/speech\n", '/speech" method="GET'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render(action_path=path)

    def test_rendering_is_deterministic_and_offline(self):
        expected = self.render()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(self.render(), expected)

    def test_explicit_default_language_preserves_existing_xml(self):
        self.assertEqual(self.render(language="en-US"), self.render())

    def test_recognition_locale_changes_only_gather_language(self):
        expected = fromstring(self.render(hints=["sales & support"]))
        for language in ("en-US", "en-GB", "en-IN", "es-US", "es-ES", "hi-IN"):
            with self.subTest(language=language):
                actual = fromstring(self.render(language=language, hints=["sales & support"]))
                self.assertEqual(actual[0].get("language"), language)
                actual[0].set("language", "en-US")
                self.assertEqual(actual[0].attrib, expected[0].attrib)
                self.assertEqual(actual[0][0].attrib, {})
                self.assertEqual(actual[0][0].text, expected[0][0].text)
                self.assertEqual([child.tag for child in actual[0]], ["Say"])

    def test_recognition_locale_requires_an_exact_allowlisted_string(self):
        for language in (None, True, 1, [], {}, b"en-US", "", "en-us", "EN-US",
                         " en-US", "en-US ", "en_US", "en", "auto", "multi",
                         "private-locale", "x" * 10000, "en-US\n", "en-US\ud800",
                         'en-US\" method=\"GET'):
            with self.subTest(language=repr(language)[:40]):
                with self.assertRaises(ValueError) as error:
                    self.render(language=language)
                self.assertIn("language", str(error.exception))
                self.assertNotIn("private-locale", str(error.exception))

    def test_invalid_locale_does_not_consume_vocabulary(self):
        class UnreadHints(list):
            def __iter__(self):
                raise AssertionError("invalid language must fail before hint iteration")

        with self.assertRaises(ValueError):
            self.render(language="auto", hints=UnreadHints(["sales"]))

    def test_explicit_locale_is_offline_and_does_not_echo_configuration(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("builtins.print") as output:
            xml = self.render(language="en-IN", hints=["support"])
            self.assertEqual(fromstring(xml)[0].get("language"), "en-IN")
            with self.assertRaises(ValueError):
                self.render(language="private-locale")
            output.assert_not_called()

    def test_omitted_and_empty_hints_preserve_existing_xml(self):
        expected = self.render()
        for hints in (None, [], ()):
            self.assertEqual(self.render(hints=hints), expected)
            self.assertNotIn("hints", fromstring(self.render(hints=hints))[0].attrib)

    def test_hints_preserve_order_duplicates_and_unicode_without_xml_injection(self):
        phrases = ['sales & support', 'Café తెలుగు 😀', '<Dial>"help"</Dial>', 'sales & support']
        original = phrases.copy()
        for hints in (phrases, tuple(phrases)):
            with self.subTest(hints=type(hints).__name__):
                root = fromstring(self.render(hints=hints))
                self.assertEqual(root[0].get("hints"), ",".join(phrases))
                self.assertEqual(root.findall(".//Dial"), [])
                self.assertEqual(root[0][0].text, "How can we help?")
                self.assertEqual(root[0].get("language"), "en-US")
        self.assertEqual(phrases, original)

    def test_hint_count_and_unicode_character_boundaries(self):
        phrases = ["😀" * 100] * 500
        self.assertEqual(fromstring(self.render(hints=phrases))[0].get("hints"),
                         ",".join(phrases))
        for hints in (["x"] * 501, ["x" * 101], ["😀" * 101]):
            with self.subTest(count=len(hints)), self.assertRaises(ValueError):
                self.render(hints=hints)

    def test_hints_reject_ambiguous_invalid_or_nontext_values_privately(self):
        for phrase in (None, True, 1, [], {}, "", " \u00a0", "private,marker",
                       "private\nmarker", "private\rmarker", "private\tmarker",
                       "private\x00marker", "private\x7fmarker", "private\ud800marker",
                       "private\uffffmarker"):
            with self.subTest(phrase=repr(phrase)), self.assertRaises(ValueError) as error:
                self.render(hints=["sales", phrase])
            self.assertIn("hints", str(error.exception))
            self.assertNotIn("private", str(error.exception))

    def test_hints_require_a_bounded_concrete_collection(self):
        for hints in ("sales,support", b"sales", {"sales"}, {"sales": "support"},
                      iter(["sales"]), True, 1):
            with self.subTest(hints=type(hints).__name__), self.assertRaises(ValueError):
                self.render(hints=hints)

        class EndlessHints(list):
            def __iter__(self):
                for _ in range(501):
                    yield "sales"
                raise AssertionError("hint snapshot consumed beyond its bound")

        with self.assertRaises(ValueError):
            self.render(hints=EndlessHints())

    def test_hints_reject_unicode_line_boundaries_without_reflecting_text(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            phrase = "synthetic-private" + separator + "support"
            for hints in ([phrase], (phrase,)):
                with self.subTest(separator=ord(separator), hints=type(hints).__name__):
                    with self.assertRaises(ValueError) as error:
                        self.render(hints=hints)
                    self.assertIn("single-line", str(error.exception))
                    self.assertNotIn("synthetic-private", str(error.exception))

    def test_hints_preserve_nonbreaking_and_other_inline_unicode_spaces(self):
        for space in ("\u00a0", "\u2003", "\u202f", "\u3000"):
            phrase = "technical" + space + "support"
            with self.subTest(space=ord(space)):
                self.assertEqual(fromstring(self.render(hints=[phrase]))[0].get("hints"),
                                 phrase)

    def test_hints_are_snapshotted_once_without_rereading_mutable_input(self):
        class ChangingHints(list):
            def __iter__(self):
                yield "sales"
                self[:] = ["support"]

        hints = ChangingHints(["sales"])
        self.assertEqual(fromstring(self.render(hints=hints))[0].get("hints"), "sales")
        self.assertEqual(hints, ["support"])

    def test_hinted_rendering_is_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ):
            self.assertEqual(fromstring(self.render(hints=["sales", "support"]))[0].get("hints"),
                             "sales,support")


if __name__ == "__main__":
    unittest.main()
