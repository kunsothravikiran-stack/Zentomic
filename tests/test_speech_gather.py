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


if __name__ == "__main__":
    unittest.main()
