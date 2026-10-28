"""Prompt text survives XML and proxy serialization without normalization."""

import json
import unittest
from xml.etree.ElementTree import fromstring

from zentomic.response import twiml_response
from zentomic.twiml import render_dtmf_gather, render_hangup, render_speech_gather


class PromptRoundtripTests(unittest.TestCase):
    def renderers(self):
        return (
            lambda prompt: render_dtmf_gather(prompt, action_path="/voice/menu"),
            lambda prompt: render_speech_gather(prompt, action_path="/voice/speech"),
            render_hangup,
        )

    def test_line_endings_survive_xml_and_proxy_roundtrip(self):
        for render in self.renderers():
            for ending in ("\r", "\r\n", "\n", "\r\r\n", "\t"):
                prompt = f"Café తెలుగు 😀{ending}Choose sales.{ending}"
                with self.subTest(renderer=render, ending=repr(ending)):
                    xml = render(prompt)
                    envelope = json.loads(json.dumps(twiml_response(xml)))
                    self.assertEqual(fromstring(envelope["body"]).find(".//Say").text, prompt)
                    self.assertNotIn("\r", xml)

    def test_literal_references_and_markup_remain_text(self):
        prompt = "First\r\n&#13; &amp; <Dial>synthetic</Dial> &#xD;"
        for render in self.renderers():
            with self.subTest(renderer=render):
                xml = render(prompt)
                root = fromstring(xml)
                self.assertEqual(root.find(".//Say").text, prompt)
                self.assertIsNone(root.find(".//Dial"))
                self.assertEqual(xml.count("&#13;"), 1)

    def test_prompt_limit_counts_input_not_serialized_references(self):
        prompt = "x" + "\r" * 999
        for render in self.renderers():
            with self.subTest(renderer=render):
                self.assertEqual(fromstring(render(prompt)).find(".//Say").text, prompt)
                with self.assertRaises(ValueError):
                    render(prompt + "\r")

    def test_oversized_prompts_are_rejected_before_content_scanning(self):
        class UnscannablePrompt(str):
            # Instrument ordering without allocating a huge real prompt or
            # depending on timing/memory measurements in an offline test.
            def strip(self, *args, **kwargs):
                raise AssertionError("oversized prompt must not be stripped")

            def __iter__(self):
                raise AssertionError("oversized prompt must not be scanned")

        for render in self.renderers():
            for text in ("x" * 1001, " " * 1001, " " + "x" * 999 + " ",
                         "😀" * 1001):
                with self.subTest(renderer=render, prefix=repr(text[:3])):
                    with self.assertRaisesRegex(
                        ValueError,
                        "^prompt must be a nonblank string of at most 1000 characters$",
                    ):
                        render(UnscannablePrompt(text))

    def test_prompt_boundary_preserves_whitespace_and_multibyte_text(self):
        for render in self.renderers():
            for prompt in (" " + "x" * 998 + " ", "😀" * 1000):
                with self.subTest(renderer=render, prefix=repr(prompt[:3])):
                    self.assertEqual(fromstring(render(prompt)).find(".//Say").text, prompt)
            with self.subTest(renderer=render, blank=True):
                with self.assertRaises(ValueError):
                    render(" " * 1000)


if __name__ == "__main__":
    unittest.main()
