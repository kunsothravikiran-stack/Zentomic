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


if __name__ == "__main__":
    unittest.main()
