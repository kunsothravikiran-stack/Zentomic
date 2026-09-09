"""Offline serialization checks; no Twilio account or live calls required."""

import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.twiml import render_dtmf_gather, render_hangup


class TwimlTests(unittest.TestCase):
    def render(self, prompt="Press 1 for sales.", **kwargs):
        config = {"action_path": "/voice/menu-result"}
        config.update(kwargs)
        return render_dtmf_gather(prompt, **config)

    def test_single_digit_collection_posts_even_on_silence(self):
        root = fromstring(self.render())
        self.assertEqual(root.tag, "Response")
        self.assertEqual([child.tag for child in root], ["Gather"])
        gather = root[0]
        self.assertEqual(gather.attrib, {
            "input": "dtmf", "numDigits": "1", "finishOnKey": "",
            "action": "/voice/menu-result", "method": "POST",
            "actionOnEmptyResult": "true", "timeout": "5",
        })
        self.assertEqual([child.tag for child in gather], ["Say"])
        self.assertEqual(gather[0].text, "Press 1 for sales.")

    def test_prompt_markup_is_text_not_executable_twiml(self):
        prompt = 'Sales & support <Dial>synthetic-target</Dial> "hello"'
        root = fromstring(self.render(prompt))
        self.assertEqual(root.find("Gather/Say").text, prompt)
        self.assertEqual(root.findall(".//Dial"), [])
        self.assertEqual(len(root.find("Gather/Say")), 0)

    def test_unicode_and_xml_whitespace_are_preserved(self):
        prompt = " Café తెలుగు 😀\t\nMenu "
        self.assertEqual(fromstring(self.render(prompt)).find("Gather/Say").text, prompt)

    def test_invalid_xml_characters_are_rejected(self):
        for char in ("\x00", "\x08", "\x0b", "\x1f", "\ud800", "\udfff", "\ufffe", "\uffff"):
            with self.subTest(char=repr(char)), self.assertRaises(ValueError):
                self.render("Menu" + char)

    def test_prompt_type_and_length_limits(self):
        for prompt in (None, True, 1, [], {}, "", " \t\n", "x" * 1001):
            with self.subTest(prompt=repr(prompt)[:40]), self.assertRaises(ValueError):
                self.render(prompt)
        self.assertEqual(len(fromstring(self.render("x" * 1000)).find("Gather/Say").text), 1000)

    def test_timeout_bounds(self):
        for timeout in (1, 10, 60):
            self.assertEqual(fromstring(self.render(timeout=timeout))[0].get("timeout"), str(timeout))
        for timeout in (0, -1, 61, True, False, 1.5, "5", None):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.render(timeout=timeout)

    def test_callback_paths_are_preserved(self):
        for path in ("/menu", "/voice/menu-result", "/v1/menu_result/"):
            self.assertEqual(fromstring(self.render(action_path=path))[0].get("action"), path)

    def test_ambiguous_or_external_callback_paths_are_rejected(self):
        for path in (
            None, 1, [], "", "/", "menu", "//example.invalid/menu",
            "https://example.invalid/menu", "/menu?next=1", "/menu#fragment",
            "/menu/../other", "/menu//other", "/menu%2fother", "/menu\\other",
            "/menu\n", '/menu" method="GET',
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render(action_path=path)

    def test_rendering_is_deterministic_and_offline(self):
        expected = self.render()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(self.render(), expected)


class HangupTests(unittest.TestCase):
    def test_default_response_contains_only_empty_hangup(self):
        root = fromstring(render_hangup())
        self.assertEqual(root.tag, "Response")
        self.assertEqual(root.attrib, {})
        self.assertEqual([child.tag for child in root], ["Hangup"])
        self.assertEqual(root[0].attrib, {})
        self.assertEqual(len(root[0]), 0)
        self.assertIsNone(root[0].text)
        self.assertEqual(render_hangup(None), render_hangup())

    def test_farewell_precedes_terminal_hangup(self):
        root = fromstring(render_hangup("Thank you. Goodbye."))
        self.assertEqual([child.tag for child in root], ["Say", "Hangup"])
        self.assertEqual(root[0].text, "Thank you. Goodbye.")
        self.assertEqual(root[1].attrib, {})
        self.assertEqual(len(root[1]), 0)

    def test_farewell_markup_cannot_inject_verbs(self):
        prompt = 'Goodbye & <Redirect>/synthetic</Redirect> "thanks"'
        root = fromstring(render_hangup(prompt))
        self.assertEqual(root[0].text, prompt)
        self.assertEqual([child.tag for child in root], ["Say", "Hangup"])
        self.assertEqual(len(root[0]), 0)

    def test_unicode_and_prompt_boundary_are_supported(self):
        for prompt in (" Café తెలుగు 😀\t\nGoodbye ", "x" * 1000):
            with self.subTest(prompt=prompt[:20]):
                self.assertEqual(fromstring(render_hangup(prompt))[0].text, prompt)

    def test_blank_oversized_and_nontext_farewells_are_rejected(self):
        for prompt in (True, False, 1, [], {}, b"Goodbye", "", " \t\n", "x" * 1001):
            with self.subTest(prompt=repr(prompt)[:40]), self.assertRaises(ValueError):
                render_hangup(prompt)

    def test_invalid_xml_characters_are_rejected(self):
        for char in ("\x00", "\x08", "\x0b", "\x1f", "\ud800", "\udfff", "\ufffe", "\uffff"):
            with self.subTest(char=repr(char)), self.assertRaises(ValueError):
                render_hangup("Goodbye" + char)

    def test_rendering_is_deterministic_and_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(render_hangup(), "<Response><Hangup /></Response>")
            self.assertEqual(render_hangup("Goodbye"), render_hangup("Goodbye"))


if __name__ == "__main__":
    unittest.main()
