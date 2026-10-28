"""Offline serialization checks; no Twilio account or live calls required."""

import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.twiml import (
    render_dial, render_dtmf_gather, render_hangup, render_redirect,
    render_reject, render_speech_gather, render_voicemail,
)


class ActionPathLimitTests(unittest.TestCase):
    def renderers(self):
        return (
            lambda path: render_dtmf_gather("Choose a department.", action_path=path),
            lambda path: render_speech_gather("Name a department.", action_path=path),
            lambda path: render_dial(["+12025550101"], action_path=path),
            lambda path: render_redirect(action_path=path),
        )

    def test_exact_path_limit_round_trips_across_all_renderers(self):
        for path in ("/" + "a" * 2047, "/a" * 1024, "/" + "a" * 2046 + "/"):
            for index, render in enumerate(self.renderers()):
                with self.subTest(renderer=index, trailing_slash=path.endswith("/")):
                    element = fromstring(render(path))[0]
                    actual = element.text if element.tag == "Redirect" else element.get("action")
                    self.assertEqual(actual, path)

    def test_oversized_paths_are_rejected_before_serialization(self):
        for path in ("/" + "a" * 2048, "/a" * 1024 + "/", "/private-" + "a" * 100000):
            for index, render in enumerate(self.renderers()):
                with self.subTest(renderer=index, length=len(path)), \
                        patch("zentomic.twiml._serialize") as serialize:
                    with self.assertRaisesRegex(ValueError, "^action_path exceeds 2048 characters$"):
                        render(path)
                    serialize.assert_not_called()

    def test_oversized_path_does_not_enter_regex_validation(self):
        with patch("zentomic.twiml.re.fullmatch") as match:
            with self.assertRaisesRegex(ValueError, "^action_path exceeds 2048 characters$"):
                render_redirect(action_path="/" + "a" * 2048)
            match.assert_not_called()


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


class VoicemailTests(unittest.TestCase):
    def render(self, prompt="Please leave a message after the beep.", **kwargs):
        config = {
            "action_path": "/voice/voicemail-result",
            "recording_status_path": "/voice/voicemail-status",
        }
        config.update(kwargs)
        return render_voicemail(prompt, **config)

    def test_prompt_precedes_bounded_recording_with_two_post_callbacks(self):
        root = fromstring(self.render())
        self.assertEqual(root.tag, "Response")
        self.assertEqual(root.attrib, {})
        self.assertEqual([child.tag for child in root], ["Say", "Record"])
        self.assertEqual(root[0].text, "Please leave a message after the beep.")
        self.assertEqual(root[1].attrib, {
            "action": "/voice/voicemail-result",
            "method": "POST",
            "recordingStatusCallback": "/voice/voicemail-status",
            "recordingStatusCallbackMethod": "POST",
            "recordingStatusCallbackEvent": "completed absent",
            "maxLength": "120",
            "timeout": "5",
            "finishOnKey": "#",
            "playBeep": "true",
            "trim": "trim-silence",
        })
        self.assertEqual(len(root[1]), 0)
        self.assertNotIn("transcribe", root[1].attrib)

    def test_project_bounds_and_finish_keys_round_trip(self):
        for max_length in (2, 120, 600):
            for timeout in (1, 5, 60):
                for finish_on_key in ("0", "9", "*", "#"):
                    with self.subTest(max_length=max_length, timeout=timeout,
                                      finish_on_key=finish_on_key):
                        record = fromstring(self.render(
                            max_length=max_length, timeout=timeout,
                            finish_on_key=finish_on_key,
                        ))[1]
                        self.assertEqual(record.get("maxLength"), str(max_length))
                        self.assertEqual(record.get("timeout"), str(timeout))
                        self.assertEqual(record.get("finishOnKey"), finish_on_key)

    def test_invalid_recording_policy_is_rejected_before_serialization(self):
        cases = (
            {"max_length": 1}, {"max_length": 601}, {"max_length": True},
            {"max_length": 2.0}, {"max_length": "120"}, {"timeout": 0},
            {"timeout": 61}, {"timeout": False}, {"timeout": 5.0},
            {"finish_on_key": ""}, {"finish_on_key": "12"},
            {"finish_on_key": "a"}, {"finish_on_key": True},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch("zentomic.twiml._serialize") as serialize:
                with self.assertRaises(ValueError):
                    self.render(**kwargs)
                serialize.assert_not_called()

    def test_both_callback_paths_use_the_shared_hosted_path_policy(self):
        valid = "/" + "a" * 2047
        root = fromstring(self.render(
            action_path=valid, recording_status_path="/v1/recording_status/",
        ))
        self.assertEqual(root[1].get("action"), valid)
        self.assertEqual(root[1].get("recordingStatusCallback"), "/v1/recording_status/")
        invalid = (None, "", "/", "voice/result", "https://example.invalid/result",
                   "/voice/result?next=1", "/voice/../other", "/" + "a" * 2048)
        for path in invalid:
            for field in ("action_path", "recording_status_path"):
                with self.subTest(field=field, path=repr(path)[:40]), \
                        self.assertRaises(ValueError):
                    self.render(**{field: path})

    def test_prompt_is_escaped_and_uses_shared_text_validation(self):
        prompt = 'Message & <Dial>synthetic</Dial> "now"'
        root = fromstring(self.render(prompt))
        self.assertEqual(root[0].text, prompt)
        self.assertEqual(root.findall(".//Dial"), [])
        for invalid in (None, True, "", " \t\n", "x" * 1001, "Message\x00"):
            with self.subTest(prompt=repr(invalid)[:40]), self.assertRaises(ValueError):
                self.render(invalid)

    def test_rendering_is_deterministic_and_offline(self):
        expected = self.render()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(self.render(), expected)


class RejectTests(unittest.TestCase):
    def test_default_rejection_is_the_only_verb(self):
        root = fromstring(render_reject())
        self.assertEqual(root.tag, "Response")
        self.assertEqual(root.attrib, {})
        self.assertEqual([child.tag for child in root], ["Reject"])
        self.assertEqual(root[0].attrib, {})
        self.assertEqual(len(root[0]), 0)
        self.assertIsNone(root[0].text)
        self.assertEqual(render_reject(reason="rejected"), render_reject())

    def test_busy_rejection_uses_the_only_supported_attribute(self):
        root = fromstring(render_reject(reason="busy"))
        self.assertEqual([child.tag for child in root], ["Reject"])
        self.assertEqual(root[0].attrib, {"reason": "busy"})
        self.assertEqual(len(root[0]), 0)
        self.assertIsNone(root[0].text)

    def test_invalid_reasons_are_rejected_before_serialization(self):
        reasons = (None, True, False, 1, [], {}, "", "Busy", " rejected ", "no-answer")
        for reason in reasons:
            with self.subTest(reason=repr(reason)), \
                    patch("zentomic.twiml._serialize") as serialize:
                with self.assertRaisesRegex(ValueError, "^reason must be 'rejected' or 'busy'$"):
                    render_reject(reason=reason)
                serialize.assert_not_called()

    def test_rendering_is_deterministic_and_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ):
            self.assertEqual(render_reject(), "<Response><Reject /></Response>")
            self.assertEqual(
                render_reject(reason="busy"),
                '<Response><Reject reason="busy" /></Response>',
            )


if __name__ == "__main__":
    unittest.main()
