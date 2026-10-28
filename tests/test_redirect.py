"""Offline checks for server-selected transitions between hosted voice steps."""

import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.response import twiml_response
from zentomic.twiml import render_redirect


class RedirectTests(unittest.TestCase):
    def test_redirect_posts_to_exact_root_relative_path(self):
        for path in ("/menu", "/voice/confirm-intent", "/v1/menu_result/"):
            with self.subTest(path=path):
                root = fromstring(render_redirect(action_path=path))
                self.assertEqual(root.tag, "Response")
                self.assertEqual(root.attrib, {})
                self.assertEqual([child.tag for child in root], ["Redirect"])
                self.assertEqual(root[0].attrib, {"method": "POST"})
                self.assertEqual(root[0].text, path)
                self.assertEqual(len(root[0]), 0)

    def test_untrusted_or_ambiguous_paths_fail_without_reflection(self):
        for path in (
            None, True, 1, [], {}, "", "/", "menu", "//example.invalid/menu",
            "https://example.invalid/menu", "/menu?attempts=0", "/menu#fragment",
            "/menu/../other", "/menu//other", "/menu%2fother", "/menu\\other",
            "/menu\n", "/menu\x00", "/menu\ud800", "/తెలుగు",
            "/synthetic-private</Redirect><Dial>injected</Dial>",
        ):
            with self.subTest(path=repr(path)), self.assertRaises(ValueError) as caught:
                render_redirect(action_path=path)
            self.assertEqual(str(caught.exception),
                             "action_path must be a root-relative path of named segments")

    def test_proxy_envelope_is_twiml_not_http_redirection(self):
        xml = render_redirect(action_path="/voice/menu")
        response = twiml_response(xml)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(response["body"], xml)
        self.assertFalse(response["isBase64Encoded"])
        self.assertNotIn("Location", response["headers"])

    def test_renderer_is_deterministic_and_offline(self):
        expected = render_redirect(action_path="/voice/menu")
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ):
            self.assertEqual(render_redirect(action_path="/voice/menu"), expected)


if __name__ == "__main__":
    unittest.main()
