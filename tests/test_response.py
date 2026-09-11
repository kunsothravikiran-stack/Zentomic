"""Offline proxy response contracts for application-generated TwiML."""

import json
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.response import twiml_response
from zentomic.twiml import (
    render_dial, render_dtmf_gather, render_hangup, render_speech_gather,
)


class TwimlResponseTests(unittest.TestCase):
    def test_all_renderers_keep_xml_as_the_unencoded_body(self):
        documents = (
            render_dtmf_gather("Press 1.", action_path="/voice/menu"),
            render_speech_gather("How can we help?", action_path="/voice/speech"),
            render_dial(["+12025550100"], action_path="/voice/dial-result"),
            render_hangup(),
        )
        for xml in documents:
            with self.subTest(xml=xml):
                result = twiml_response(xml)
                self.assertEqual(result, {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/xml; charset=utf-8",
                                "Cache-Control": "no-store"},
                    "body": xml,
                    "isBase64Encoded": False,
                })
                # Lambda serializes the envelope; the body must not be JSON-quoted.
                delivered = json.loads(json.dumps(result))["body"]
                self.assertEqual(delivered, xml)
                self.assertEqual(fromstring(delivered).tag, "Response")

    def test_unicode_and_xml_escaping_survive_proxy_serialization(self):
        prompt = 'Café & తెలుగు <Support> "hello" 😀'
        xml = render_hangup(prompt)
        response = json.loads(json.dumps(twiml_response(xml), ensure_ascii=False))
        self.assertEqual(fromstring(response["body"]).find("Say").text, prompt)
        self.assertEqual(response["body"].encode("utf-8"), xml.encode("utf-8"))

    def test_invalid_body_errors_are_generic(self):
        for body in (None, b"<Response/>", {}, [], 1, True, "", " \t\n",
                     "synthetic-private-\ud800", "\udfff"):
            with self.subTest(body=repr(body)), self.assertRaises(ValueError) as caught:
                twiml_response(body)
            self.assertEqual(str(caught.exception),
                             "xml must be a nonblank UTF-8 encodable string")

    def test_response_and_headers_are_fresh_for_each_invocation(self):
        first = twiml_response(render_hangup())
        first["statusCode"] = 500
        first["headers"]["Content-Type"] = "application/json"
        first["headers"]["X-Private"] = "synthetic"
        second = twiml_response(render_hangup())
        self.assertEqual(second["statusCode"], 200)
        self.assertEqual(second["headers"], {
            "Content-Type": "application/xml; charset=utf-8",
            "Cache-Control": "no-store",
        })

    def test_builder_is_offline_and_does_not_log_body(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ), patch("builtins.print") as output:
            self.assertEqual(twiml_response(render_hangup())["statusCode"], 200)
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
