"""Offline forwarding serialization with fictional NANP 555-01xx fixtures."""

import copy
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import fromstring

from zentomic.twiml import render_dial


class DialTests(unittest.TestCase):
    def render(self, numbers=None, **kwargs):
        config = {"action_path": "/voice/dial-result"}
        config.update(kwargs)
        return render_dial(["+12025550100"] if numbers is None else numbers, **config)

    def test_single_destination_posts_result_without_recording(self):
        root = fromstring(self.render())
        self.assertEqual(root.tag, "Response")
        self.assertEqual(root.attrib, {})
        self.assertEqual([child.tag for child in root], ["Dial"])
        self.assertEqual(root[0].attrib, {
            "action": "/voice/dial-result", "method": "POST", "timeout": "20",
            "sequential": "false", "record": "do-not-record",
        })
        self.assertEqual([child.tag for child in root[0]], ["Number"])
        self.assertEqual(root[0][0].text, "+12025550100")
        self.assertEqual(root[0][0].attrib, {})
        self.assertEqual(len(root[0][0]), 0)

    def test_up_to_ten_destinations_preserve_order_and_input(self):
        for size in (1, 2, 10):
            for container in (list, tuple):
                numbers = container(f"+1202555010{i}" for i in range(size))
                original = copy.deepcopy(numbers)
                root = fromstring(self.render(numbers))
                self.assertEqual([child.text for child in root[0]], list(numbers))
                self.assertEqual(numbers, original)

    def test_invalid_containers_and_counts_are_rejected(self):
        for numbers in (None, " +12025550100", b"+12025550100", {}, set(),
                        iter(["+12025550100"]), [], (), ["+12025550100"] * 11):
            with self.subTest(numbers=repr(numbers)), self.assertRaises(ValueError):
                render_dial(numbers, action_path="/voice/dial-result")

    def test_duplicates_are_rejected_without_silently_dropping_a_member(self):
        with self.assertRaises(ValueError):
            self.render(["+12025550100", "+12025550100"])

    def test_invalid_destinations_are_rejected_even_after_valid_entries(self):
        for number in (None, True, 1, [], {}, "", "+", "+1", "+012025550100",
                       "12025550100", "+12025550100 ", " +12025550100",
                       "+12025550100\n", "+1-202-555-0100", "+12025550100x1",
                       "+１２０２５５５０１００", "+1234567890123456",
                       "sip:synthetic@example.invalid", "<Hangup />"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                self.render(["+12025550100", number])

    def test_timeout_bounds_and_types(self):
        for timeout in (5, 20, 60):
            self.assertEqual(fromstring(self.render(timeout=timeout))[0].get("timeout"), str(timeout))
        for timeout in (4, 61, -1, True, False, 20.0, "20", None):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.render(timeout=timeout)

    def test_callback_paths_use_existing_hosted_response_restrictions(self):
        for path in ("/result", "/voice/dial-result", "/v1/dial_result/"):
            self.assertEqual(fromstring(self.render(action_path=path))[0].get("action"), path)
        for path in (None, 1, [], "", "/", "result", "//example.invalid/result",
                     "https://example.invalid/result", "/result?x=1", "/result#x",
                     "/result/../x", "/result//x", "/result%2fx", "/result\\x",
                     "/result\n", '/result" method="GET'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render(action_path=path)

    def test_errors_do_not_echo_destinations(self):
        for numbers in (["SyntheticPrivateDestination"], ["+12025550100"] * 2):
            with self.assertRaises(ValueError) as caught:
                self.render(numbers)
            for number in numbers:
                self.assertNotIn(number, str(caught.exception))

    def test_renderer_is_deterministic_offline_and_silent(self):
        expected = self.render()
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden")
        ), patch("builtins.print") as output:
            self.assertEqual(self.render(), expected)
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
