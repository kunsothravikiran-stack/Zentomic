import unittest

from zentomic.webhook_event import parse_form_event


FORM = "application/x-www-form-urlencoded"


def event(version="1.0"):
    candidate = {
        "version": version,
        "headers": {"Content-Type": FORM},
        "body": "Digits=1",
    }
    if version == "2.0":
        candidate["requestContext"] = {"http": {"method": "POST"}}
    else:
        candidate["httpMethod"] = "POST"
    return candidate


class PrimitiveBoundaryTests(unittest.TestCase):
    def test_plain_events_still_parse(self):
        for version in ("1.0", "2.0"):
            with self.subTest(version=version):
                self.assertEqual(parse_form_event(event(version)), {"Digits": "1"})

    def test_active_event_dictionary_is_rejected_before_get(self):
        class ActiveEvent(dict):
            def get(self, *args, **kwargs):
                raise AssertionError("active event lookup must not run")

        with self.assertRaisesRegex(ValueError, "^event must be a plain dictionary$"):
            parse_form_event(ActiveEvent(event()))

    def test_active_control_strings_are_rejected_before_comparison(self):
        class ActiveString(str):
            def __eq__(self, other):
                raise AssertionError("active comparison must not run")

            def __ne__(self, other):
                raise AssertionError("active comparison must not run")

        cases = (
            ({**event(), "version": ActiveString("1.0")},
             "^unsupported proxy event version$"),
            ({**event(), "httpMethod": ActiveString("POST")},
             "^form webhook requires POST$"),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_form_event(candidate)

    def test_active_v2_contexts_are_rejected_before_get(self):
        class ActiveContext(dict):
            def get(self, *args, **kwargs):
                raise AssertionError("active context lookup must not run")

        cases = (
            ({**event("2.0"), "requestContext": ActiveContext({"http": {"method": "POST"}})},
             "^v2 request context must be a plain dictionary$"),
            ({**event("2.0"), "requestContext": {"http": ActiveContext({"method": "POST"})}},
             "^v2 HTTP context must be a plain dictionary$"),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_form_event(candidate)


if __name__ == "__main__":
    unittest.main()
