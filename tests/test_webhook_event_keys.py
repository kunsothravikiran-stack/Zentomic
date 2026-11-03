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


class PlainKeyBoundaryTests(unittest.TestCase):
    def test_plain_event_keys_still_parse(self):
        for version in ("1.0", "2.0"):
            with self.subTest(version=version):
                self.assertEqual(parse_form_event(event(version)), {"Digits": "1"})

    def test_active_dictionary_keys_are_rejected_before_lookup(self):
        class ActiveKey(str):
            armed = False

            def __hash__(self):
                if self.armed:
                    raise AssertionError("active key hashing must not run")
                return super().__hash__()

            def __eq__(self, other):
                if self.armed:
                    raise AssertionError("active key comparison must not run")
                return super().__eq__(other)

        cases = []
        top_level = event()
        top_key = ActiveKey("synthetic")
        top_level[top_key] = "value"
        cases.append((top_level, "^event keys must be plain strings$"))

        context_event = event("2.0")
        context_key = ActiveKey("synthetic")
        context_event["requestContext"][context_key] = "value"
        cases.append((context_event, "^v2 request context keys must be plain strings$"))

        http_event = event("2.0")
        http_key = ActiveKey("synthetic")
        http_event["requestContext"]["http"][http_key] = "value"
        cases.append((http_event, "^v2 HTTP context keys must be plain strings$"))

        for candidate, message in cases:
            for key in (top_key, context_key, http_key):
                key.armed = True
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_form_event(candidate)


if __name__ == "__main__":
    unittest.main()
