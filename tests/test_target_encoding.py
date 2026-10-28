"""Routing configuration must survive UTF-8 session serialization."""

import unittest

from zentomic.dial import resolve_dial_result
from zentomic.gather import resolve_gather
from zentomic.intent import resolve_intent, resolve_intent_confirmation
from zentomic.routing import MAX_TARGET_IDENTIFIER_BYTES, resolve_dtmf
from zentomic.speech import resolve_speech_gather


class TargetEncodingTests(unittest.TestCase):
    def fallback_cases(self, target):
        return (
            lambda: resolve_dtmf("1", {"1": "sales"}, fallback_target=target),
            lambda: resolve_intent("sales", {"sales": "sales"},
                                   fallback_target=target, confirmed=True),
            lambda: resolve_gather("9", {}, fallback_target=target,
                                   attempts=0, hangup_digit="9"),
            lambda: resolve_gather(None, {}, fallback_target=target, attempts=3),
            lambda: resolve_intent_confirmation("1", "sales", {"sales": "sales"},
                                                fallback_target=target, attempts=0),
            lambda: resolve_speech_gather("help", fallback_target=target, attempts=0),
            lambda: resolve_dial_result("completed", fallback_target=target),
            lambda: resolve_dial_result("busy", fallback_target=target),
        )

    def test_surrogate_fallbacks_fail_before_any_decision(self):
        for target in ("private-\ud800", "private-\udfff", "private-\ud83d\ude00"):
            for index, resolve in enumerate(self.fallback_cases(target)):
                with self.subTest(target=ascii(target), case=index):
                    with self.assertRaises(ValueError) as caught:
                        resolve()
                    self.assertNotIn("private-", str(caught.exception))
                    self.assertNotIsInstance(caught.exception, UnicodeError)

    def test_unused_surrogate_destinations_are_rejected(self):
        for target in ("private-\ud800", "private-\udfff"):
            cases = (
                lambda: resolve_dtmf(None, {"1": target}, fallback_target="reception"),
                lambda: resolve_intent(None, {"sales": target}, fallback_target="reception"),
                lambda: resolve_gather(None, {"1": target},
                                       fallback_target="reception", attempts=3),
                lambda: resolve_intent_confirmation(None, None, {"sales": target},
                                                    fallback_target="reception", attempts=3),
            )
            for index, resolve in enumerate(cases):
                with self.subTest(target=ascii(target), case=index):
                    with self.assertRaises(ValueError) as caught:
                        resolve()
                    self.assertNotIn("private-", str(caught.exception))

    def test_valid_unicode_targets_are_preserved_exactly(self):
        for target in (" team/受付 ", "team/😀", "team/e\u0301", "team/é"):
            with self.subTest(target=target):
                self.assertEqual(resolve_dtmf("1", {"1": target},
                                             fallback_target="reception"), target)
                self.assertEqual(resolve_intent("sales", {"sales": target},
                                               fallback_target="reception", confirmed=True), target)
                self.assertEqual(resolve_dial_result("busy", fallback_target=target).target, target)
                self.assertEqual(resolve_speech_gather(None, fallback_target=target,
                                                      attempts=3).target, target)

    def test_target_byte_limit_accepts_exact_boundaries(self):
        for target in ("a" * MAX_TARGET_IDENTIFIER_BYTES,
                       "é" * (MAX_TARGET_IDENTIFIER_BYTES // 2),
                       "😀" * (MAX_TARGET_IDENTIFIER_BYTES // 4)):
            with self.subTest(target=target[:4]):
                self.assertEqual(resolve_dtmf("1", {"1": target},
                                             fallback_target="reception"), target)
                self.assertEqual(resolve_intent("sales", {"sales": target},
                                               fallback_target="reception",
                                               confirmed=True), target)
                self.assertEqual(resolve_dial_result("busy", fallback_target=target).target,
                                 target)

    def test_oversized_targets_fail_across_shared_policies(self):
        for target in ("a" * (MAX_TARGET_IDENTIFIER_BYTES + 1),
                       "é" * (MAX_TARGET_IDENTIFIER_BYTES // 2) + "a",
                       "😀" * (MAX_TARGET_IDENTIFIER_BYTES // 4) + "a",
                       "private-" + "x" * 10000):
            for index, resolve in enumerate(self.fallback_cases(target)):
                with self.subTest(target=target[:8], case=index), \
                        self.assertRaises(ValueError) as caught:
                    resolve()
                self.assertNotIn("private-", str(caught.exception))

            for index, resolve in enumerate((
                lambda: resolve_dtmf(None, {"1": target}, fallback_target="reception"),
                lambda: resolve_intent(None, {"sales": target},
                                       fallback_target="reception"),
                lambda: resolve_gather(None, {"1": target},
                                       fallback_target="reception", attempts=3),
                lambda: resolve_intent_confirmation(
                    None, None, {"sales": target},
                    fallback_target="reception", attempts=3,
                ),
            )):
                with self.subTest(target=target[:8], unused_case=index), \
                        self.assertRaises(ValueError):
                    resolve()

    def test_oversized_target_is_rejected_before_content_scanning(self):
        class OversizedTarget(str):
            def strip(self, *args, **kwargs):
                raise AssertionError("oversized target must not be stripped")

            def encode(self, *args, **kwargs):
                raise AssertionError("oversized target must not be encoded")

        target = OversizedTarget("x" * (MAX_TARGET_IDENTIFIER_BYTES + 1))
        with self.assertRaisesRegex(
            ValueError, "^fallback_target must be a nonblank UTF-8 string of at most 256 bytes$",
        ):
            resolve_dtmf(None, {}, fallback_target=target)


if __name__ == "__main__":
    unittest.main()
