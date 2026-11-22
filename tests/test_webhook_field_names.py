"""Boundary checks for untrusted form field names."""

import unittest
from urllib.parse import quote

from zentomic.webhook import MAX_FORM_FIELD_NAME_BYTES, parse_form_body


class FormFieldNameTests(unittest.TestCase):
    def test_name_byte_limit_is_inclusive(self):
        names = (
            "A" * MAX_FORM_FIELD_NAME_BYTES,
            "é" * (MAX_FORM_FIELD_NAME_BYTES // 2),
        )
        for name in names:
            with self.subTest(size=len(name.encode("utf-8"))):
                self.assertEqual(
                    parse_form_body(f"{quote(name)}=1"),
                    {name: "1"},
                )

    def test_oversized_names_are_rejected_before_dictionary_hashing(self):
        names = (
            "A" * (MAX_FORM_FIELD_NAME_BYTES + 1),
            "é" * (MAX_FORM_FIELD_NAME_BYTES // 2 + 1),
        )
        for name in names:
            with self.subTest(size=len(name.encode("utf-8"))), self.assertRaisesRegex(
                ValueError, "^form field names must be nonempty, bounded, and contain no whitespace or controls$",
            ):
                parse_form_body(f"{quote(name)}=1")

    def test_whitespace_and_control_characters_are_rejected_in_names(self):
        for name in ("First Name", "First\tName", "First\nName", "A\x00B", "A\x7fB", "A\u2003B"):
            with self.subTest(name=ascii(name)), self.assertRaises(ValueError):
                parse_form_body(f"{quote(name)}=1")

    def test_values_remain_opaque(self):
        for value in ("First Name", "line\nbreak", "A\x00B", "A\u2003B"):
            with self.subTest(value=ascii(value)):
                self.assertEqual(
                    parse_form_body(f"Field={quote(value)}"),
                    {"Field": value},
                )


if __name__ == "__main__":
    unittest.main()
