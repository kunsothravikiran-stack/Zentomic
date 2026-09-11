"""Keep public Python examples executable without credentials or services."""

from pathlib import Path
import re
import unittest
from unittest.mock import patch


class ReadmeExampleTests(unittest.TestCase):
    def test_internal_navigation_targets_real_unique_headings(self):
        readme = Path(__file__).resolve().parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8")
        # Ignore code comments and literal links inside fenced examples.
        prose = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", text, flags=re.M | re.S)
        headings = re.findall(r"^#{1,6} (.+)$", prose, re.M)
        # The README currently uses plain ASCII headings. Keep this check
        # scoped to those, rather than claiming a general Markdown renderer.
        anchors = [re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                   for heading in headings]
        links = re.findall(r"\[[^\]\n]+\]\(#([^\s)]+)\)", prose)
        self.assertTrue(links, "README must provide internal navigation")
        for target in links:
            with self.subTest(target=target):
                self.assertEqual(anchors.count(target), 1,
                                 "navigation must name exactly one real heading")

    def test_python_examples_are_independently_runnable(self):
        readme = Path(__file__).resolve().parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8")
        examples = list(re.finditer(r"^```python\n(.*?)^```$", text, re.M | re.S))
        self.assertTrue(examples, "README must contain Python examples")
        with patch.dict("os.environ", {}, clear=True), patch(
            "socket.socket", side_effect=AssertionError("Network forbidden"),
        ):
            for example in examples:
                line = text.count("\n", 0, example.start()) + 2
                with self.subTest(line=line):
                    source = example.group(1)
                    # Retain README line numbers and assertions, even under -O.
                    code = compile("\n" * (line - 1) + source, str(readme),
                                   "exec", optimize=0)
                    # Explicitly marked integration sketches are syntax-checked
                    # only: they require configuration outside this scaffold.
                    if source.startswith("# example: compile-only\n"):
                        continue
                    exec(code, {"__name__": "__readme_example__"})


if __name__ == "__main__":
    unittest.main()
