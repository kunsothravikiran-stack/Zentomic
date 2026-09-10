"""Guard against default unittest discovery silently omitting test modules."""

from pathlib import Path
import unittest


def _test_modules(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _test_modules(test)
        else:
            yield type(test).__module__


class DiscoveryTests(unittest.TestCase):
    def test_repository_root_discovers_every_test_module(self):
        root = Path(__file__).resolve().parent.parent
        loader = unittest.TestLoader()
        suite = loader.discover(str(root), top_level_dir=str(root))
        self.assertEqual(loader.errors, [])
        expected = {f"tests.{path.stem}" for path in (root / "tests").glob("test_*.py")}
        self.assertTrue(expected)
        self.assertEqual(set(_test_modules(suite)), expected)


if __name__ == "__main__":
    unittest.main()
