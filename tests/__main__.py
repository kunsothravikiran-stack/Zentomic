"""Opt-in guarded test discovery and execution: python -m tests.

This catches accidental environment/network use, not hostile code. It cannot
isolate subprocesses, native libraries, or previously captured socket handles.
"""

import argparse
import os
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def offline_guard():
    """Hide inherited settings and block common Python socket/DNS entrypoints."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname",
                     "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
            stack.enter_context(patch(
                "socket." + name,
                side_effect=AssertionError("Network forbidden in offline tests"),
            ))
        yield


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*",
        help="dotted unittest module, class, or method names; omit for the full suite",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    # Keep discovery and named loading inside the guard, including imports.
    with offline_guard():
        loader = unittest.TestLoader()
        if args.names:
            suite = loader.loadTestsFromNames(args.names)
        else:
            suite = loader.discover(str(root / "tests"), top_level_dir=str(root))
        if suite.countTestCases() == 0:
            raise RuntimeError("Offline test discovery found no tests")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
