"""Opt-in guarded test discovery and execution: python -m tests.

This catches accidental environment/network use, not hostile code. It cannot
isolate subprocesses, native libraries, or previously captured socket handles.
"""

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
                     "gethostbyname_ex", "gethostbyaddr"):
            stack.enter_context(patch(
                "socket." + name,
                side_effect=AssertionError("Network forbidden in offline tests"),
            ))
        yield


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    # Keep discovery inside the guard so import-time work is covered too.
    with offline_guard():
        suite = unittest.TestLoader().discover(
            str(root / "tests"), top_level_dir=str(root),
        )
        if suite.countTestCases() == 0:
            raise RuntimeError("Offline test discovery found no tests")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
