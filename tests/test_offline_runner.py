"""Verify the opt-in offline runner's guard without making network requests."""

import os
import socket
import unittest
from unittest.mock import Mock, patch

from tests.__main__ import main, offline_guard


class OfflineRunnerTests(unittest.TestCase):
    def test_environment_is_empty_and_restored(self):
        with patch.dict(os.environ, {"ZENTOMIC_SYNTHETIC_SETTING": "test"}):
            before = dict(os.environ)
            with offline_guard():
                self.assertEqual(dict(os.environ), {})
            self.assertEqual(dict(os.environ), before)

    def test_socket_connection_and_dns_entrypoints_are_blocked(self):
        with offline_guard():
            for operation, args in (
                (socket.socket, ()),
                (socket.create_connection, (("example.invalid", 443),)),
                (socket.getaddrinfo, ("example.invalid", 443)),
                (socket.gethostbyname, ("example.invalid",)),
                (socket.gethostbyname_ex, ("example.invalid",)),
                (socket.gethostbyaddr, ("192.0.2.1",)),
            ):
                with self.subTest(operation=operation), self.assertRaisesRegex(
                    AssertionError, "^Network forbidden in offline tests$",
                ):
                    operation(*args)

    def test_guards_restore_after_failure(self):
        originals = (socket.socket, socket.create_connection, socket.getaddrinfo)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with offline_guard():
                raise RuntimeError("synthetic failure")
        self.assertEqual(
            (socket.socket, socket.create_connection, socket.getaddrinfo), originals,
        )

    def test_runner_exit_status_and_guard_cover_discovery_and_execution(self):
        suite = Mock()
        suite.countTestCases.return_value = 1

        def guarded_result(value):
            self.assertEqual(dict(os.environ), {})
            with self.assertRaises(AssertionError):
                socket.getaddrinfo("example.invalid", 443)
            return value

        for successful, code in ((True, 0), (False, 1)):
            result = Mock()
            result.wasSuccessful.return_value = successful
            with self.subTest(successful=successful), patch(
                "tests.__main__.unittest.TestLoader",
            ) as loader, patch("tests.__main__.unittest.TextTestRunner") as runner:
                loader.return_value.discover.side_effect = lambda *a, **k: guarded_result(suite)
                runner.return_value.run.side_effect = lambda s: guarded_result(result)
                self.assertEqual(main(), code)
                runner.return_value.run.assert_called_once_with(suite)

    def test_empty_discovery_fails_before_running(self):
        with patch("tests.__main__.unittest.TestLoader") as loader, patch(
            "tests.__main__.unittest.TextTestRunner",
        ) as runner:
            loader.return_value.discover.return_value.countTestCases.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "found no tests"):
                main()
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
