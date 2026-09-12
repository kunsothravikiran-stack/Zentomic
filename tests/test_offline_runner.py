"""Verify the opt-in offline runner's guard without making network requests."""

import os
import socket
import unittest
from contextlib import redirect_stderr
from io import StringIO
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

    def test_getnameinfo_is_blocked_and_restored_without_real_dns(self):
        # The native reverse resolver need not call Python's gethostbyaddr.
        # Install a sentinel outside the guard so a regression cannot use DNS.
        for fail in (False, True):
            with self.subTest(fail=fail), patch("socket.getnameinfo") as resolver:
                try:
                    with offline_guard():
                        with self.assertRaisesRegex(
                            AssertionError, "^Network forbidden in offline tests$",
                        ):
                            socket.getnameinfo(("192.0.2.1", 443), socket.NI_NAMEREQD)
                        resolver.assert_not_called()
                        if fail:
                            raise RuntimeError("synthetic failure")
                except RuntimeError as error:
                    self.assertEqual(str(error), "synthetic failure")
                self.assertIs(socket.getnameinfo, resolver)
                resolver.assert_not_called()

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
                self.assertEqual(main([]), code)
                runner.return_value.run.assert_called_once_with(suite)

    def test_empty_discovery_fails_before_running(self):
        with patch("tests.__main__.unittest.TestLoader") as loader, patch(
            "tests.__main__.unittest.TextTestRunner",
        ) as runner:
            loader.return_value.discover.return_value.countTestCases.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "found no tests"):
                main([])
            runner.assert_not_called()

    def test_named_selection_is_loaded_and_run_inside_guard(self):
        suite = Mock()
        suite.countTestCases.return_value = 2
        names = ["tests.test_handler", "tests.test_cli.EventCliTests"]

        def load(selected):
            self.assertEqual(selected, names)
            self.assertEqual(dict(os.environ), {})
            with self.assertRaises(AssertionError):
                socket.getaddrinfo("example.invalid", 443)
            return suite

        def run(selected):
            self.assertIs(selected, suite)
            self.assertEqual(dict(os.environ), {})
            with self.assertRaises(AssertionError):
                socket.socket()
            return Mock(wasSuccessful=lambda: True)

        with patch("tests.__main__.unittest.TestLoader") as loader, patch(
            "tests.__main__.unittest.TextTestRunner",
        ) as runner:
            loader.return_value.loadTestsFromNames.side_effect = load
            runner.return_value.run.side_effect = run
            self.assertEqual(main(names), 0)
            loader.return_value.discover.assert_not_called()
            loader.return_value.loadTestsFromNames.assert_called_once_with(names)
            runner.return_value.run.assert_called_once_with(suite)

    def test_module_class_and_method_selections_run_real_tests(self):
        for name, count in (
            ("tests.test_discovery", 1),
            ("tests.test_discovery.DiscoveryTests", 1),
            ("tests.test_discovery.DiscoveryTests."
             "test_repository_root_discovers_every_test_module", 1),
        ):
            errors = StringIO()
            with self.subTest(name=name), redirect_stderr(errors):
                self.assertEqual(main([name]), 0)
                self.assertIn(f"Ran {count} test", errors.getvalue())

    def test_missing_module_class_or_method_fails_selection(self):
        for name in (
            "tests.test_missing_synthetic_module",
            "tests.test_handler.MissingSyntheticClass",
            "tests.test_handler.HandlerTests.test_missing_synthetic_method",
        ):
            with self.subTest(name=name), redirect_stderr(StringIO()):
                self.assertEqual(main([name]), 1)

    def test_empty_named_selection_fails_before_running(self):
        with patch("tests.__main__.unittest.TestLoader") as loader, patch(
            "tests.__main__.unittest.TextTestRunner",
        ) as runner:
            loader.return_value.loadTestsFromNames.return_value.countTestCases.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "found no tests"):
                main(["tests.synthetic_empty"])
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
