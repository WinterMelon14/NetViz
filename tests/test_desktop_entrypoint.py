import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from desktop import __main__ as desktop_main
from desktop.host import default_worker_command, main as host_main


class _Event:
    def __iadd__(self, _handler):
        return self


class DesktopEntrypointTests(unittest.TestCase):
    def test_worker_command_uses_module_dispatcher_in_source_mode(self):
        request = Path("request.json")
        with patch.object(sys, "frozen", False, create=True):
            command = default_worker_command(request)
        self.assertEqual(command, [sys.executable, "-m", "desktop", "--trace-worker", str(request)])

    def test_worker_command_reuses_executable_when_frozen(self):
        request = Path("request.json")
        with patch.object(sys, "frozen", True, create=True):
            command = default_worker_command(request)
        self.assertEqual(command, [sys.executable, "--trace-worker", str(request)])

    def test_self_check_reports_release_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.html"
            index_path.write_text("<html></html>", encoding="utf-8")
            output = io.StringIO()
            with patch("desktop.host.frontend_index_path", return_value=index_path), redirect_stdout(output):
                exit_code = desktop_main.main(["--self-check"])
        result = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(result["frontend_assets"])
        self.assertFalse(result["debug"])
        self.assertTrue(result["worker_dispatcher"])

    def test_trace_worker_dispatches_without_starting_gui(self):
        with patch("desktop.trace_worker.main") as worker_main:
            exit_code = desktop_main.main(["--trace-worker", "request.json"])
        self.assertEqual(exit_code, 0)
        worker_main.assert_called_once_with("request.json")

    def test_invalid_arguments_return_usage_error(self):
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = desktop_main.main(["--unknown"])
        self.assertEqual(exit_code, 2)
        self.assertIn("Usage:", error.getvalue())

    def test_release_host_uses_bundled_assets_and_internal_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.html"
            index_path.write_text("<html></html>", encoding="utf-8")
            calls: dict = {}

            def create_window(title, url, js_api):
                calls["window"] = (title, url, js_api)
                return SimpleNamespace(events=SimpleNamespace(closed=_Event()))

            def start(**kwargs):
                calls["start"] = kwargs

            webview = SimpleNamespace(create_window=create_window, start=start)
            with patch.dict(sys.modules, {"webview": webview}), patch("desktop.host.frontend_index_path", return_value=index_path):
                host_main(development=False)

        self.assertEqual(calls["window"][0:2], ("NetViz", str(index_path)))
        self.assertEqual(calls["start"], {"debug": False, "http_server": True})


if __name__ == "__main__":
    unittest.main()
