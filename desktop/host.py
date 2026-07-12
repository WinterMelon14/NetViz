import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from desktop.source_inspection import inspect_model_source_request
from desktop.trace_protocol import PROTOCOL_VERSION, trace_error, validate_worker_result

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_SERVER_URL = "http://localhost:5173/"
DEFAULT_TRACE_TIMEOUT_SECONDS = 20

WorkerCommandFactory = Callable[[Path], list[str]]


@dataclass
class ActiveTraceRun:
    run_id: str
    process: subprocess.Popen[str]
    temp_dir: tempfile.TemporaryDirectory[str]
    request_path: Path
    output_path: Path
    cancel_requested: bool = False


def default_worker_command(request_path: Path) -> list[str]:
    return [sys.executable, "-m", "desktop.trace_worker", str(request_path)]


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


class TraceRunManager:
    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TRACE_TIMEOUT_SECONDS,
        worker_command_factory: WorkerCommandFactory = default_worker_command,
        temp_root: Path | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.worker_command_factory = worker_command_factory
        self.temp_root = temp_root
        self._lock = threading.Lock()
        self._active_runs: dict[str, ActiveTraceRun] = {}

    def run_known_model_trace(self, run_id: str | None = None) -> dict[str, Any]:
        active_run_id = run_id or str(uuid4())
        if not active_run_id:
            return trace_error(
                None,
                "invalid_run_id",
                "Trace run could not start",
                "The frontend did not provide a valid run ID.",
                "host_lifecycle",
            )

        with self._lock:
            if self._active_runs:
                return trace_error(
                    active_run_id,
                    "duplicate_run_request",
                    "Trace already running",
                    "A trace worker is already active. Wait for it to finish or cancel it before starting another run.",
                    "host_lifecycle",
                    {"active_run_ids": list(self._active_runs.keys())},
                )

        temp_dir = tempfile.TemporaryDirectory(prefix="tensor-trace-run-", dir=self.temp_root)
        request_path = Path(temp_dir.name) / "request.json"
        output_path = Path(temp_dir.name) / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": active_run_id,
                    "model": "desktop.known_model.TestModel",
                    "input": "torch.randn(1, 4)",
                    "output_path": str(output_path),
                }
            ),
            encoding="utf-8",
        )

        try:
            process = subprocess.Popen(
                self.worker_command_factory(request_path),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "worker_start_failed",
                "Trace worker could not start",
                str(exc),
                "host_lifecycle",
                exc=exc,
            )

        active_run = ActiveTraceRun(active_run_id, process, temp_dir, request_path, output_path)
        with self._lock:
            self._active_runs[active_run_id] = active_run

        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_process(process)
            _, stderr = process.communicate()
            return trace_error(
                active_run_id,
                "timeout",
                "Trace timed out",
                "The trace worker exceeded the development timeout and was terminated.",
                "worker_timeout",
                {"timeout_seconds": self.timeout_seconds, "stderr": stderr},
            )
        finally:
            with self._lock:
                removed_run = self._active_runs.pop(active_run_id, None)
            cancelled = bool(removed_run and removed_run.cancel_requested)
            active_run.temp_dir.cleanup()

        if cancelled:
            return trace_error(
                active_run_id,
                "cancelled",
                "Trace cancelled",
                "The trace worker was cancelled before it completed.",
                "worker_cancelled",
            )

        if process.returncode != 0:
            return trace_error(
                active_run_id,
                "worker_crashed",
                "Trace worker failed",
                "The trace process exited before producing a successful result.",
                "worker_execution",
                {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr},
            )

        return self._parse_worker_stdout(active_run_id, stdout, stderr)

    def cancel_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            active_run = self._active_runs.get(run_id)
            if not active_run:
                return trace_error(
                    run_id,
                    "run_not_found",
                    "Trace run was not active",
                    "No active trace worker matched the requested run ID.",
                    "host_lifecycle",
                )
            active_run.cancel_requested = True
            process = active_run.process

        terminate_process(process)
        return trace_error(
            run_id,
            "cancelled",
            "Trace cancelled",
            "The trace worker was cancelled before it completed.",
            "worker_cancelled",
        )

    @staticmethod
    def _parse_worker_stdout(run_id: str, stdout: str, stderr: str) -> dict[str, Any]:
        messages = [line for line in stdout.splitlines() if line.strip()]
        if not messages:
            return trace_error(
                run_id,
                "worker_protocol_error",
                "Trace worker returned no result",
                "The trace process exited without writing a protocol message.",
                "worker_protocol",
                {"stderr": stderr},
            )

        if len(messages) > 1:
            return trace_error(
                run_id,
                "worker_protocol_error",
                "Trace worker returned multiple result messages",
                "The trace process wrote more than one protocol message.",
                "worker_protocol",
                {"stdout": stdout, "stderr": stderr},
            )

        try:
            result = json.loads(messages[0])
        except json.JSONDecodeError as exc:
            return trace_error(
                run_id,
                "worker_protocol_error",
                "Trace worker returned invalid JSON",
                str(exc),
                "worker_protocol",
                {"stdout": stdout, "stderr": stderr},
                exc,
            )

        return validate_worker_result(result, run_id, stderr)


class DesktopTraceApi:
    def __init__(self, manager: TraceRunManager | None = None):
        self.manager = manager or TraceRunManager()

    def runKnownModelTrace(self, runId: str | None = None):
        return self.manager.run_known_model_trace(runId)

    def cancelTrace(self, runId: str):
        return self.manager.cancel_trace(runId)

    def inspectModelSource(self, request: Any):
        return inspect_model_source_request(request)


def main():
    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "pywebview is not installed in this Python environment. "
            "Install pywebview, start the Vite dev server, then run this module again."
        ) from exc

    webview.create_window(
        "PyTorch Trace Visualizer",
        DEV_SERVER_URL,
        js_api=DesktopTraceApi(),
    )
    webview.start(debug=True)


if __name__ == "__main__":
    main()
