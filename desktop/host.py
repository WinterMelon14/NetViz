import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from desktop.source_handles import SourceHandles
from desktop.trace_protocol import (
    PROTOCOL_VERSION,
    trace_error,
    validate_worker_result,
)
from desktop.user_trace_constants import (
    CANCEL_TERMINATION_TIMEOUT_SECONDS,
    DEFAULT_TRACE_TIMEOUT_SECONDS,
    MAX_DIAGNOSTIC_BYTES,
    MAX_PROTOCOL_OUTPUT_BYTES,
    MAX_REMEMBERED_CANCELLED_RUNS,
    MAX_TRACE_FILE_BYTES,
    TRACE_FILE_TTL_SECONDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_SERVER_URL = "http://localhost:5173/"
WorkerCommandFactory = Callable[[Path], list[str]]


@dataclass
class ActiveTraceRun:
    run_id: str
    process: subprocess.Popen[str] | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    request_path: Path | None = None
    output_path: Path | None = None
    cancel_requested: bool = False


@dataclass
class CompletedTraceFile:
    run_id: str
    temp_dir: tempfile.TemporaryDirectory[str]
    path: Path
    expires_at: float
    timer: threading.Timer | None = None


def read_bounded_text(path: Path, max_bytes: int) -> tuple[str, bool]:
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    exceeded = len(content) > max_bytes
    return content[:max_bytes].decode("utf-8", errors="replace"), exceeded


def default_worker_command(request_path: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--trace-worker", str(request_path)]
    return [sys.executable, "-m", "desktop", "--trace-worker", str(request_path)]


def frontend_index_path() -> Path:
    return REPO_ROOT / "frontend" / "dist" / "index.html"


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=CANCEL_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=CANCEL_TERMINATION_TIMEOUT_SECONDS)


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
        self._completed_trace_files: dict[str, CompletedTraceFile] = {}
        self._cancelled_run_ids: list[str] = []

    def _remember_cancelled(self, run_id: str) -> None:
        if run_id in self._cancelled_run_ids:
            return
        self._cancelled_run_ids.append(run_id)
        del self._cancelled_run_ids[:-MAX_REMEMBERED_CANCELLED_RUNS]

    def run_user_trace(self, request: Any) -> dict[str, Any]:
        active_run_id = request.get("run_id") if isinstance(request, dict) else None
        if not active_run_id:
            return trace_error(
                None,
                "invalid_run_id",
                "Trace run could not start",
                "The frontend did not provide a valid run ID.",
                "host_lifecycle",
            )
        return self._run_trace_request(active_run_id, request)

    def _run_trace_request(self, active_run_id: str, bridge_request: dict[str, Any]) -> dict[str, Any]:

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
            active_run = ActiveTraceRun(active_run_id)
            self._active_runs[active_run_id] = active_run

        try:
            temp_dir = tempfile.TemporaryDirectory(prefix="tensor-trace-run-", dir=self.temp_root)
            request_path = Path(temp_dir.name) / "request.json"
            output_path = Path(temp_dir.name) / "result.json"
            protocol_path = Path(temp_dir.name) / "protocol.jsonl"
            diagnostic_path = Path(temp_dir.name) / "diagnostic.log"
            active_run.temp_dir = temp_dir
            active_run.request_path = request_path
            active_run.output_path = output_path
            worker_request = dict(bridge_request)
            worker_request.update({
                "protocol_version": PROTOCOL_VERSION,
                "run_id": active_run_id,
                "output_path": str(output_path),
            })
            request_path.write_text(
                json.dumps(worker_request),
                encoding="utf-8",
            )

            with self._lock:
                if active_run.cancel_requested:
                    self._active_runs.pop(active_run_id, None)
                    temp_dir.cleanup()
                    return trace_error(
                        active_run_id,
                        "cancelled",
                        "Trace cancelled",
                        "The trace run was cancelled before its worker started.",
                        "worker_cancelled",
                    )
                protocol_stream = protocol_path.open("wb")
                diagnostic_stream = diagnostic_path.open("wb")
                try:
                    process = subprocess.Popen(
                        self.worker_command_factory(request_path),
                        cwd=REPO_ROOT,
                        stdout=protocol_stream,
                        stderr=diagnostic_stream,
                    )
                except Exception:
                    protocol_stream.close()
                    diagnostic_stream.close()
                    raise
                active_run.process = process
        except Exception as exc:
            with self._lock:
                self._active_runs.pop(active_run_id, None)
            if active_run.temp_dir:
                active_run.temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "worker_start_failed",
                "Trace worker could not start",
                str(exc),
                "host_lifecycle",
                exc=exc,
            )

        timed_out = False
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process)
        finally:
            protocol_stream.close()
            diagnostic_stream.close()
            with self._lock:
                removed_run = self._active_runs.pop(active_run_id, None)
            cancelled = bool(removed_run and removed_run.cancel_requested)
            if cancelled:
                self._remember_cancelled(active_run_id)

        stdout, stdout_exceeded = read_bounded_text(protocol_path, MAX_PROTOCOL_OUTPUT_BYTES)
        stderr, stderr_exceeded = read_bounded_text(diagnostic_path, MAX_DIAGNOSTIC_BYTES)
        if stderr_exceeded:
            stderr += "\n[diagnostic output truncated]"

        if timed_out:
            temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "timeout",
                "Trace timed out",
                "The trace worker exceeded the configured timeout and was terminated.",
                "worker_timeout",
                {"timeout_seconds": self.timeout_seconds, "stderr": stderr},
            )

        if cancelled:
            temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "cancelled",
                "Trace cancelled",
                "The trace worker was cancelled before it completed.",
                "worker_cancelled",
            )

        if process.returncode != 0:
            temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "worker_crashed",
                "Trace worker failed",
                "The trace process exited before producing a successful result.",
                "worker_execution",
                {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr},
            )

        if stdout_exceeded:
            temp_dir.cleanup()
            return trace_error(
                active_run_id,
                "worker_protocol_error",
                "Trace worker protocol output is too large",
                "The worker exceeded the configured protocol output limit.",
                "worker_protocol",
                {"max_bytes": MAX_PROTOCOL_OUTPUT_BYTES, "stderr": stderr},
            )

        result = self._parse_worker_stdout(active_run_id, stdout, stderr)
        trace = result.get("trace") if result.get("type") == "success" else None
        if isinstance(trace, dict) and trace.get("transfer") == "file":
            if Path(trace["path"]) != output_path or not output_path.is_file():
                temp_dir.cleanup()
                return trace_error(
                    active_run_id,
                    "worker_protocol_error",
                    "Trace worker returned an invalid result path",
                    "The file-backed result did not match the host-provided output path.",
                    "worker_protocol",
                )
            actual_size = output_path.stat().st_size
            if actual_size != trace.get("size_bytes") or actual_size > MAX_TRACE_FILE_BYTES:
                temp_dir.cleanup()
                return trace_error(
                    active_run_id,
                    "trace_too_large",
                    "Trace result size is invalid",
                    "The file-backed trace size did not match the protocol or exceeded the configured limit.",
                    "worker_transport",
                    {"size_bytes": actual_size, "max_bytes": MAX_TRACE_FILE_BYTES},
                )
            self._retain_trace_file(active_run_id, temp_dir, output_path)
            return result

        temp_dir.cleanup()
        return result

    def _retain_trace_file(self, run_id: str, temp_dir: tempfile.TemporaryDirectory[str], path: Path) -> None:
        completed = CompletedTraceFile(run_id, temp_dir, path, time.monotonic() + TRACE_FILE_TTL_SECONDS)
        timer = threading.Timer(TRACE_FILE_TTL_SECONDS, self._expire_trace_file, args=(run_id,))
        timer.daemon = True
        completed.timer = timer
        with self._lock:
            self._completed_trace_files[run_id] = completed
        timer.start()

    def _expire_trace_file(self, run_id: str) -> None:
        with self._lock:
            completed = self._completed_trace_files.pop(run_id, None)
        if completed:
            completed.temp_dir.cleanup()

    def consume_trace_file(self, run_id: str, path: str) -> dict[str, Any]:
        with self._lock:
            completed = self._completed_trace_files.pop(run_id, None)
        if not completed or str(completed.path) != path or time.monotonic() > completed.expires_at:
            if completed:
                completed.timer.cancel() if completed.timer else None
                completed.temp_dir.cleanup()
            return trace_error(
                run_id,
                "trace_file_unavailable",
                "Trace file is unavailable",
                "The file-backed trace was already consumed, expired, or did not match this run.",
                "host_transport",
            )

        if completed.timer:
            completed.timer.cancel()
        try:
            payload_text, exceeded = read_bounded_text(completed.path, MAX_TRACE_FILE_BYTES)
            if exceeded:
                return trace_error(
                    run_id,
                    "trace_too_large",
                    "Trace result is too large",
                    "The file-backed trace exceeds the configured consumption limit.",
                    "host_transport",
                )
            payload = json.loads(payload_text)
            return {"ok": True, "run_id": run_id, "payload": payload}
        except Exception as exc:
            return trace_error(
                run_id,
                "trace_file_invalid",
                "Trace file could not be loaded",
                str(exc),
                "host_transport",
                exc=exc,
            )
        finally:
            completed.temp_dir.cleanup()

    def cancel_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            active_run = self._active_runs.get(run_id)
            if not active_run:
                if run_id in self._cancelled_run_ids:
                    return trace_error(
                        run_id,
                        "cancelled",
                        "Trace cancelled",
                        "The trace worker was already cancelled.",
                        "worker_cancelled",
                    )
                return trace_error(
                    run_id,
                    "run_not_found",
                    "Trace run was not active",
                    "No active trace worker matched the requested run ID.",
                    "host_lifecycle",
                )
            active_run.cancel_requested = True
            self._remember_cancelled(run_id)
            process = active_run.process

        if process is not None:
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
    def __init__(self, manager: TraceRunManager | None = None, selected_files: SourceHandles | None = None):
        self.manager = manager or TraceRunManager()
        self.sources = selected_files or SourceHandles()

    def runUserTrace(self, request: Any):
        resolved = self.sources.trace_request(request)
        if resolved.get("type") == "error":
            return resolved
        source_path = resolved["source"]["file_path"]
        source_id = request.get("source", {}).get("source_id") if isinstance(request, dict) else None
        display_name = self.sources.display_name(source_id)
        result = self.manager.run_user_trace(resolved)
        if result.get("type") == "error" and display_name:
            error = result.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                error["message"] = error["message"].replace(source_path, display_name)
        return result

    def selectPythonFile(self):
        return self.sources.select()

    def registerInlinePythonSource(self, request: Any):
        return self.sources.register_inline(request)

    def inspectPythonSource(self, sourceId: Any):
        return self.sources.inspect(sourceId)

    def releasePythonSource(self, sourceId: Any):
        return self.sources.release(sourceId)

    def cancelTrace(self, runId: str):
        return self.manager.cancel_trace(runId)

    def consumeTraceFile(self, runId: str, path: str):
        return self.manager.consume_trace_file(runId, path)

    def close(self):
        self.sources.close()

def main(development: bool = False):
    frontend_url: str
    if development:
        frontend_url = DEV_SERVER_URL
    else:
        index_path = frontend_index_path()
        if not index_path.is_file():
            raise SystemExit(
                "NetViz frontend assets are missing. Reinstall NetViz or rebuild "
                "the frontend before launching the desktop application."
            )
        frontend_url = str(index_path)

    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "NetViz could not load its desktop webview runtime. Reinstall NetViz "
            "and confirm that Microsoft Edge WebView2 Runtime is available."
        ) from exc

    api = DesktopTraceApi()
    window = webview.create_window(
        "NetViz",
        frontend_url,
        js_api=api,
    )
    window.events.closed += lambda *args: api.close()
    try:
        webview.start(debug=development, http_server=not development)
    except Exception as exc:
        api.close()
        raise SystemExit(
            "NetViz could not start its desktop window. Confirm that Microsoft "
            f"Edge WebView2 Runtime is installed. Technical detail: {exc}"
        ) from exc


if __name__ == "__main__":
    main(development="--dev" in sys.argv[1:])
