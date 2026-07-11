import json
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

PROTOCOL_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_SERVER_URL = "http://localhost:5173/"


def protocol_error(code: str, title: str, message: str, stage: str, details: dict | None = None):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "run_id": None,
        "error": {
            "code": code,
            "title": title,
            "message": message,
            "stage": stage,
            "details": details or {},
            "traceback": None,
        },
    }


class DesktopTraceApi:
    def runKnownModelTrace(self):
        with tempfile.TemporaryDirectory(prefix="tensor-trace-spike-") as temp_dir:
            request_path = Path(temp_dir) / "request.json"
            request_path.write_text(json.dumps({
                "protocol_version": PROTOCOL_VERSION,
                "run_id": str(uuid4()),
                "model": "desktop.known_model.TestModel",
                "input": "torch.randn(1, 4)",
            }), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, "-m", "desktop.trace_worker", str(request_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            return protocol_error(
                "worker_crashed",
                "Trace worker failed",
                "The trace process exited before producing a successful result.",
                "worker_execution",
                {"exit_code": completed.returncode, "stderr": completed.stderr},
            )

        if not stdout:
            return protocol_error(
                "worker_protocol_error",
                "Trace worker returned no result",
                "The trace process exited without writing a protocol message.",
                "worker_protocol",
                {"stderr": completed.stderr},
            )

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return protocol_error(
                "worker_protocol_error",
                "Trace worker returned invalid JSON",
                str(exc),
                "worker_protocol",
                {"stdout": stdout, "stderr": completed.stderr},
            )

        if not isinstance(result, dict) or result.get("protocol_version") != PROTOCOL_VERSION:
            return protocol_error(
                "worker_protocol_error",
                "Trace worker returned an unsupported result",
                "The trace worker protocol version or message shape was not recognized.",
                "worker_protocol",
                {"stdout": stdout, "stderr": completed.stderr},
            )

        return result


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
