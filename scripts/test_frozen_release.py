import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop.host import TraceRunManager


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bridge_request(source: Path, run_id: str, *, shape: list[int] | None = None) -> dict:
    return {
        "run_id": run_id,
        "source": {
            "file_path": str(source),
            "class_name": "UserModel",
            "content_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "constructor": {"args": [], "kwargs": {}},
        "inputs": [{
            "kind": "tensor",
            "parameter_name": "x",
            "shape": shape or [1, 4],
            "dtype": "float32",
            "generator": "random_normal",
        }],
    }


def frozen_manager(executable: Path, **kwargs) -> TraceRunManager:
    return TraceRunManager(
        worker_command_factory=lambda request: [str(executable), "--trace-worker", str(request)],
        **kwargs,
    )


def wait_for_active(manager: TraceRunManager, run_id: str) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        if run_id in manager._active_runs:
            return
        time.sleep(0.01)
    raise AssertionError(f"Frozen worker did not become active: {run_id}")


def run_self_check(executable: Path) -> None:
    completed = subprocess.run(
        [str(executable), "--self-check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    require(result.get("ok") is True, f"Frozen self-check failed: {result}")
    require(result.get("frozen") is True, f"Self-check did not report frozen mode: {result}")
    require(result.get("frontend_assets") is True, "Bundled frontend was not found.")
    require(result.get("debug") is False, "Release debug mode must be disabled.")
    require(result.get("worker_dispatcher") is True, "Frozen worker dispatcher is unavailable.")


def run_direct_worker(executable: Path, root: Path) -> None:
    source = root / "numpy_model.py"
    source.write_text(
        "import numpy as np\nimport torch\n"
        "class UserModel(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.scale = float(np.array([2.0], dtype=np.float32)[0])\n"
        "    def forward(self, x): return x * self.scale\n",
        encoding="utf-8",
    )
    request = bridge_request(source, "frozen-direct")
    request.update({"protocol_version": 1, "output_path": str(root / "result.json")})
    request_path = root / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [str(executable), "--trace-worker", str(request_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    require(result.get("type") == "success", f"Frozen direct worker failed: {result}\n{completed.stderr}")


def run_host_lifecycle(executable: Path, root: Path) -> None:
    numpy_source = root / "host_numpy_model.py"
    numpy_source.write_text(
        "import numpy as np\nimport torch\n"
        "class UserModel(torch.nn.Module):\n"
        "    def forward(self, x): return x + float(np.ones(1)[0])\n",
        encoding="utf-8",
    )
    manager = frozen_manager(executable, timeout_seconds=60)
    success = manager.run_user_trace(bridge_request(numpy_source, "frozen-host-success"))
    require(success.get("type") == "success", f"Frozen host/worker trace failed: {success}")

    unsupported = root / "unsupported_model.py"
    unsupported.write_text("import transformers\nclass UserModel: pass\n", encoding="utf-8")
    unavailable = manager.run_user_trace(bridge_request(unsupported, "frozen-unsupported"))
    require(unavailable.get("error", {}).get("code") == "module_import_failed", f"Unexpected unsupported import result: {unavailable}")

    crash = root / "crash_model.py"
    crash.write_text(
        "import os\ndef crash(value): os._exit(9)\n@crash\nclass UserModel: pass\n",
        encoding="utf-8",
    )
    crashed = manager.run_user_trace(bridge_request(crash, "frozen-crash"))
    require(crashed.get("error", {}).get("code") == "worker_crashed", f"Frozen crash was not contained: {crashed}")

    noisy = ROOT / "tests" / "fixtures" / "user_models" / "excessive_stderr.py"
    diagnostic = manager.run_user_trace(bridge_request(noisy, "frozen-stderr"))
    stderr = diagnostic.get("error", {}).get("details", {}).get("stderr", "")
    require("[diagnostic output truncated]" in stderr, "Frozen stderr was not bounded and marked as truncated.")

    sleeper = root / "sleep_model.py"
    sleeper.write_text(
        "import time\nimport torch\nclass UserModel(torch.nn.Module):\n"
        "    def forward(self, x): time.sleep(60); return x\n",
        encoding="utf-8",
    )
    timeout = frozen_manager(executable, timeout_seconds=0.2).run_user_trace(bridge_request(sleeper, "frozen-timeout"))
    require(timeout.get("error", {}).get("code") == "timeout", f"Frozen timeout failed: {timeout}")

    cancel_manager = frozen_manager(executable, timeout_seconds=60)
    cancelled: dict = {}
    thread = threading.Thread(
        target=lambda: cancelled.update(cancel_manager.run_user_trace(bridge_request(sleeper, "frozen-cancel")))
    )
    thread.start()
    wait_for_active(cancel_manager, "frozen-cancel")
    first = cancel_manager.cancel_trace("frozen-cancel")
    thread.join(timeout=10)
    second = cancel_manager.cancel_trace("frozen-cancel")
    require(not thread.is_alive(), "Frozen cancellation did not terminate the worker.")
    require(first.get("error", {}).get("code") == "cancelled", f"Cancellation failed: {first}")
    require(cancelled.get("error", {}).get("code") == "cancelled", f"Worker did not report cancellation: {cancelled}")
    require(second.get("error", {}).get("code") == "cancelled", f"Cancellation was not idempotent: {second}")


def run_file_transport(executable: Path) -> None:
    source = ROOT / "tests" / "fixtures" / "user_models" / "large_trace.py"
    manager = frozen_manager(executable, timeout_seconds=90)
    result = manager.run_user_trace(bridge_request(source, "frozen-file"))
    require(result.get("trace", {}).get("transfer") == "file", f"Frozen file transport was not selected: {result}")
    path = result["trace"]["path"]
    consumed = manager.consume_trace_file("frozen-file", path)
    consumed_again = manager.consume_trace_file("frozen-file", path)
    require(consumed.get("ok") is True, f"Frozen file result could not be consumed: {consumed}")
    require(consumed_again.get("error", {}).get("code") == "trace_file_unavailable", "Frozen file result was not single-use.")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: test_frozen_release.py DIST/NetViz/NetViz.exe")
    executable = Path(sys.argv[1]).resolve()
    if not executable.is_file():
        raise SystemExit(f"Frozen executable does not exist: {executable}")
    run_self_check(executable)
    with tempfile.TemporaryDirectory(prefix="netviz-frozen-test-") as temp_dir:
        root = Path(temp_dir)
        run_direct_worker(executable, root)
        run_host_lifecycle(executable, root)
    run_file_transport(executable)
    print("NetViz frozen release tests passed.")


if __name__ == "__main__":
    main()
