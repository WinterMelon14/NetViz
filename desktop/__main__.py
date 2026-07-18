import json
import multiprocessing
import sys
from pathlib import Path

APP_NAME = "NetViz"
APP_VERSION = "1.0.1"


def _ensure_cli_streams() -> None:
    # The windowed PyInstaller bootloader leaves Python streams unset. CLI
    # modes inherit pipe handles from their parent, so wrap those handles before
    # writing worker protocol or diagnostic output.
    if sys.stdout is None:
        sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    if sys.stderr is None:
        sys.stderr = open(2, "w", encoding="utf-8", closefd=False)


def _usage_error(message: str) -> int:
    print(f"{message}\nUsage: NetViz.exe [--dev | --trace-worker REQUEST | --self-check]", file=sys.stderr)
    return 2


def _self_check() -> int:
    from desktop.host import default_worker_command, frontend_index_path

    frozen = bool(getattr(sys, "frozen", False))
    command = default_worker_command(Path("REQUEST.json"))
    result = {
        "ok": frontend_index_path().is_file(),
        "app": APP_NAME,
        "version": APP_VERSION,
        "frozen": frozen,
        "frontend_assets": frontend_index_path().is_file(),
        "development": False,
        "debug": False,
        "worker_dispatcher": "--trace-worker" in command,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] and result["worker_dispatcher"] else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from desktop.host import main as host_main

        host_main(development=False)
        return 0
    _ensure_cli_streams()
    if args == ["--dev"]:
        from desktop.host import main as host_main

        host_main(development=True)
        return 0
    if args == ["--self-check"]:
        return _self_check()
    if args and args[0] == "--trace-worker":
        if len(args) != 2:
            return _usage_error("--trace-worker requires exactly one request path.")
        from desktop.trace_worker import main as worker_main

        worker_main(args[1])
        return 0
    return _usage_error(f"Unknown NetViz argument: {' '.join(args)}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
