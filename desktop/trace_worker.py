import json
import sys
import traceback
import uuid
from pathlib import Path

PROTOCOL_VERSION = 1


def error_result(run_id: str, code: str, title: str, message: str, stage: str, exc: BaseException | None = None):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "run_id": run_id,
        "error": {
            "code": code,
            "title": title,
            "message": message,
            "stage": stage,
            "details": {},
            "traceback": "".join(traceback.format_exception(exc)) if exc else None,
        },
    }


def success_result(run_id: str, payload: dict):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "success",
        "run_id": run_id,
        "trace": {
            "transfer": "inline",
            "payload": payload,
        },
        "warnings": [],
    }


def load_request(path: str | None):
    if not path:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": str(uuid.uuid4()),
        }

    with Path(path).open("r", encoding="utf-8") as request_file:
        request = json.load(request_file)

    if not isinstance(request, dict) or request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported worker request protocol.")

    return request


def run_trace(request_path: str | None = None):
    try:
        request = load_request(request_path)
        run_id = str(request.get("run_id") or uuid.uuid4())
    except Exception as exc:
        run_id = str(uuid.uuid4())
        print(traceback.format_exc(), file=sys.stderr)
        return error_result(
            run_id,
            "worker_protocol_error",
            "Trace worker could not read its request",
            str(exc),
            "worker_protocol",
            exc,
        )

    try:
        from desktop.known_model import TestModel, known_model_input
        from util.summary import model_summary

        model = TestModel()
        example_input = known_model_input()
        payload = model_summary(model, example_input)
        return success_result(run_id, payload)
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        return error_result(
            run_id,
            "known_model_trace_failed",
            "Known model trace failed",
            str(exc),
            "worker_execution",
            exc,
        )


def main():
    request_path = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(run_trace(request_path)), flush=True)


if __name__ == "__main__":
    main()
