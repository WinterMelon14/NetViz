import json
import sys
import traceback
import uuid
from pathlib import Path

from desktop.trace_protocol import (
    MAX_INLINE_TRACE_BYTES,
    MAX_TRACE_FILE_BYTES,
    PROTOCOL_VERSION,
    trace_error,
    trace_file_success,
    trace_success,
)


def trace_success_for_transport(run_id: str, payload: dict, output_path: str | None):
    payload_text = json.dumps(payload, separators=(",", ":"))
    payload_size = len(payload_text.encode("utf-8"))
    if payload_size <= MAX_INLINE_TRACE_BYTES:
        return trace_success(run_id, payload)
    if payload_size > MAX_TRACE_FILE_BYTES:
        return trace_error(
            run_id,
            "trace_too_large",
            "Trace result is too large",
            "The trace exceeds the configured file-backed transport limit.",
            "worker_transport",
            {"size_bytes": payload_size, "max_bytes": MAX_TRACE_FILE_BYTES},
        )
    if not output_path:
        return trace_error(
            run_id,
            "worker_protocol_error",
            "Trace result path is unavailable",
            "The worker request did not provide an output path for a large trace.",
            "worker_transport",
        )

    destination = Path(output_path)
    pending = destination.with_suffix(".pending")
    pending.write_text(payload_text, encoding="utf-8")
    pending.replace(destination)
    return trace_file_success(run_id, str(destination), payload_size)


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
        return trace_error(
            run_id,
            "worker_protocol_error",
            "Trace worker could not read its request",
            str(exc),
            "worker_protocol",
            exc=exc,
        )

    try:
        from desktop.known_model import TestModel, known_model_input
        from util.summary import model_summary

        model = TestModel()
        example_input = known_model_input()
        payload = model_summary(model, example_input, run_shape_prop=False)
        return trace_success_for_transport(run_id, payload, request.get("output_path"))
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        return trace_error(
            run_id,
            "known_model_trace_failed",
            "Known model trace failed",
            str(exc),
            "worker_execution",
            exc=exc,
        )


def main():
    request_path = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(run_trace(request_path)), flush=True)


if __name__ == "__main__":
    main()
