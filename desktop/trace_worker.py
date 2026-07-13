import json
import math
import sys
import traceback
import uuid
from pathlib import Path

from desktop.user_model_runtime import (
    UserTraceRuntimeError,
    build_tensor_inputs,
    build_provider_inputs,
    instantiate_model,
    load_sanitized_user_module,
)
from desktop.user_trace_request import UserTraceRequestError, validate_user_trace_request
from desktop.trace_protocol import (
    PROTOCOL_VERSION,
    trace_error,
    trace_file_success,
    trace_success,
)
from desktop.user_trace_constants import FLOAT32_BYTES, MAX_INLINE_TRACE_BYTES, MAX_TRACE_FILE_BYTES


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
        raise ValueError("A host-created request file is required.")

    with Path(path).open("r", encoding="utf-8") as request_file:
        request = json.load(request_file)

    return request


def input_error_details(input_specs: list[dict]) -> dict:
    return {
        "inputs": [{
            "index": index,
            "parameter_name": spec["parameter_name"],
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "generator": spec["generator"],
            "estimated_bytes": math.prod(spec["shape"]) * FLOAT32_BYTES,
        } for index, spec in enumerate(input_specs)]
    }


def run_trace(request_path: str | None = None):
    try:
        raw_request = load_request(request_path)
        run_id = str(raw_request.get("run_id") or uuid.uuid4()) if isinstance(raw_request, dict) else str(uuid.uuid4())
        expected_output_path = Path(request_path).parent / "result.json" if request_path else None
        request = validate_user_trace_request(raw_request, expected_output_path=expected_output_path)
    except UserTraceRequestError as exc:
        return trace_error(
            locals().get("run_id"),
            "user_trace_request_invalid",
            "Trace request is invalid",
            str(exc),
            "request_validation",
            {"path": exc.path},
        )
    except Exception as exc:
        run_id = str(uuid.uuid4())
        print(traceback.format_exc(), file=sys.stderr)
        return trace_error(
            run_id,
            "user_trace_request_invalid",
            "Trace worker could not read its request",
            str(exc),
            "request_validation",
            exc=exc,
        )

    try:
        with load_sanitized_user_module(
            request["source"]["file_path"],
            run_id,
            request["source"]["content_sha256"],
            Path(request_path).parent,
        ) as module:
            model = instantiate_model(
                module,
                request["source"]["class_name"],
                request["constructor"]["args"],
                request["constructor"]["kwargs"],
            )
            if request["input_provider"]:
                example_inputs, diagnostic_specs = build_provider_inputs(module, request["input_provider"])
            else:
                example_inputs = build_tensor_inputs(request["inputs"])
                diagnostic_specs = request["inputs"]
            from util.summary import model_summary

            payload = model_summary(model, *example_inputs, run_shape_prop=False)
            return trace_success_for_transport(run_id, payload, request.get("output_path"))
    except UserTraceRuntimeError as exc:
        return trace_error(
            run_id,
            exc.code,
            exc.title,
            exc.message,
            exc.stage,
            exc.details,
            exc,
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        return trace_error(
            run_id,
            "trace_execution_failed",
            "Model trace failed",
            str(exc),
            "forward_trace",
            input_error_details(locals().get("diagnostic_specs", request["inputs"])),
            exc=exc,
        )


def main():
    request_path = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(run_trace(request_path)), flush=True)


if __name__ == "__main__":
    main()
