import traceback
from typing import Any

PROTOCOL_VERSION = 1


def trace_error(
    run_id: str | None,
    code: str,
    title: str,
    message: str,
    stage: str,
    details: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "run_id": run_id,
        "error": {
            "code": code,
            "title": title,
            "message": message,
            "stage": stage,
            "details": details or {},
            "traceback": "".join(traceback.format_exception(exc)) if exc else None,
        },
    }


def trace_success(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def validate_worker_result(value: Any, expected_run_id: str, stderr: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return trace_error(
            expected_run_id,
            "worker_protocol_error",
            "Trace worker returned an invalid protocol message",
            "The worker result was not a JSON object.",
            "worker_protocol",
            {"stderr": stderr},
        )

    if value.get("protocol_version") != PROTOCOL_VERSION:
        return trace_error(
            expected_run_id,
            "worker_protocol_error",
            "Trace worker returned an unsupported protocol version",
            "The worker result does not match the supported trace protocol version.",
            "worker_protocol",
            {"protocol_version": value.get("protocol_version"), "stderr": stderr},
        )

    if value.get("run_id") != expected_run_id:
        return trace_error(
            expected_run_id,
            "worker_protocol_error",
            "Trace worker returned a stale result",
            "The worker result run ID did not match the active trace run.",
            "worker_protocol",
            {"actual_run_id": value.get("run_id"), "stderr": stderr},
        )

    message_type = value.get("type")
    if message_type == "success":
        trace = value.get("trace")
        if not isinstance(trace, dict):
            return trace_error(
                expected_run_id,
                "worker_protocol_error",
                "Trace worker returned an incomplete success result",
                "The worker success message did not include the required trace transfer fields.",
                "worker_protocol",
                {"stderr": stderr},
            )

        warnings = value.get("warnings")
        if not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings):
            return trace_error(
                expected_run_id,
                "worker_protocol_error",
                "Trace worker returned an incomplete success result",
                "The worker success message did not include a warnings list.",
                "worker_protocol",
                {"stderr": stderr},
            )

        transfer = trace.get("transfer")
        if transfer == "inline":
            payload = trace.get("payload")
            if not _is_trace_payload(payload):
                return trace_error(
                    expected_run_id,
                    "worker_protocol_error",
                    "Trace worker returned an invalid inline trace",
                    "The inline trace payload must contain a model name and graph node and edge arrays.",
                    "worker_protocol",
                    {"stderr": stderr},
                )
        elif transfer == "file":
            path = trace.get("path")
            if not isinstance(path, str) or not path:
                return trace_error(
                    expected_run_id,
                    "worker_protocol_error",
                    "Trace worker returned an invalid file trace",
                    "A file trace transfer must include a non-empty path.",
                    "worker_protocol",
                    {"stderr": stderr},
                )
        else:
            return trace_error(
                expected_run_id,
                "worker_protocol_error",
                "Trace worker returned an unsupported transfer type",
                "The worker trace transfer must be either inline or file.",
                "worker_protocol",
                {"transfer": transfer, "stderr": stderr},
            )

        return value

    if message_type == "error":
        error = value.get("error")
        required_error_fields = ("code", "title", "message", "stage")
        if not isinstance(error, dict) or any(not isinstance(error.get(field), str) for field in required_error_fields):
            return trace_error(
                expected_run_id,
                "worker_protocol_error",
                "Trace worker returned an incomplete error result",
                "The worker error message did not include the required structured error fields.",
                "worker_protocol",
                {"stderr": stderr},
            )

        details = error.get("details")
        if stderr and isinstance(details, dict) and "stderr" not in details:
            details["stderr"] = stderr
        return value

    return trace_error(
        expected_run_id,
        "worker_protocol_error",
        "Trace worker returned an unsupported message type",
        "The worker result message type was not recognized.",
        "worker_protocol",
        {"message_type": message_type, "stderr": stderr},
    )


def _is_trace_payload(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("model_name"), str):
        return False
    graph = value.get("graph")
    return isinstance(graph, dict) and isinstance(graph.get("nodes"), list) and isinstance(graph.get("edges"), list)
