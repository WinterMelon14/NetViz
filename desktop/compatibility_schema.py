"""Contract helpers for versioned, nonexecuting compatibility reports."""

from typing import Any

from desktop.user_trace_constants import COMPATIBILITY_REPORT_VERSION

COMPATIBILITY_STATUSES = frozenset({"supported", "configuration_required", "unsupported", "unknown"})
FINDING_CATEGORIES = frozenset({"class", "constructor", "forward", "input", "import", "resource", "runtime", "fx"})
FINDING_ORIGINS = frozenset({"source", "heuristic", "runtime"})


class CompatibilityReportError(ValueError):
    pass


def finding(
    category: str,
    code: str,
    status: str,
    title: str,
    explanation: str,
    origin: str,
    evidence: list[dict[str, Any]],
    remediation: str | None = None,
    target: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": category,
        "code": code,
        "status": status,
        "title": title,
        "explanation": explanation,
        "origin": origin,
        "evidence": evidence,
    }
    if remediation:
        result["remediation"] = remediation
    if target:
        result["target"] = target
    return result


def source_evidence(text: str, line_number: int | None = None) -> list[dict[str, Any]]:
    item: dict[str, Any] = {"kind": "source", "text": text}
    if line_number is not None:
        item["lineNumber"] = line_number
    return [item]


def validate_compatibility_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise CompatibilityReportError("compatibility report must be an object")
    version = report.get("schemaVersion")
    if version != COMPATIBILITY_REPORT_VERSION:
        raise CompatibilityReportError(f"unsupported compatibility report version: {version!r}")
    if not isinstance(report.get("className"), str) or not report["className"]:
        raise CompatibilityReportError("className must be a non-empty string")
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise CompatibilityReportError("findings must be an array")
    for index, item in enumerate(findings):
        _validate_finding(item, index)
    outcome = report.get("tracingOutcome")
    if not isinstance(outcome, dict) or outcome.get("status") != "unknown" or not isinstance(outcome.get("statement"), str):
        raise CompatibilityReportError("tracingOutcome must state that execution remains unknown")
    return report


def _validate_finding(item: Any, index: int) -> None:
    path = f"findings[{index}]"
    if not isinstance(item, dict):
        raise CompatibilityReportError(f"{path} must be an object")
    for field in ("code", "title", "explanation"):
        if not isinstance(item.get(field), str) or not item[field]:
            raise CompatibilityReportError(f"{path}.{field} must be a non-empty string")
    if item.get("status") not in COMPATIBILITY_STATUSES:
        raise CompatibilityReportError(f"{path}.status is invalid")
    if item.get("category") not in FINDING_CATEGORIES:
        raise CompatibilityReportError(f"{path}.category is invalid")
    if item.get("origin") not in FINDING_ORIGINS:
        raise CompatibilityReportError(f"{path}.origin is invalid")
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise CompatibilityReportError(f"{path}.evidence must be a non-empty array")
    for evidence_item in evidence:
        if not isinstance(evidence_item, dict) or not isinstance(evidence_item.get("kind"), str) or not isinstance(evidence_item.get("text"), str):
            raise CompatibilityReportError(f"{path}.evidence contains an invalid item")
        line_number = evidence_item.get("lineNumber")
        if line_number is not None and (isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1):
            raise CompatibilityReportError(f"{path}.evidence lineNumber is invalid")
    target = item.get("target")
    if target is not None and (
        not isinstance(target, dict)
        or target.get("kind") not in {"constructor_parameter", "forward_parameter", "source"}
        or (target["kind"] != "source" and not isinstance(target.get("name"), str))
    ):
        raise CompatibilityReportError(f"{path}.target is invalid")
