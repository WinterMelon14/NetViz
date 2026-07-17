import platform
import statistics
import time
from typing import Any

import torch
import torch.fx as fx


class TimingInterpreter(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.durations_ns: dict[str, int] = {}

    def run_node(self, node: fx.Node):
        start_ns = time.perf_counter_ns()
        result = super().run_node(node)
        end_ns = time.perf_counter_ns()
        if node.op not in {"placeholder", "output"}:
            self.durations_ns[node.name] = end_ns - start_ns
        return result


def profile_graph_module(
    gm: fx.GraphModule,
    interpreter_args: list[Any],
    *,
    warmup_runs: int,
    measurement_runs: int,
    percentiles: list[int],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    gm.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            TimingInterpreter(gm).run(*interpreter_args)

        samples: dict[str, list[float]] = {}
        for _ in range(measurement_runs):
            interpreter = TimingInterpreter(gm)
            interpreter.run(*interpreter_args)
            for node_id, duration_ns in interpreter.durations_ns.items():
                samples.setdefault(node_id, []).append(duration_ns / 1_000_000)

    node_timings = [_node_timing(node, samples.get(node["id"], []), percentiles) for node in graph_nodes]
    observed = [item for item in node_timings if item["sample_count"] > 0]
    expensive = sorted(observed, key=lambda item: (-item["median_ms"], item["node_id"]))
    critical_path = _critical_path(node_timings, graph_edges)
    total_profiled_ms = sum(item["median_ms"] for item in observed)

    return {
        "schemaVersion": 1,
        "mode": "cpu",
        "config": {
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "percentiles": percentiles,
        },
        "environment": {
            "timer": "time.perf_counter_ns",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "semantics": {
            "duration": "wall-clock CPU duration measured around each FX node execution",
            "aggregation": "warmup samples are excluded; medians and percentiles are computed from measurement samples",
            "repeated_execution": warmup_runs + measurement_runs,
        },
        "total_profiled_ms": total_profiled_ms,
        "nodes": node_timings,
        "expensive_operations": expensive,
        "critical_path": critical_path,
    }


def _node_timing(node: dict[str, Any], values: list[float], percentiles: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "node_id": node["id"],
            "label": node.get("label", node["id"]),
            "kind": node.get("kind", "unknown"),
            "target": node.get("target", ""),
            "module_path": (node.get("module") or {}).get("path"),
            "sample_count": 0,
            "median_ms": None,
            "percentiles_ms": {},
        }
    sorted_values = sorted(values)
    return {
        "node_id": node["id"],
        "label": node.get("label", node["id"]),
        "kind": node.get("kind", "unknown"),
        "target": node.get("target", ""),
        "module_path": (node.get("module") or {}).get("path"),
        "sample_count": len(values),
        "median_ms": statistics.median(sorted_values),
        "percentiles_ms": {str(percentile): _percentile(sorted_values, percentile) for percentile in percentiles},
    }


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _critical_path(node_timings: list[dict[str, Any]], graph_edges: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {
        item["node_id"]: item["median_ms"]
        for item in node_timings
        if isinstance(item.get("median_ms"), (int, float))
    }
    node_ids = [item["node_id"] for item in node_timings]
    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in graph_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in outgoing and target in incoming:
            outgoing[source].append(target)
            incoming[target].append(source)

    best_cost: dict[str, float] = {}
    best_path: dict[str, list[str]] = {}
    for node_id in node_ids:
        predecessor_paths = [(best_cost[pred], best_path[pred]) for pred in incoming[node_id] if pred in best_cost]
        base_cost, base_path = max(predecessor_paths, default=(0.0, []), key=lambda item: item[0])
        weight = weights.get(node_id)
        if weight is None and node_id not in weights:
            if not predecessor_paths and outgoing[node_id]:
                continue
            weight = 0.0
        best_cost[node_id] = base_cost + float(weight)
        best_path[node_id] = [*base_path, node_id]

    if not best_cost:
        return {"node_ids": [], "total_ms": 0, "weight": "median_ms", "missing_timing_nodes": node_ids}

    end_node = max(best_cost, key=lambda node_id: (best_cost[node_id], node_id))
    missing = [node_id for node_id in best_path[end_node] if node_id not in weights]
    return {
        "node_ids": best_path[end_node],
        "total_ms": best_cost[end_node],
        "weight": "median_ms",
        "missing_timing_nodes": missing,
    }
