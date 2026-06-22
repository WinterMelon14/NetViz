from typing import Any
from util.tensor import tensor_value_record
from util.util import jsonable
from util.inference import infer_constant_role, is_tensor_list_value
import torch.fx as fx
import torch
from util.ops import resolve_semantic_source
# ============================================================
# Input flattening
# ============================================================

def flatten_node_runtime_inputs(
    spec: Any,
    value: Any,
    *,
    node_value_cache: dict[str, Any],
    max_preview_items: int,
    input_records: list[dict],
    input_refs: list[dict],
    container_role: str | None = None,
    arg_name: str | None = None,
):
    """
    Recursively align an FX arg spec with its runtime value.

    This is what lets torch.cat([x1, x2], dim=1) become:

    inputs: [
      {"index": 0, "from_node": "x1", ...},
      {"index": 1, "from_node": "x2", ...},
      {"index": 2, "role": "dim", "value": 1}
    ]

    and edges:
      x1:0 -> cat:0
      x2:0 -> cat:1
    """
    if isinstance(spec, fx.Node):
        resolved = resolve_semantic_source(spec)

        source_node = resolved["source"] or spec.name
        source_output = resolved["source_output"]
        transforms = resolved["transforms"]
        creates_edge = resolved.get("creates_edge", True)

        idx = len(input_records)

        if isinstance(value, torch.Tensor):
            rec = tensor_value_record(value, idx, max_preview_items=max_preview_items)
            rec["from"] = source_node
            rec["source_output"] = source_output
            rec["edge"] = creates_edge

            if transforms:
                rec["transforms"] = transforms

            if resolved.get("role") is not None:
                rec["role"] = resolved["role"]
            elif container_role == "tensor_list":
                rec["role"] = "tensor_list"

            input_records.append(rec)

            input_refs.append({
                "node": source_node,
                "output": source_output,
                "target_input": idx,
                "transforms": transforms,
                "creates_edge": creates_edge,
            })

        else:
            rec = {
                "index": idx,
                "role": resolved.get("role") or "unknown",
                "from": source_node,
                "source_output": source_output,
                "value": jsonable(value),
                "edge": creates_edge,
            }

            if transforms:
                rec["transforms"] = transforms

            input_records.append(rec)

            input_refs.append({
                "node": source_node,
                "output": source_output,
                "target_input": idx,
                "transforms": transforms,
                "creates_edge": creates_edge,
            })

        return
    if isinstance(spec, (list, tuple)):
        next_container_role = "tensor_list" if is_tensor_list_value(value) else container_role

        for i, child_spec in enumerate(spec):
            child_value = value[i] if isinstance(value, (list, tuple)) and i < len(value) else None

            flatten_node_runtime_inputs(
                child_spec,
                child_value,
                node_value_cache=node_value_cache,
                max_preview_items=max_preview_items,
                input_records=input_records,
                input_refs=input_refs,
                container_role=next_container_role,
                arg_name=arg_name,
            )

        return

    if isinstance(spec, dict):
        for key, child_spec in spec.items():
            child_value = value.get(key) if isinstance(value, dict) else None

            flatten_node_runtime_inputs(
                child_spec,
                child_value,
                node_value_cache=node_value_cache,
                max_preview_items=max_preview_items,
                input_records=input_records,
                input_refs=input_refs,
                container_role=container_role,
                arg_name=str(key),
            )

        return

    idx = len(input_records)

    input_records.append({
        "index": idx,
        "role": infer_constant_role(value, arg_name=arg_name),
        "value": jsonable(value),
    })


def build_input_records(
    node: fx.Node,
    runtime_args,
    runtime_kwargs,
    *,
    node_value_cache: dict[str, Any],
    max_preview_items: int,
):
    records = []
    refs = []

    # Positional args
    for i, spec in enumerate(node.args):
        value = runtime_args[i] if i < len(runtime_args) else None

        flatten_node_runtime_inputs(
            spec,
            value,
            node_value_cache=node_value_cache,
            max_preview_items=max_preview_items,
            input_records=records,
            input_refs=refs,
            arg_name=None,
        )

    # kwargs
    for key, spec in node.kwargs.items():
        value = runtime_kwargs.get(key)

        flatten_node_runtime_inputs(
            spec,
            value,
            node_value_cache=node_value_cache,
            max_preview_items=max_preview_items,
            input_records=records,
            input_refs=refs,
            arg_name=str(key),
        )

    return records, refs
