import json
import math
import operator
from typing import Any

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


# ============================================================
# Small utilities
# ============================================================

def human_bytes(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"


def tensor_num_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def jsonable(x: Any):
    if isinstance(x, fx.Node):
        return x.name

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return {
            "start": jsonable(x.start),
            "stop": jsonable(x.stop),
            "step": jsonable(x.step),
        }

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if hasattr(x, "__name__"):
        return x.__name__

    return str(x)


def target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def node_kind(node: fx.Node):
    if node.op == "placeholder":
        return "input"
    if node.op == "call_module":
        return "module"
    if node.op == "call_function":
        return "function"
    if node.op == "call_method":
        return "method"
    if node.op == "get_attr":
        return "attribute"
    if node.op == "output":
        return "output"
    return node.op


# ============================================================
# Tensor summaries
# ============================================================

def tensor_preview(t: torch.Tensor, max_items=8):
    with torch.no_grad():
        td = t.detach().cpu()
        flat = td.flatten()

        if flat.numel() == 0:
            return []

        return flat[:max_items].tolist()


def tensor_stats(t: torch.Tensor):
    with torch.no_grad():
        td = t.detach().cpu()
        flat = td.flatten()

        if flat.numel() == 0:
            return {
                "numel": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "zeros_pct": None,
                "has_nan": False,
                "has_inf": False,
            }

        stats = {
            "numel": int(flat.numel()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.float().mean()),
            "std": float(flat.float().std()) if flat.numel() > 1 else 0.0,
            "zeros_pct": 100.0* float((flat == 0).float().mean()),
        }

        if torch.is_floating_point(flat):
            stats["has_nan"] = bool(torch.isnan(flat).any())
            stats["has_inf"] = bool(torch.isinf(flat).any())
        else:
            stats["has_nan"] = False
            stats["has_inf"] = False

        return stats


def tensor_value_record(t: torch.Tensor, index: int, max_preview_items=8):
    num_bytes = tensor_num_bytes(t)

    return {
        "index": index,
        "role": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "preview": tensor_preview(t, max_items=max_preview_items),
        "summary": tensor_stats(t),
        "memory": {
            "num_bytes": num_bytes,
            "human": human_bytes(num_bytes),
        },
    }


# ============================================================
# Param summaries
# ============================================================

def module_params_summary(module: nn.Module):
    shapes = {}
    dtypes = {}
    count = 0
    num_bytes = 0

    for name, p in module.named_parameters(recurse=False):
        shapes[name] = list(p.shape)
        dtypes[name] = str(p.dtype)
        count += p.numel()
        num_bytes += tensor_num_bytes(p)

    return {
        "count": int(count),
        "shapes": shapes,
        "dtypes": dtypes,
        "memory": {
            "num_bytes": int(num_bytes),
            "human": human_bytes(num_bytes),
        },
    }


def model_param_stats(gm: fx.GraphModule):
    total = 0
    trainable = 0
    non_trainable = 0
    num_bytes = 0

    seen_param_ids = set()

    for _, p in gm.named_parameters():
        # Avoid double counting shared parameters.
        pid = id(p)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)

        n = p.numel()
        b = tensor_num_bytes(p)

        total += n
        num_bytes += b

        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(non_trainable),
        "total_param_memory": {
            "num_bytes": int(num_bytes),
            "human": human_bytes(num_bytes),
        },
    }


def input_specs_from_graph(nodes: list[dict]):
    specs = []

    for node in nodes:
        if node.get("kind") != "input":
            continue

        outputs = node.get("outputs", [])

        for out in outputs:
            if out.get("role") != "tensor":
                continue

            specs.append({
                "index": len(specs),
                "name": node["id"],
                "shape": out.get("shape"),
                "dtype": out.get("dtype"),
                "memory": out.get("memory"),
            })

    return specs


def activation_memory_stats(nodes: list[dict], *, include_inputs=False):
    total_bytes = 0

    for node in nodes:
        if not include_inputs and node.get("kind") == "input":
            continue

        for out in node.get("outputs", []):
            mem = out.get("memory")
            if not mem:
                continue

            total_bytes += int(mem.get("num_bytes", 0))

    return {
        "total_activation_memory": {
            "num_bytes": int(total_bytes),
            "human": human_bytes(total_bytes),
        }
    }


# ============================================================
# Role inference
# ============================================================

def infer_constant_role(value: Any, *, arg_name: str | None = None):
    if arg_name in {"dim", "axis"}:
        return "dim"

    if arg_name in {"size", "shape"}:
        return "shape"

    if isinstance(value, torch.Size):
        return "shape"

    if isinstance(value, (int, float, bool)):
        return "scalar"

    if isinstance(value, (tuple, list)):
        if all(isinstance(v, int) for v in value):
            return "shape"

    return "constant"


def is_tensor_list_value(value: Any):
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, torch.Tensor) for v in value)
    )


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
        source_node = spec.name
        source_output = 0

        idx = len(input_records)

        if isinstance(value, torch.Tensor):
            rec = tensor_value_record(value, idx, max_preview_items=max_preview_items)
            rec["from_node"] = source_node
            rec["source_output"] = source_output

            if container_role == "tensor_list":
                rec["role"] = "tensor_list"

            input_records.append(rec)

            input_refs.append({
                "node": source_node,
                "output": source_output,
                "target_input": idx,
            })

        else:
            rec = {
                "index": idx,
                "role": "unknown",
                "from_node": source_node,
                "source_output": source_output,
                "value": jsonable(value),
            }
            input_records.append(rec)

            input_refs.append({
                "node": source_node,
                "output": source_output,
                "target_input": idx,
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

    # Positional args.
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

    # Keyword args.
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


# ============================================================
# Output flattening
# ============================================================

def flatten_outputs(value: Any, *, max_preview_items: int):
    outputs = []

    def visit(v):
        idx = len(outputs)

        if isinstance(v, torch.Tensor):
            outputs.append(tensor_value_record(v, idx, max_preview_items=max_preview_items))
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        elif isinstance(v, dict):
            for item in v.values():
                visit(item)
        else:
            outputs.append({
                "index": idx,
                "role": "constant",
                "value": jsonable(v),
            })

    visit(value)
    return outputs


# ============================================================
# Operation attrs and formulas
# ============================================================
def pair(x):
    if isinstance(x, tuple):
        return list(x)
    return [x, x]


def module_attrs(module: nn.Module):
    if isinstance(module, nn.Linear):
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }

    if isinstance(module, nn.Conv1d):
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }

    if isinstance(module, nn.Conv2d):
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }

    if isinstance(module, nn.Conv3d):
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": module.groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return {
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
        }

    if isinstance(module, nn.LayerNorm):
        return {
            "normalized_shape": list(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
        }

    if isinstance(module, nn.Dropout):
        return {
            "p": module.p,
            "inplace": module.inplace,
        }

    if isinstance(module, nn.ReLU):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        return {
            "kernel_size": jsonable(module.kernel_size),
            "stride": jsonable(module.stride),
            "padding": jsonable(module.padding),
            "dilation": jsonable(module.dilation),
            "return_indices": module.return_indices,
            "ceil_mode": module.ceil_mode,
        }

    if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        return {
            "kernel_size": jsonable(module.kernel_size),
            "stride": jsonable(module.stride),
            "padding": jsonable(module.padding),
            "ceil_mode": module.ceil_mode,
            "count_include_pad": module.count_include_pad,
        }

    if isinstance(module, nn.Embedding):
        return {
            "num_embeddings": module.num_embeddings,
            "embedding_dim": module.embedding_dim,
            "padding_idx": module.padding_idx,
            "max_norm": module.max_norm,
            "norm_type": module.norm_type,
            "scale_grad_by_freq": module.scale_grad_by_freq,
            "sparse": module.sparse,
        }

    return {}


def attrs_for_node(node: fx.Node, runtime_output=None):
    attrs = {}

    if node.op == "call_method":
        if node.target == "permute":
            attrs["dims"] = [jsonable(a) for a in node.args[1:]]

        elif node.target in {"reshape", "view"}:
            attrs["requested_shape"] = [jsonable(a) for a in node.args[1:]]

            if isinstance(runtime_output, torch.Tensor):
                attrs["resolved_shape"] = list(runtime_output.shape)

        elif node.target == "transpose":
            attrs["dims"] = [jsonable(a) for a in node.args[1:]]

        elif node.target == "flatten":
            attrs["args"] = [jsonable(a) for a in node.args[1:]]
            attrs["kwargs"] = jsonable(node.kwargs)

    elif node.op == "call_function":
        if node.target in {torch.cat, torch.concat}:
            attrs["dim"] = jsonable(node.kwargs.get("dim", 0))

        elif node.target is torch.stack:
            attrs["dim"] = jsonable(node.kwargs.get("dim", 0))

        elif node.target is torch.flatten:
            attrs["start_dim"] = jsonable(node.kwargs.get("start_dim", 0))
            attrs["end_dim"] = jsonable(node.kwargs.get("end_dim", -1))

        elif node.kwargs:
            attrs["kwargs"] = jsonable(node.kwargs)

    return attrs


def formula_for(node: fx.Node, label: str, module_label: str | None = None):
    name = module_label or label

    if name == "Linear":
        return "y = xW^T + b"
    if name == "Conv1d":
        return "y = conv1d(x, W) + b"
    if name == "Conv2d":
        return "y = conv2d(x, W) + b"
    if name == "Conv3d":
        return "y = conv3d(x, W) + b"
    if name == "ReLU":
        return "y = max(0, x)"
    if name == "Sigmoid":
        return "y = 1 / (1 + exp(-x))"
    if name == "Tanh":
        return "y = tanh(x)"
    if name in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
        return "normalize using running/channel statistics"
    if name == "LayerNorm":
        return "normalize over the last D dimensions"
    if name == "Dropout":
        return "randomly zero elements during training"
    if name == "MaxPool2d":
        return "take maximum value over each pooling window"
    if name == "AvgPool2d":
        return "take average value over each pooling window"
    if name == "Embedding":
        return "lookup embedding vectors by index"

    if node.op == "call_method":
        if node.target == "permute":
            return "reorder tensor dimensions"
        if node.target in {"reshape", "view"}:
            return "change tensor shape without changing element values"
        if node.target == "transpose":
            return "swap tensor dimensions"
        if node.target == "flatten":
            return "flatten tensor dimensions"
        if node.target == "mean":
            return "reduce tensor by mean"
        if node.target == "sum":
            return "reduce tensor by sum"

    if node.op == "call_function":
        if node.target in {torch.cat, torch.concat}:
            return "concatenate tensors along a dimension"
        if node.target is torch.stack:
            return "stack tensors along a new dimension"
        if node.target is torch.flatten:
            return "flatten tensor dimensions"
        if node.target in {torch.relu, torch.nn.functional.relu}:
            return "y = max(0, x)"
        if node.target == operator.add:
            return "y = a + b"
        if node.target == operator.mul:
            return "y = a * b"
        if node.target == operator.sub:
            return "y = a - b"
        if node.target == operator.truediv:
            return "y = a / b"

    return None


# ============================================================
# Interpreter
# ============================================================

class SummaryInterpreter(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, max_preview_items=8):
        super().__init__(gm)
        self.max_preview_items = max_preview_items

        self.node_values = {}
        self.node_records = {}
        self.events = []
        self.step = 0

    def run_node(self, node: fx.Node):
        runtime_args, runtime_kwargs = self.fetch_args_kwargs_from_env(node)

        result = super().run_node(node)

        self.node_values[node.name] = result

        inputs, input_refs = build_input_records(
            node,
            runtime_args,
            runtime_kwargs,
            node_value_cache=self.node_values,
            max_preview_items=self.max_preview_items,
        )

        outputs = flatten_outputs(
            result,
            max_preview_items=self.max_preview_items,
        )

        self.node_records[node.name] = {
            "inputs": inputs,
            "outputs": outputs,
            "input_refs": input_refs,
        }

        if node.op != "output":
            self.events.append({
                "step": self.step,
                "phase": "forward",
                "event": "node_executed",
                "node": node.name,
                "inputs": [
                    {
                        "node": ref["node"],
                        "output": ref["output"],
                    }
                    for ref in input_refs
                ],
                "outputs": [
                    {
                        "node": node.name,
                        "output": out["index"],
                    }
                    for out in outputs
                ],
            })
            self.step += 1

        return result


# ============================================================
# Main API
# ============================================================

def model_summary(
    model_or_gm,
    *example_args,
    example_kwargs=None,
    max_preview_items=8,
    include_placeholders=True,
    include_output=False,
    run_shape_prop=True,
):
    example_kwargs = example_kwargs or {}

    if isinstance(model_or_gm, fx.GraphModule):
        gm = model_or_gm
        model_name = type(gm).__name__
    else:
        model = model_or_gm
        model_name = type(model).__name__
        gm = fx.symbolic_trace(model)

    gm.eval()

    if run_shape_prop:
        try:
            ShapeProp(gm).propagate(*example_args, **example_kwargs)
        except Exception:
            pass

    # Count module targets to detect module reuse.
    module_target_counts = {}

    for node in gm.graph.nodes:
        if node.op == "call_module":
            target = str(node.target)
            module_target_counts[target] = module_target_counts.get(target, 0) + 1

    interp = SummaryInterpreter(gm, max_preview_items=max_preview_items)

    with torch.no_grad():
        interp.run(*example_args, **example_kwargs)

    nodes = []
    edges = []

    for node in gm.graph.nodes:
        if node.op == "output" and not include_output:
            continue

        if node.op == "placeholder" and not include_placeholders:
            continue

        record = interp.node_records.get(node.name, {
            "inputs": [],
            "outputs": [],
            "input_refs": [],
        })

        mod = None
        mod_type = None
        params = None
        module_reused = None
        module_reuse_count = None

        if node.op == "call_module":
            mod = gm.get_submodule(str(node.target))
            mod_type = type(mod).__name__
            label = mod_type

            params = module_params_summary(mod)

            target = str(node.target)
            module_reuse_count = module_target_counts.get(target, 1)
            module_reused = module_reuse_count > 1

        else:
            label = target_name(node.target)

        info = {
            "id": node.name,
            "kind": node_kind(node),
            "label": label,
            "fx_op": node.op,
            "target": target_name(node.target),
            "inputs": record["inputs"],
            "outputs": record["outputs"],
        }

        if mod_type is not None:
            info["module"] = {
                "path": str(node.target),
                "type": mod_type,
                "is_reused": module_reused,
                "reuse_group": str(node.target),
                "reuse_count": module_reuse_count,
            }

        if params is not None:
            info["params"] = params

        attrs = {}

        if node.op == "call_module":
            mod = gm.get_submodule(str(node.target))
            mod_type = type(mod).__name__
            label = mod_type

            params = module_params_summary(mod)

            module_path = str(node.target)
            module_reuse_count = module_target_counts.get(module_path, 1)
            module_reused = module_reuse_count > 1

        else:
            label = target_name(node.target)

        if mod is not None:
            attrs.update(module_attrs(mod))

        attrs.update(attrs_for_node(
            node,
            runtime_output=interp.node_values.get(node.name),
        ))

        if attrs:
            info["attrs"] = attrs

        formula = formula_for(node, label=label, module_label=mod_type)
        if formula is not None:
            info["formula"] = formula

        nodes.append(info)

        # Edges come from the input refs, which preserve logical target input index.
        for ref in record["input_refs"]:
            source = ref["node"]
            source_output = ref["output"]
            target = node.name
            target_input = ref["target_input"]

            edge_id = f"{source}:{source_output}->{target}:{target_input}"

            edges.append({
                "id": edge_id,
                "source": source,
                "source_output": source_output,
                "target": target,
                "target_input": target_input,
            })

    param_stats = model_param_stats(gm)
    input_specs = input_specs_from_graph(nodes)
    activation_stats = activation_memory_stats(nodes, include_inputs=False)

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        **param_stats,
        **activation_stats,
        "input_specs": input_specs,
    }

    return {
        "model_name": model_name,
        "stats": stats,
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "events": interp.events,
    }