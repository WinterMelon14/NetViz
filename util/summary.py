import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from util.Interpreter import SummaryInterpreter
from util.params import * 
from util.ops import *
from util.util import *
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

        if should_inline_node(node):
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

            module_path = str(node.target)
            module_reuse_count = module_target_counts.get(module_path, 1)
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

        # Edges come from the input refs, which preserve logical target input index
        for ref in record["input_refs"]:
            if not ref.get("creates_edge", True):
                continue
            source = ref["node"]
            source_output = ref["output"]
            target = node.name
            target_input = ref["target_input"]

            edge_id = f"{source}:{source_output}->{target}:{target_input}"

            edge = {
                "id": edge_id,
                "source": source,
                "source_output": source_output,
                "target": target,
                "target_input": target_input,
            }

            if ref.get("transforms"):
                edge["transforms"] = ref["transforms"]

            edges.append(edge)

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
