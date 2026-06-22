from util.util import tensor_num_bytes, human_bytes
import torch.fx as fx
import torch.nn as nn 
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
