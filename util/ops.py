
import torch.nn as nn
from util.util import jsonable
import torch 
import torch.fx as fx
import operator
import builtins
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


# Helpers for determining get_attrs and getitems
def is_getitem_node(node: fx.Node):
    return (
        node.op == "call_function"
        and node.target == operator.getitem
    )


def is_getattr_node(node: fx.Node):
    return (
        node.op == "call_function"
        and node.target == builtins.getattr
    )


def is_shape_helper_node(node: fx.Node):
    """
    True for nodes like:
      getattr(x, "shape")
      x.shape[0]
      x.size(0)
    which usually exist only to feed reshape/view.
    """
    if is_getattr_node(node):
        return True

    if is_getitem_node(node):
        src = node.args[0] if node.args else None

        if isinstance(src, fx.Node):
            # getattr(x, "shape")[0]
            if is_getattr_node(src):
                return True

            # chained indexing or slicing
            if is_getitem_node(src):
                return True

    if node.op == "call_method" and node.target in {"size", "dim", "numel"}:
        return True

    return False


def is_tensor_slice_node(node: fx.Node):
    """
    True for x[..., start:end]-style FX getitem nodes.

    These are semantically more important than shape helpers,
    but often should be inlined into pad/cat attrs instead of drawn as nodes.
    """
    if not is_getitem_node(node):
        return False

    index_spec = node.args[1] if len(node.args) > 1 else None

    def contains_slice(v):
        if isinstance(v, slice):
            return True
        if isinstance(v, tuple):
            return any(contains_slice(x) for x in v)
        return False

    return contains_slice(index_spec)


def is_transparent_helper_node(node: fx.Node):
    return is_shape_helper_node(node) or is_tensor_slice_node(node)


def is_shape_getattr_node(node: fx.Node):
    return (
        is_getattr_node(node)
        and len(node.args) >= 2
        and node.args[1] == "shape"
    )


def is_shape_metadata_node(node: fx.Node):
    """
    True for:
      x.shape
      x.shape[0]
      x.size(0)
      x.dim()
      x.numel()

    These should appear in node.inputs as shape/scalar metadata,
    but usually should NOT create graph edges.
    """
    if is_shape_getattr_node(node):
        return True

    if is_getitem_node(node):
        src = node.args[0] if node.args else None
        return isinstance(src, fx.Node) and is_shape_metadata_node(src)

    if node.op == "call_method" and node.target in {"size", "dim", "numel"}:
        return True

    return False

# Determine whether a node is a transparent helper node.

def is_tensor_index_getitem_node(node: fx.Node):
    """
    True for real tensor extraction/view ops like:
      qkv[0]
      qkv[1]
      qkv[2]
      x[..., start:end]

    These are tensor-flow, not metadata-flow.
    """
    return is_getitem_node(node) and not is_shape_metadata_node(node)

def should_inline_node(node: fx.Node):
    """
    Nodes that should not be drawn as graph nodes.

    Shape metadata nodes:
      x.shape
      x.shape[0]

    Tensor extraction nodes:
      qkv[0], qkv[1], qkv[2]
      x[..., start:end]

    These are represented as transforms on consumer inputs/edges.
    """
    if is_shape_metadata_node(node):
        return True

    if is_tensor_index_getitem_node(node):
        return True

    return False


def format_slice_spec(spec):
    if spec is Ellipsis:
        return "..."

    if isinstance(spec, slice):
        return {
            "start": jsonable(spec.start),
            "stop": jsonable(spec.stop),
            "step": jsonable(spec.step),
        }

    if isinstance(spec, tuple):
        return [format_slice_spec(s) for s in spec]

    return jsonable(spec)


def resolve_semantic_source(node: fx.Node):
    """
    Returns:
      {
        "source": str | None,
        "source_output": int,
        "transforms": list,
        "creates_edge": bool,
        "role": str | None,
      }

    Important:
    - Shape-derived nodes do NOT create graph edges by default.
    - Tensor getitem/slice nodes DO create graph edges.
    """
    transforms = []
    cur = node
    metadata_flow = False

    while isinstance(cur, fx.Node):
        # x.shape
        if is_shape_getattr_node(cur):
            base = cur.args[0]
            transforms.append({
                "op": "getattr",
                "attr": "shape",
            })
            cur = base
            metadata_flow = True
            continue

        # x.shape[0], x.shape[1], etc.
        if is_getitem_node(cur) and is_shape_metadata_node(cur):
            base = cur.args[0]
            index = cur.args[1] if len(cur.args) > 1 else None

            transforms.append({
                "op": "getitem",
                "index": jsonable(index),
            })

            cur = base
            metadata_flow = True
            continue

        # x.size(0), x.dim(), x.numel()
        if cur.op == "call_method" and cur.target in {"size", "dim", "numel"}:
            base = cur.args[0]
            transforms.append({
                "op": str(cur.target),
                "args": jsonable(cur.args[1:]),
                "kwargs": jsonable(cur.kwargs),
            })
            cur = base
            metadata_flow = True
            continue

        # handles tensor indexing (qkv[0], qkv[1], qkv[2])
        if is_tensor_index_getitem_node(cur):
            base = cur.args[0]
            index_spec = cur.args[1] if len(cur.args) > 1 else None

            transforms.append({
                "op": "getitem",
                "index": format_slice_spec(index_spec),
            })

            cur = base
            continue

        break

    if isinstance(cur, fx.Node):
        return {
            "source": cur.name,
            "source_output": 0,
            "transforms": list(reversed(transforms)),
            "creates_edge": not metadata_flow,
            "role": "shape" if metadata_flow else None,
        }

    return {
        "source": None,
        "source_output": 0,
        "transforms": list(reversed(transforms)),
        "creates_edge": False,
        "role": "shape" if metadata_flow else None,
    }