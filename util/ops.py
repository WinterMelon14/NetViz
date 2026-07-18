
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


def node_arg(node: fx.Node, position: int, keyword: str, default=None):
    if keyword in node.kwargs:
        return node.kwargs[keyword]
    if len(node.args) > position:
        return node.args[position]
    return default


def module_bias(module: nn.Module):
    return getattr(module, "bias", None) is not None


def callable_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def target_matches(target, names):
    return callable_name(target) in names


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
            "training": module.training,
            "bias": module_bias(module),
        }

    if isinstance(module, nn.LayerNorm):
        return {
            "normalized_shape": list(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
            "bias": module_bias(module),
        }

    if hasattr(nn, "RMSNorm") and isinstance(module, nn.RMSNorm):
        return {
            "normalized_shape": list(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
            "bias": False,
        }

    if isinstance(module, nn.GroupNorm):
        return {
            "num_groups": module.num_groups,
            "num_channels": module.num_channels,
            "eps": module.eps,
            "affine": module.affine,
            "bias": module_bias(module),
        }

    if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return {
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
            "training": module.training,
            "bias": module_bias(module),
        }

    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        return {
            "p": module.p,
            "inplace": module.inplace,
            "training": module.training,
        }

    if isinstance(module, nn.ReLU):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.LeakyReLU):
        return {
            "negative_slope": module.negative_slope,
            "inplace": module.inplace,
        }

    if isinstance(module, nn.ELU):
        return {
            "alpha": module.alpha,
            "inplace": module.inplace,
        }

    if isinstance(module, nn.SELU):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.Hardsigmoid):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.SiLU):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.Mish):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.Hardswish):
        return {
            "inplace": module.inplace,
        }

    if isinstance(module, nn.Softmax):
        return {
            "dim": module.dim,
        }

    if isinstance(module, nn.LogSoftmax):
        return {
            "dim": module.dim,
        }

    if isinstance(module, nn.GELU):
        return {
            "approximate": module.approximate,
        }

    if isinstance(module, nn.Softplus):
        return {
            "beta": module.beta,
            "threshold": module.threshold,
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
            "divisor_override": getattr(module, "divisor_override", None),
        }

    if isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        return {
            "output_size": jsonable(module.output_size),
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

    if isinstance(module, nn.Flatten):
        return {
            "start_dim": module.start_dim,
            "end_dim": module.end_dim,
        }

    if isinstance(module, nn.MultiheadAttention):
        return {
            "embed_dim": module.embed_dim,
            "num_heads": module.num_heads,
            "dropout": module.dropout,
            "batch_first": module.batch_first,
            "bias": module.in_proj_bias is not None,
            "add_bias_kv": module.bias_k is not None and module.bias_v is not None,
            "add_zero_attn": module.add_zero_attn,
            "kdim": module.kdim,
            "vdim": module.vdim,
            "training": module.training,
        }

    if isinstance(module, nn.TransformerEncoderLayer):
        return {
            "d_model": module.self_attn.embed_dim,
            "nhead": module.self_attn.num_heads,
            "dim_feedforward": module.linear1.out_features,
            "dropout": module.dropout.p,
            "activation": callable_name(module.activation),
            "batch_first": module.self_attn.batch_first,
            "norm_first": module.norm_first,
            "layer_norm_eps": module.norm1.eps,
            "bias": module.linear1.bias is not None,
            "training": module.training,
        }

    if isinstance(module, nn.TransformerDecoderLayer):
        return {
            "d_model": module.self_attn.embed_dim,
            "nhead": module.self_attn.num_heads,
            "dim_feedforward": module.linear1.out_features,
            "dropout": module.dropout.p,
            "activation": callable_name(module.activation),
            "batch_first": module.self_attn.batch_first,
            "norm_first": module.norm_first,
            "layer_norm_eps": module.norm1.eps,
            "bias": module.linear1.bias is not None,
            "training": module.training,
        }

    if isinstance(module, nn.TransformerEncoder):
        return {
            "num_layers": len(module.layers),
            "has_norm": module.norm is not None,
        }

    if isinstance(module, nn.TransformerDecoder):
        return {
            "num_layers": len(module.layers),
            "has_norm": module.norm is not None,
        }

    return {}


def attrs_for_node(node: fx.Node, runtime_output=None):
    attrs = {}

    if node.op == "call_method":
        if node.target == "permute":
            dims = node_arg(node, 1, "dims")
            attrs["dims"] = jsonable(dims if isinstance(dims, (list, tuple)) else node.args[1:])

        elif node.target in {"reshape", "view"}:
            attrs["requested_shape"] = [jsonable(a) for a in node.args[1:]]

            if isinstance(runtime_output, torch.Tensor):
                attrs["resolved_shape"] = list(runtime_output.shape)

        elif node.target == "transpose":
            dim0 = node_arg(node, 1, "dim0")
            dim1 = node_arg(node, 2, "dim1")
            attrs["dims"] = [jsonable(dim0), jsonable(dim1)]
            attrs["dim0"] = jsonable(dim0)
            attrs["dim1"] = jsonable(dim1)

        elif node.target == "flatten":
            attrs["start_dim"] = jsonable(node_arg(node, 1, "start_dim", 0))
            attrs["end_dim"] = jsonable(node_arg(node, 2, "end_dim", -1))

        elif node.target == "unsqueeze":
            attrs["dim"] = jsonable(node_arg(node, 1, "dim", 0))

        elif node.target == "expand":
            attrs["size"] = [jsonable(a) for a in node.args[1:]]

        elif node.target == "contiguous":
            attrs["memory_format"] = jsonable(node.kwargs.get("memory_format"))

        elif node.target == "chunk":
            attrs["chunks"] = jsonable(node_arg(node, 1, "chunks"))
            attrs["dim"] = jsonable(node_arg(node, 2, "dim", 0))

        elif node.target == "split":
            attrs["split_size_or_sections"] = jsonable(node_arg(node, 1, "split_size", node_arg(node, 1, "split_size_or_sections")))
            attrs["dim"] = jsonable(node_arg(node, 2, "dim", 0))

        elif node.target == "unbind":
            attrs["dim"] = jsonable(node_arg(node, 1, "dim", 0))

        elif node.target == "repeat":
            attrs["repeats"] = [jsonable(a) for a in node.args[1:]]

        elif node.target == "narrow":
            attrs["dim"] = jsonable(node_arg(node, 1, "dim"))
            attrs["start"] = jsonable(node_arg(node, 2, "start"))
            attrs["length"] = jsonable(node_arg(node, 3, "length"))

        elif node.target in {"mean", "sum", "max", "min", "std", "var", "norm"}:
            attrs["dim"] = jsonable(node_arg(node, 1, "dim"))
            attrs["keepdim"] = jsonable(node_arg(node, 2, "keepdim", False))
            if node.target == "norm":
                attrs["p"] = jsonable(node_arg(node, 1, "p", "fro"))
                attrs["dim"] = jsonable(node_arg(node, 2, "dim"))
                attrs["keepdim"] = jsonable(node_arg(node, 3, "keepdim", False))
            elif node.target in {"std", "var"}:
                attrs["unbiased"] = jsonable(node_arg(node, 2, "unbiased", True))
                attrs["keepdim"] = jsonable(node_arg(node, 3, "keepdim", False))

        elif node.target == "masked_fill":
            attrs["value"] = jsonable(node_arg(node, 2, "value"))

    elif node.op == "call_function":
        target = callable_name(node.target)

        if node.target in {torch.cat, torch.concat}:
            attrs["dim"] = jsonable(node_arg(node, 1, "dim", 0))

        elif node.target is torch.stack:
            attrs["dim"] = jsonable(node_arg(node, 1, "dim", 0))

        elif node.target is torch.flatten:
            attrs["start_dim"] = jsonable(node_arg(node, 1, "start_dim", 0))
            attrs["end_dim"] = jsonable(node_arg(node, 2, "end_dim", -1))

        elif node.target is torch.roll:
            attrs["shifts"] = jsonable(node_arg(node, 1, "shifts"))
            attrs["dims"] = jsonable(node_arg(node, 2, "dims"))

        elif node.target is torch.flip:
            attrs["dims"] = jsonable(node_arg(node, 1, "dims"))

        elif node.target is torch.add:
            attrs["alpha"] = jsonable(node.kwargs.get("alpha", 1))

        elif node.target in {torch.clamp, torch.clip} or target == "clamp":
            attrs["min"] = jsonable(node_arg(node, 1, "min"))
            attrs["max"] = jsonable(node_arg(node, 2, "max"))

        elif target in {"matmul", "mm", "bmm"}:
            attrs["operation"] = target

        elif target == "einsum":
            attrs["equation"] = jsonable(node_arg(node, 0, "equation"))

        elif target in {"sum", "mean", "max", "min", "norm", "std", "var"}:
            if target == "norm":
                attrs["p"] = jsonable(node_arg(node, 1, "p", "fro"))
                attrs["dim"] = jsonable(node_arg(node, 2, "dim"))
                attrs["keepdim"] = jsonable(node_arg(node, 3, "keepdim", False))
            else:
                attrs["dim"] = jsonable(node_arg(node, 1, "dim"))
                attrs["keepdim"] = jsonable(node_arg(node, 2, "keepdim", False))
                if target in {"std", "var"}:
                    attrs["unbiased"] = jsonable(node_arg(node, 2, "unbiased", True))
                    attrs["keepdim"] = jsonable(node_arg(node, 3, "keepdim", False))

        elif target == "where":
            attrs["condition"] = "condition"

        elif target == "masked_select":
            attrs["mask"] = "mask"

        elif target == "interpolate":
            attrs["size"] = jsonable(node_arg(node, 1, "size"))
            attrs["scale_factor"] = jsonable(node_arg(node, 2, "scale_factor"))
            attrs["mode"] = jsonable(node_arg(node, 3, "mode", "nearest"))
            attrs["align_corners"] = jsonable(node_arg(node, 4, "align_corners"))
            attrs["recompute_scale_factor"] = jsonable(node_arg(node, 5, "recompute_scale_factor"))
            attrs["antialias"] = jsonable(node_arg(node, 6, "antialias", False))

        elif node.target in {torch.nn.functional.pad, torch._C._nn.pad}:
            attrs["pad"] = jsonable(node_arg(node, 1, "pad"))
            attrs["mode"] = jsonable(node_arg(node, 2, "mode", "constant"))
            attrs["value"] = jsonable(node_arg(node, 3, "value", None))

        elif node.target in {torch.relu, torch.nn.functional.relu}:
            attrs["inplace"] = jsonable(node.kwargs.get("inplace", False))

        elif target == "scaled_dot_product_attention":
            attrs["dropout_p"] = jsonable(node_arg(node, 4, "dropout_p", 0.0))
            attrs["is_causal"] = jsonable(node_arg(node, 5, "is_causal", False))
            attrs["scale"] = jsonable(node.kwargs.get("scale"))
            attrs["enable_gqa"] = jsonable(node.kwargs.get("enable_gqa", False))
            attrs["has_attn_mask"] = node_arg(node, 3, "attn_mask") is not None

        elif target == "multi_head_attention_forward":
            attrs["embed_dim"] = jsonable(node_arg(node, 3, "embed_dim_to_check"))
            attrs["num_heads"] = jsonable(node_arg(node, 4, "num_heads"))
            attrs["add_zero_attn"] = jsonable(node_arg(node, 9, "add_zero_attn", False))
            attrs["dropout_p"] = jsonable(node_arg(node, 10, "dropout_p", 0.0))
            attrs["training"] = jsonable(node_arg(node, 13, "training", True))
            attrs["has_key_padding_mask"] = node_arg(node, 14, "key_padding_mask") is not None
            attrs["need_weights"] = jsonable(node_arg(node, 15, "need_weights", True))
            attrs["has_attn_mask"] = node_arg(node, 16, "attn_mask") is not None
            attrs["use_separate_proj_weight"] = jsonable(node_arg(node, 17, "use_separate_proj_weight", False))
            attrs["average_attn_weights"] = jsonable(node_arg(node, 24, "average_attn_weights", True))
            attrs["is_causal"] = jsonable(node_arg(node, 25, "is_causal", False))

        elif "attention" in target or "flash" in target:
            attrs["kwargs"] = jsonable(node.kwargs)

        elif node.kwargs:
            attrs["kwargs"] = jsonable(node.kwargs)

    return attrs


def formula_for(node: fx.Node, label: str, module_label: str | None = None, attrs: dict | None = None):
    name = module_label or label
    attrs = attrs or {}

    if name == "Linear":
        return "y = xW^T + b" if attrs.get("bias", True) else "y = xW^T"
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
    if name == "MultiheadAttention":
        return "project Q/K/V, apply scaled dot-product attention across heads, then project output"
    if name == "TransformerEncoderLayer":
        return "self-attention plus feed-forward block with residual connections and layer normalization"
    if name == "TransformerDecoderLayer":
        return "self-attention, cross-attention, and feed-forward block with residual connections and layer normalization"
    if name == "TransformerEncoder":
        return "stack of Transformer encoder layers"
    if name == "TransformerDecoder":
        return "stack of Transformer decoder layers"

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
        if node.target in {"max", "min", "norm", "std", "var"}:
            return f"reduce tensor by {node.target}"
        if node.target == "masked_fill":
            return "replace masked tensor elements with a fill value"

    if node.op == "call_function":
        target = callable_name(node.target)
        if node.target in {torch.cat, torch.concat}:
            return "concatenate tensors along a dimension"
        if node.target is torch.stack:
            return "stack tensors along a new dimension"
        if node.target is torch.flatten:
            return "flatten tensor dimensions"
        if node.target in {torch.relu, torch.nn.functional.relu}:
            return "y = max(0, x)"
        if target == "scaled_dot_product_attention":
            return "softmax((QK^T * scale) + mask) V"
        if target == "multi_head_attention_forward":
            return "project Q/K/V, run multi-head scaled dot-product attention, project output"
        if "attention" in target or "flash" in target:
            return "optimized scaled dot-product attention kernel"
        if node.target == operator.add:
            return "y = a + b"
        if node.target == operator.mul:
            return "y = a * b"
        if node.target == operator.sub:
            return "y = a - b"
        if node.target == operator.truediv:
            return "y = a / b"
        if target == "pow":
            return "y = x ** exponent"
        if target == "sqrt":
            return "y = sqrt(x)"
        if target == "exp":
            return "y = exp(x)"
        if target == "log":
            return "y = log(x)"
        if target == "abs":
            return "y = abs(x)"
        if target == "neg":
            return "y = -x"
        if target == "clamp":
            return "y = min(max(x, min), max)"
        if target in {"matmul", "mm", "bmm"}:
            return "matrix multiplication"
        if target == "einsum":
            return "Einstein summation over named dimensions"
        if target in {"sum", "mean", "max", "min", "norm", "std", "var"}:
            return f"reduce tensor by {target}"
        if target == "where":
            return "select elements from one of two tensors based on a condition"
        if target == "masked_select":
            return "select elements where a boolean mask is true"
        if target == "interpolate":
            return "resize spatial or temporal dimensions by interpolation"

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
