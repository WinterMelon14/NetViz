import torch.fx as fx 
from util.outputs import flatten_outputs 
from util.inputs import build_input_records
from util.ops import attrs_for_node
# ============================================================
# Interpreter
# ============================================================

class SummaryInterpreter(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, max_preview_items=8):
        super().__init__(gm)
        self.max_preview_items = max_preview_items

        self.node_runtime_attrs = {}
        self.node_records = {}
        self.events = []
        self.step = 0

    def run_node(self, node: fx.Node):
        runtime_args, runtime_kwargs = self.fetch_args_kwargs_from_env(node)

        result = super().run_node(node)

        inputs, input_refs = build_input_records(
            node,
            runtime_args,
            runtime_kwargs,
            max_preview_items=self.max_preview_items,
        )

        runtime_attrs = attrs_for_node(node, runtime_output=result)
        if runtime_attrs:
            self.node_runtime_attrs[node.name] = runtime_attrs

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
