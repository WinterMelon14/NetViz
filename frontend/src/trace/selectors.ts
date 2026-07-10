import type { TensorValue, TraceNode } from './types'

export function tensorValues(values: TensorValue[]) {
  return values.filter((value) => value.shape || value.summary)
}

export function primaryInput(node: TraceNode) {
  return tensorValues(node.inputs)[0]
}

export function primaryOutput(node: TraceNode) {
  return tensorValues(node.outputs)[0]
}
