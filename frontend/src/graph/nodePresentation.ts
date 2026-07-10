import { primaryInput, primaryOutput } from '../trace/selectors'
import type { TraceNode } from '../trace/types'
import { nodeWidth } from './constants'

export function nodeCardWidth(node: TraceNode) {
  const inputDims = primaryInput(node)?.shape?.length ?? 1
  const outputDims = primaryOutput(node)?.shape?.length ?? 1
  const shapeFlowWidth = 72 + (inputDims + outputDims) * 24
  const titleWidth = 76 + node.label.length * 8

  return Math.max(nodeWidth, shapeFlowWidth, titleWidth)
}

export function totalParamLabel(node: TraceNode) {
  if (node.params?.count && node.params.count > 0) {
    return `${node.params.count.toLocaleString()} params`
  }
  return node.params?.memory?.human ?? 'no params'
}

export function kindBadge(node: TraceNode) {
  const labelByKind: Record<string, string> = {
    input: 'I',
    module: 'M',
    function: 'F',
    method: 'T',
  }

  return labelByKind[node.kind] ?? node.kind.slice(0, 1).toUpperCase()
}
