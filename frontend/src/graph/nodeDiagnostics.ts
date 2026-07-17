import type { TraceNode } from '../trace/types'
import { sparseDefaultThresholdPercent, sparseReluThresholdPercent } from './constants'

export type NodeDiagnostics = {
  hasNan: boolean
  hasInf: boolean
  sparsePercent: number | null
}

function isReluNode(node: TraceNode) {
  return [node.label, node.target, node.module?.type]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase().includes('relu'))
}

function isPadNode(node: TraceNode) {
  return [node.label, node.target, node.module?.type]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase().includes('pad'))
}

export function nodeDiagnostics(node: TraceNode): NodeDiagnostics {
  const summaries = node.outputs
    .map((output) => output.summary)
    .filter((summary) => summary !== undefined)
  if (isPadNode(node)) {
    return {
      hasNan: summaries.some((summary) => summary?.has_nan === true),
      hasInf: summaries.some((summary) => summary?.has_inf === true),
      sparsePercent: null,
    }
  }
  const threshold = isReluNode(node)
    ? sparseReluThresholdPercent
    : sparseDefaultThresholdPercent
  const sparseValues = summaries
    .map((summary) => summary?.zeros_pct)
    .filter((value): value is number => typeof value === 'number' && value > threshold)

  return {
    hasNan: summaries.some((summary) => summary?.has_nan === true),
    hasInf: summaries.some((summary) => summary?.has_inf === true),
    sparsePercent: sparseValues.length ? Math.max(...sparseValues) : null,
  }
}
