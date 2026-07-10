import type { TraceEdge, TraceNode } from '../trace/types'
import { columnGap, nodeHeight, nodeWidth, padding, rowGap } from './constants'

export type LayoutDirection = 'left-right' | 'top-bottom'

export type LayoutResult = {
  nodes: (TraceNode & {
    depth: number
    x: number
    y: number
  })[]
  width: number
  height: number
}

function maxColumnSpan(columns: Map<number, TraceNode[]>, direction: LayoutDirection) {
  const primarySize = direction === 'left-right' ? nodeHeight : nodeWidth
  return Math.max(
    primarySize,
    ...Array.from(columns.values()).map((column) => column.length * primarySize + (column.length - 1) * rowGap),
  )
}

export function buildLayout(nodes: TraceNode[], edges: TraceEdge[], direction: LayoutDirection): LayoutResult {
  const nodeIds = new Set(nodes.map((node) => node.id))
  const inbound = new Map(nodes.map((node) => [node.id, 0]))
  const outgoing = new Map<string, string[]>()

  edges.forEach((edge) => {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return
    inbound.set(edge.target, (inbound.get(edge.target) ?? 0) + 1)
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target])
  })

  const queue = nodes.filter((node) => inbound.get(node.id) === 0).map((node) => node.id)
  const depth = new Map(nodes.map((node) => [node.id, 0]))

  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index]
    const currentDepth = depth.get(current) ?? 0

    ;(outgoing.get(current) ?? []).forEach((next) => {
      depth.set(next, Math.max(depth.get(next) ?? 0, currentDepth + 1))
      inbound.set(next, (inbound.get(next) ?? 1) - 1)
      if (inbound.get(next) === 0) queue.push(next)
    })
  }

  const columns = new Map<number, TraceNode[]>()
  nodes.forEach((node) => {
    const column = depth.get(node.id) ?? 0
    columns.set(column, [...(columns.get(column) ?? []), node])
  })

  const widestLevel = maxColumnSpan(columns, direction)
  const layoutNodes = nodes.map((node) => {
    const column = depth.get(node.id) ?? 0
    const columnNodes = columns.get(column) ?? []
    const row = columnNodes.findIndex((candidate) => candidate.id === node.id)
    const levelSpan = columnNodes.length * (direction === 'left-right' ? nodeHeight : nodeWidth) + (columnNodes.length - 1) * rowGap

    return {
      ...node,
      depth: column,
      x:
        direction === 'left-right'
          ? padding + column * (nodeWidth + columnGap)
          : padding + Math.max(0, (widestLevel - levelSpan) / 2) + row * (nodeWidth + rowGap),
      y:
        direction === 'left-right'
          ? padding + Math.max(0, (widestLevel - levelSpan) / 2) + row * (nodeHeight + rowGap)
          : padding + column * (nodeHeight + columnGap),
    }
  })

  return {
    nodes: layoutNodes,
    width:
      direction === 'left-right'
        ? padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeWidth + Math.max(0, columns.size - 1) * columnGap
        : padding * 2 + Math.max(nodeWidth, widestLevel),
    height:
      direction === 'left-right'
        ? padding * 2 + widestLevel
        : padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeHeight + Math.max(0, columns.size - 1) * columnGap,
  }
}
