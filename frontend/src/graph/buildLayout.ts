import type { TraceEdge, TraceNode } from '../trace/types'
import { columnGap, nodeHeight, nodeWidth, padding, rowGap } from './constants'

export type LayoutResult = {
  nodes: (TraceNode & {
    depth: number
    x: number
    y: number
  })[]
  width: number
  height: number
}

function maxColumnSpan(columns: Map<number, TraceNode[]>) {
  return Math.max(
    nodeHeight,
    ...Array.from(columns.values()).map((column) => column.length * nodeHeight + (column.length - 1) * rowGap),
  )
}

export function buildLayout(nodes: TraceNode[], edges: TraceEdge[]): LayoutResult {
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

  const widestLevel = maxColumnSpan(columns)
  const layoutNodes = nodes.map((node) => {
    const column = depth.get(node.id) ?? 0
    const columnNodes = columns.get(column) ?? []
    const row = columnNodes.findIndex((candidate) => candidate.id === node.id)
    const levelSpan = columnNodes.length * nodeHeight + (columnNodes.length - 1) * rowGap

    return {
      ...node,
      depth: column,
      x: padding + column * (nodeWidth + columnGap),
      y: padding + Math.max(0, (widestLevel - levelSpan) / 2) + row * (nodeHeight + rowGap),
    }
  })

  return {
    nodes: layoutNodes,
    width: padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeWidth + Math.max(0, columns.size - 1) * columnGap,
    height: padding * 2 + widestLevel,
  }
}
