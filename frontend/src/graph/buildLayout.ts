import * as dagre from 'dagre'
import type { TraceEdge, TraceNode } from '../trace/types'
import { columnGap, nodeHeight, nodeWidth, padding, rowGap } from './constants'
import { nodeCardWidth } from './nodePresentation'
export type LayoutResult = {
  nodes: (TraceNode & {
    depth: number
    x: number
    y: number
  })[]
  width: number
  height: number
}

type DagreNode = {
  x?: number
  y?: number
  rank?: number
}

export function buildLayout(nodes: TraceNode[], edges: TraceEdge[]): LayoutResult {
  const graph = new dagre.graphlib.Graph()

  graph.setGraph({
    rankdir: 'LR',
    nodesep: rowGap,
    ranksep: columnGap,
    marginx: padding,
    marginy: padding,
  })
  graph.setDefaultEdgeLabel(() => ({}))

  const nodeIds = new Set(nodes.map((node) => node.id))

  nodes.forEach((node) => {
    graph.setNode(node.id, { width: nodeCardWidth(node), height: nodeHeight })
  })

  edges.forEach((edge) => {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return
    graph.setEdge(edge.source, edge.target)
  })

  dagre.layout(graph)

  const layoutNodes = nodes.map((node) => {
    const position = graph.node(node.id) as DagreNode | undefined
    const x = position?.x ?? nodeWidth / 2
    const y = position?.y ?? nodeHeight / 2

    return {
      ...node,
      depth: position?.rank ?? 0,
      x: x - nodeCardWidth(node) / 2,
      y: y - nodeHeight / 2,
    }
  })

  const graphInfo = graph.graph()

  return {
    nodes: layoutNodes,
    width: graphInfo.width ?? 0,
    height: graphInfo.height ?? 0,
  }
}
