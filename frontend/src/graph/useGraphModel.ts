import { useMemo } from 'react'
import { buildLayout } from './buildLayout'
import { nodeHeight, whiteboardPadding } from './constants'
import type { LayoutPositions } from './layoutStorage'
import { nodeCardWidth } from './nodePresentation'
import type { GraphStageBounds, PositionedTraceNode } from './types'
import type { TraceEdge, TracePayload } from '../trace/types'

const emptyStageBounds: GraphStageBounds = {
  width: 4000,
  height: 3000,
}

function edgeMap(edges: TraceEdge[] | undefined, key: 'source' | 'target') {
  const map = new Map<string, TraceEdge[]>()
  edges?.forEach((edge) => {
    map.set(edge[key], [...(map.get(edge[key]) ?? []), edge])
  })
  return map
}

function graphStageBounds(
  layout: ReturnType<typeof buildLayout> | null,
  layoutNodes: PositionedTraceNode[],
) {
  const xs = layoutNodes.map((node) => node.x)
  const ys = layoutNodes.map((node) => node.y)
  const minX = Math.min(0, ...xs)
  const minY = Math.min(0, ...ys)
  const maxX = Math.max(layout?.width ?? 0, ...layoutNodes.map((node) => node.x + nodeCardWidth(node)))
  const maxY = Math.max(layout?.height ?? 0, ...layoutNodes.map((node) => node.y + nodeHeight))

  return {
    width: Math.max(emptyStageBounds.width, maxX - minX + whiteboardPadding),
    height: Math.max(emptyStageBounds.height, maxY - minY + whiteboardPadding),
  }
}

export function useGraphModel({
  trace,
  layoutPositions,
  selectedNodeId,
}: {
  trace: TracePayload | null
  layoutPositions: LayoutPositions
  selectedNodeId: string | null
}) {
  const layout = useMemo(() => (trace ? buildLayout(trace.graph.nodes, trace.graph.edges) : null), [trace])
  const layoutNodes = useMemo(() => {
    return (
      layout?.nodes.map((node) => ({
        ...node,
        x: layoutPositions[node.id]?.x ?? node.x,
        y: layoutPositions[node.id]?.y ?? node.y,
      })) ?? []
    )
  }, [layout, layoutPositions])
  const nodesById = useMemo(() => new Map(layoutNodes.map((node) => [node.id, node])), [layoutNodes])
  const outgoingEdgesByNode = useMemo(() => edgeMap(trace?.graph.edges, 'source'), [trace])
  const incomingEdgesByNode = useMemo(() => edgeMap(trace?.graph.edges, 'target'), [trace])
  const outputNodes = useMemo(() => {
    return layoutNodes.filter((node) => node.inputs.length > 0 && !(outgoingEdgesByNode.get(node.id)?.length))
  }, [layoutNodes, outgoingEdgesByNode])
  const outputNodeIds = useMemo(() => new Set(outputNodes.map((node) => node.id)), [outputNodes])
  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) : undefined
  const stageBounds = useMemo(() => graphStageBounds(layout, layoutNodes), [layout, layoutNodes])

  return {
    layout,
    layoutNodes,
    nodesById,
    outgoingEdgesByNode,
    incomingEdgesByNode,
    outputNodes,
    outputNodeIds,
    selectedNode,
    inspectorNode: selectedNode,
    stageBounds,
  }
}
