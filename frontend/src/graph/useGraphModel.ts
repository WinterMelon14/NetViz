import { useEffect, useMemo, useRef, useState } from 'react'
import { buildLayout } from './buildLayout'
import type { LayoutResult } from './buildLayout'
import { edgeMap, graphStageBounds } from './graphDerivations'
import type { LayoutPositions } from './layoutStorage'
import type { TracePayload } from '../trace/types'

type LayoutState = {
  trace: TracePayload | null
  layout: LayoutResult | null
  isLayoutPending: boolean
  layoutError: string | null
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
  const layoutRequestRef = useRef(0)
  const [layoutState, setLayoutState] = useState<LayoutState>({
    trace: null,
    layout: null,
    isLayoutPending: false,
    layoutError: null,
  })

  useEffect(() => {
    layoutRequestRef.current += 1
    const requestId = layoutRequestRef.current
    let isCancelled = false

    if (!trace) return undefined

    Promise.resolve()
      .then(() => {
        if (isCancelled || layoutRequestRef.current !== requestId) return undefined
        setLayoutState({ trace, layout: null, isLayoutPending: true, layoutError: null })
        return buildLayout(trace.graph.nodes, trace.graph.edges)
      })
      .then((nextLayout) => {
        if (!nextLayout) return
        if (isCancelled || layoutRequestRef.current !== requestId) return
        setLayoutState({ trace, layout: nextLayout, isLayoutPending: false, layoutError: null })
      })
      .catch((error: unknown) => {
        if (isCancelled || layoutRequestRef.current !== requestId) return
        setLayoutState({
          trace,
          layout: null,
          isLayoutPending: false,
          layoutError: error instanceof Error ? error.message : 'Could not lay out graph.',
        })
      })

    return () => {
      isCancelled = true
    }
  }, [trace])

  const isCurrentLayout = layoutState.trace === trace
  const layout = isCurrentLayout ? layoutState.layout : null
  const isLayoutPending = Boolean(trace) && (!isCurrentLayout || layoutState.isLayoutPending)
  const layoutError = isCurrentLayout ? layoutState.layoutError : null

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
    isLayoutPending,
    layoutError,
  }
}
