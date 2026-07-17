import type { PointerEvent, Ref } from 'react'
import { GraphEdgeLayer } from './GraphEdgeLayer'
import { GraphNodeCard } from './GraphNodeCard'
import type { GraphStageBounds, GraphView, PositionedTraceNode } from './types'
import type { TraceEdge } from '../trace/types'

export function GraphPanel({
  modelName,
  viewportRef,
  nodes,
  edges,
  nodesById,
  outputNodeIds,
  stageBounds,
  view,
  selectedNodeId,
  isDraggingNode,
  onViewportPointerDown,
  onViewportPointerMove,
  onViewportPointerUp,
  onNodePointerDown,
  onNodePointerMove,
  onNodePointerUp,
  onSelectNode,
}: {
  modelName: string
  viewportRef: Ref<HTMLDivElement>
  nodes: PositionedTraceNode[]
  edges: TraceEdge[]
  nodesById: Map<string, PositionedTraceNode>
  outputNodeIds: Set<string>
  stageBounds: GraphStageBounds
  view: GraphView
  selectedNodeId: string | null
  isDraggingNode: boolean
  onViewportPointerDown: (event: PointerEvent<HTMLDivElement>) => void
  onViewportPointerMove: (event: PointerEvent<HTMLDivElement>) => void
  onViewportPointerUp: (event: PointerEvent<HTMLDivElement>) => void
  onNodePointerDown: (event: PointerEvent<HTMLButtonElement>, nodeId: string) => void
  onNodePointerMove: (event: PointerEvent<HTMLButtonElement>) => void
  onNodePointerUp: (event: PointerEvent<HTMLButtonElement>) => void
  onSelectNode: (nodeId: string) => void
}) {
  const timingValues = nodes.flatMap((node) => typeof node.profile?.median_ms === 'number' ? [node.profile.median_ms] : [])
  const timingMedian = median(timingValues)
  const timingStdDev = standardDeviation(timingValues)

  return (
    <section className="graph-panel" aria-label={`${modelName} graph`}>
      <div
        ref={viewportRef}
        className="graph-viewport"
        onPointerDown={onViewportPointerDown}
        onPointerMove={onViewportPointerMove}
        onPointerUp={onViewportPointerUp}
        onPointerCancel={onViewportPointerUp}
      >
        <div className="graph-transform" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
          <div className="graph-stage" style={{ width: stageBounds.width, height: stageBounds.height }}>
            <GraphEdgeLayer
              edges={edges}
              nodesById={nodesById}
              stageBounds={stageBounds}
              selectedNodeId={selectedNodeId}
              isDraggingNode={isDraggingNode}
            />

            {nodes.map((node) => (
              <GraphNodeCard
                key={node.id}
                node={node}
                timingSeverity={timingSeverity(node.profile?.median_ms, timingMedian, timingStdDev)}
                isOutputNode={outputNodeIds.has(node.id)}
                isSelected={node.id === selectedNodeId}
                onPointerDown={onNodePointerDown}
                onPointerMove={onNodePointerMove}
                onPointerUp={onNodePointerUp}
                onSelectNode={onSelectNode}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function median(values: number[]) {
  if (!values.length) return null
  const sorted = [...values].sort((left, right) => left - right)
  const midpoint = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[midpoint - 1] + sorted[midpoint]) / 2 : sorted[midpoint]
}

function standardDeviation(values: number[]) {
  if (values.length < 2) return 0
  const mean = values.reduce((total, value) => total + value, 0) / values.length
  const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function timingSeverity(value: number | undefined, baselineMedian: number | null, stdDev: number) {
  if (typeof value !== 'number' || baselineMedian === null) return null
  if (value >= baselineMedian + stdDev * 2 && value > baselineMedian) return 'high'
  if (value > baselineMedian) return 'medium'
  return 'low'
}
