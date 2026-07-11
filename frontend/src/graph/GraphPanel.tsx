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
