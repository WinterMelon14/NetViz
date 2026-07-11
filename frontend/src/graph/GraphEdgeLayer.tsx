import { nodeHeight } from './constants'
import { nodeCardWidth } from './nodePresentation'
import type { PositionedTraceNode, GraphStageBounds } from './types'
import type { TraceEdge } from '../trace/types'

export function GraphEdgeLayer({
  edges,
  nodesById,
  stageBounds,
  selectedNodeId,
}: {
  edges: TraceEdge[]
  nodesById: Map<string, PositionedTraceNode>
  stageBounds: GraphStageBounds
  selectedNodeId: string | null
}) {
  return (
    <svg className="edge-layer" width={stageBounds.width} height={stageBounds.height} aria-hidden="true">
      {edges.map((edge) => {
        const source = nodesById.get(edge.source)
        const target = nodesById.get(edge.target)
        if (!source || !target) return null

        const sourceWidth = nodeCardWidth(source)
        const startX = source.x + sourceWidth
        const startY = source.y + nodeHeight / 2
        const endX = target.x
        const endY = target.y + nodeHeight / 2
        const curve = Math.max(60, (endX - startX) / 2)
        const isSelected = selectedNodeId === edge.source

        const path = `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`

        return (
          <g key={edge.id}>
            <path
              className={`edge ${isSelected ? 'edge--active' : ''}`}
              d={path}
            />
            {isSelected ? <path className="edge-streak" d={path} /> : null}
          </g>
        )
      })}
    </svg>
  )
}
