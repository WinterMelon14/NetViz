import type { LayoutDirection } from './buildLayout'
import { nodeHeight } from './constants'
import { nodeCardWidth } from './nodePresentation'
import type { PositionedTraceNode, GraphStageBounds } from './types'
import type { TraceEdge } from '../trace/types'

export function GraphEdgeLayer({
  edges,
  nodesById,
  stageBounds,
  layoutDirection,
  selectedNodeId,
}: {
  edges: TraceEdge[]
  nodesById: Map<string, PositionedTraceNode>
  stageBounds: GraphStageBounds
  layoutDirection: LayoutDirection
  selectedNodeId: string | null
}) {
  return (
    <svg className="edge-layer" width={stageBounds.width} height={stageBounds.height} aria-hidden="true">
      {edges.map((edge) => {
        const source = nodesById.get(edge.source)
        const target = nodesById.get(edge.target)
        if (!source || !target) return null

        const sourceWidth = nodeCardWidth(source)
        const targetWidth = nodeCardWidth(target)
        const startX = layoutDirection === 'left-right' ? source.x + sourceWidth : source.x + sourceWidth / 2
        const startY = layoutDirection === 'left-right' ? source.y + nodeHeight / 2 : source.y + nodeHeight
        const endX = layoutDirection === 'left-right' ? target.x : target.x + targetWidth / 2
        const endY = layoutDirection === 'left-right' ? target.y + nodeHeight / 2 : target.y
        const curve = Math.max(60, layoutDirection === 'left-right' ? (endX - startX) / 2 : (endY - startY) / 2)
        const isSelected = selectedNodeId === edge.source

        const path =
          layoutDirection === 'left-right'
            ? `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`
            : `M ${startX} ${startY} C ${startX} ${startY + curve}, ${endX} ${endY - curve}, ${endX} ${endY}`

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
