import type { LayoutResult } from './buildLayout.ts'
import { nodeHeight, whiteboardPadding } from './constants.ts'
import { calculateStageBounds, edgeMap } from './graphDerivationsCore.ts'
import { nodeCardWidth } from './nodePresentation.ts'
import type { GraphStageBounds, PositionedTraceNode } from './types.ts'

const emptyStageBounds: GraphStageBounds = { width: 4000, height: 3000 }

export { edgeMap }

export function graphStageBounds(layout: LayoutResult | null, layoutNodes: PositionedTraceNode[]) {
  const { minX, minY, maxX, maxY } = calculateStageBounds(
    layout,
    layoutNodes,
    nodeCardWidth,
    nodeHeight,
    whiteboardPadding,
  )

  return {
    width: Math.max(emptyStageBounds.width, maxX - minX + whiteboardPadding),
    height: Math.max(emptyStageBounds.height, maxY - minY + whiteboardPadding),
  }
}
