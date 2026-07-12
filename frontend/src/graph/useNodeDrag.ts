import { useCallback, useRef, useState } from 'react'
import type { Dispatch, PointerEvent, SetStateAction } from 'react'
import type { LayoutResult } from './buildLayout'
import type { LayoutPositions } from './layoutStorage'
import type { PositionedTraceNode } from './types'

export function useNodeDrag({
  layout,
  nodesById,
  scale,
  setLayoutPositions,
}: {
  layout: LayoutResult | null
  nodesById: Map<string, PositionedTraceNode>
  scale: number
  setLayoutPositions: Dispatch<SetStateAction<LayoutPositions>>
}) {
  const nodeDragRef = useRef({ active: false, nodeId: '', x: 0, y: 0, startNodeX: 0, startNodeY: 0, moved: false })
  const pendingPositionRef = useRef<{ nodeId: string; x: number; y: number } | null>(null)
  const dragFrameRef = useRef<number | null>(null)
  const [isDraggingNode, setIsDraggingNode] = useState(false)

  const commitPendingPosition = useCallback(() => {
    const pending = pendingPositionRef.current
    pendingPositionRef.current = null
    dragFrameRef.current = null
    if (!pending) return
    setLayoutPositions((current) => ({
      ...current,
      [pending.nodeId]: { x: pending.x, y: pending.y },
    }))
  }, [setLayoutPositions])

  const onNodePointerDown = useCallback((event: PointerEvent<HTMLButtonElement>, nodeId: string) => {
    event.stopPropagation()
    const node = nodesById.get(nodeId)
    setIsDraggingNode(true)
    nodeDragRef.current = {
      active: true,
      nodeId,
      x: event.clientX,
      y: event.clientY,
      startNodeX: node?.x ?? 0,
      startNodeY: node?.y ?? 0,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }, [nodesById])

  const onNodePointerMove = useCallback((event: PointerEvent<HTMLButtonElement>) => {
    if (!nodeDragRef.current.active || !layout) return
    const dx = (event.clientX - nodeDragRef.current.x) / scale
    const dy = (event.clientY - nodeDragRef.current.y) / scale
    if (Math.abs(dx) + Math.abs(dy) > 1) nodeDragRef.current.moved = true
    const node = nodesById.get(nodeDragRef.current.nodeId)
    if (!node) return

    pendingPositionRef.current = {
      nodeId: node.id,
      x: nodeDragRef.current.startNodeX + dx,
      y: nodeDragRef.current.startNodeY + dy,
    }
    if (dragFrameRef.current === null) {
      dragFrameRef.current = window.requestAnimationFrame(commitPendingPosition)
    }
  }, [commitPendingPosition, layout, nodesById, scale])

  const onNodePointerUp = useCallback(() => {
    if (dragFrameRef.current !== null) window.cancelAnimationFrame(dragFrameRef.current)
    commitPendingPosition()
    nodeDragRef.current.active = false
    setIsDraggingNode(false)
  }, [commitPendingPosition])

  const wasNodeDragged = useCallback(() => nodeDragRef.current.moved, [])

  return {
    onNodePointerDown,
    onNodePointerMove,
    onNodePointerUp,
    isDraggingNode,
    wasNodeDragged,
  }
}
