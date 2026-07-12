import { useCallback, useEffect, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import type { LayoutResult } from './buildLayout'
import { maxScale, minScale, nodeHeight } from './constants'
import { nodeCardWidth } from './nodePresentation'
import type { GraphView, PositionedTraceNode } from './types'

const fitPadding = 36

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function useGraphViewport({
  layout,
  nodesById,
  onClearSelection,
}: {
  layout: LayoutResult | null
  nodesById: Map<string, PositionedTraceNode>
  onClearSelection: () => void
}) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef({ active: false, x: 0, y: 0, moved: false })
  const pendingPanRef = useRef({ dx: 0, dy: 0 })
  const panFrameRef = useRef<number | null>(null)
  const [view, setView] = useState<GraphView>({ x: 36, y: 36, scale: 1 })

  const fitView = useCallback(() => {
    if (!layout || !viewportRef.current) return
    const bounds = viewportRef.current.getBoundingClientRect()
    if (!layout.nodes.length || bounds.width <= 0 || bounds.height <= 0) return
    const availableWidth = Math.max(1, bounds.width - fitPadding * 2)
    const availableHeight = Math.max(1, bounds.height - fitPadding * 2)
    const scale = Math.min(
      maxScale,
      availableWidth / layout.width,
      availableHeight / layout.height,
    )

    setView({
      scale,
      x: (bounds.width - layout.width * scale) / 2,
      y: (bounds.height - layout.height * scale) / 2,
    })
  }, [layout])

  const centerNode = useCallback((nodeId: string) => {
    const node = nodesById.get(nodeId)
    const viewport = viewportRef.current
    if (!node || !viewport) return

    const bounds = viewport.getBoundingClientRect()
    const nextScale = clamp(Math.max(view.scale, 1.12), minScale, maxScale)
    setView({
      scale: nextScale,
      x: bounds.width / 2 - (node.x + nodeCardWidth(node) / 2) * nextScale,
      y: bounds.height / 2 - (node.y + nodeHeight / 2) * nextScale,
    })
  }, [nodesById, view.scale])

  useEffect(() => {
    const timer = window.setTimeout(fitView, 50)
    return () => window.clearTimeout(timer)
  }, [fitView])

  const onViewportPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragRef.current = { active: true, x: event.clientX, y: event.clientY, moved: false }
    event.currentTarget.setPointerCapture(event.pointerId)
  }, [])

  const onViewportPointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.active) return
    const dx = event.clientX - dragRef.current.x
    const dy = event.clientY - dragRef.current.y
    if (Math.abs(dx) + Math.abs(dy) > 2) dragRef.current.moved = true
    dragRef.current.x = event.clientX
    dragRef.current.y = event.clientY
    pendingPanRef.current.dx += dx
    pendingPanRef.current.dy += dy
    if (panFrameRef.current === null) {
      panFrameRef.current = window.requestAnimationFrame(() => {
        const pending = pendingPanRef.current
        pendingPanRef.current = { dx: 0, dy: 0 }
        panFrameRef.current = null
        setView((current) => ({ ...current, x: current.x + pending.dx, y: current.y + pending.dy }))
      })
    }
  }, [])

  const onViewportPointerUp = useCallback(() => {
    if (dragRef.current.active && !dragRef.current.moved) {
      onClearSelection()
    }
    dragRef.current.active = false
  }, [onClearSelection])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return undefined
    const currentViewport = viewport

    function onWheel(event: WheelEvent) {
      event.preventDefault()
      const rect = currentViewport.getBoundingClientRect()
      const zoomFactor = Math.exp(-event.deltaY * 0.0035)
      setView((current) => {
        const nextScale = clamp(current.scale * zoomFactor, Math.min(minScale, current.scale), maxScale)
        const graphX = (event.clientX - rect.left - current.x) / current.scale
        const graphY = (event.clientY - rect.top - current.y) / current.scale
        return {
          scale: nextScale,
          x: event.clientX - rect.left - graphX * nextScale,
          y: event.clientY - rect.top - graphY * nextScale,
        }
      })
    }

    currentViewport.addEventListener('wheel', onWheel, { passive: false })
    return () => currentViewport.removeEventListener('wheel', onWheel)
  }, [layout])

  useEffect(() => () => {
    if (panFrameRef.current !== null) window.cancelAnimationFrame(panFrameRef.current)
  }, [])

  return {
    viewportRef,
    view,
    fitView,
    centerNode,
    onViewportPointerDown,
    onViewportPointerMove,
    onViewportPointerUp,
  }
}
