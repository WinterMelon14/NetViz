import type { TracePayload } from '../trace/types'

const layoutStorageVersion = 'elk-v1'

export type NodePosition = {
  x: number
  y: number
}

export type LayoutPositions = Record<string, NodePosition>

function hashTrace(payload: TracePayload) {
  const source = JSON.stringify({
    layout_version: layoutStorageVersion,
    model_name: payload.model_name,
    nodes: payload.graph.nodes.map((node) => node.id),
    edges: payload.graph.edges.map((edge) => [edge.source, edge.target, edge.source_output, edge.target_input]),
  })
  let hash = 5381

  for (let index = 0; index < source.length; index += 1) {
    hash = (hash * 33) ^ source.charCodeAt(index)
  }

  return (hash >>> 0).toString(36)
}

function layoutStorageKey(payload: TracePayload) {
  return `trace-layout:${hashTrace(payload)}`
}

function warnMalformedStoredLayout(error?: unknown) {
  if (!import.meta.env.DEV) return
  console.warn('Discarding malformed saved graph layout positions.', error)
}

function isNodePosition(value: unknown): value is NodePosition {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.x === 'number' && Number.isFinite(candidate.x)
    && typeof candidate.y === 'number' && Number.isFinite(candidate.y)
}

function isLayoutPositions(value: unknown): value is LayoutPositions {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  return Object.values(value).every(isNodePosition)
}

export function loadStoredPositions(payload: TracePayload): LayoutPositions {
  const key = layoutStorageKey(payload)
  try {
    const stored = window.localStorage.getItem(key)
    if (!stored) return {}
    const parsed = JSON.parse(stored) as unknown
    const positions = parsed && typeof parsed === 'object' && 'layout' in parsed
      ? (parsed as { layout?: unknown }).layout
      : undefined
    const savedPositions = positions && typeof positions === 'object' && 'positions' in positions
      ? (positions as { positions?: unknown }).positions
      : undefined

    if (isLayoutPositions(savedPositions)) return savedPositions

    window.localStorage.removeItem(key)
    warnMalformedStoredLayout()
    return {}
  } catch (error) {
    try {
      window.localStorage.removeItem(key)
    } catch (removeError) {
      warnMalformedStoredLayout(removeError)
    }
    warnMalformedStoredLayout(error)
    return {}
  }
}

export function saveStoredPositions(payload: TracePayload, positions: LayoutPositions) {
  const key = layoutStorageKey(payload)
  if (!Object.keys(positions).length) {
    window.localStorage.removeItem(key)
    return
  }

  window.localStorage.setItem(key, JSON.stringify({ layout: { positions } }))
}
