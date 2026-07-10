import type { TracePayload } from '../trace/types'

export type NodePosition = {
  x: number
  y: number
}

export type LayoutPositions = Record<string, NodePosition>

function hashTrace(payload: TracePayload) {
  const source = JSON.stringify({
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

export function loadStoredPositions(payload: TracePayload): LayoutPositions {
  try {
    const stored = window.localStorage.getItem(layoutStorageKey(payload))
    if (!stored) return {}
    const parsed = JSON.parse(stored) as { layout?: { positions?: LayoutPositions } }
    return parsed.layout?.positions ?? {}
  } catch {
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
