import type { TracePayload } from './types'

export function parseTraceJson(text: string) {
  const payload = JSON.parse(text) as TracePayload
  if (!payload.graph?.nodes || !payload.graph?.edges) {
    throw new Error('JSON must include graph.nodes and graph.edges.')
  }
  return payload
}
