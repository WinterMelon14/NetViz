/// <reference lib="webworker" />

import { buildLayoutPositions } from './buildLayout'
import type { LayoutNodeInput } from './buildLayout'
import type { TraceEdge } from '../trace/types'

type LayoutWorkerRequest = { requestId: number; nodes: LayoutNodeInput[]; edges: TraceEdge[] }

self.onmessage = async (event: MessageEvent<LayoutWorkerRequest>) => {
  const { requestId, nodes, edges } = event.data
  try {
    self.postMessage({ requestId, ok: true, layout: await buildLayoutPositions(nodes, edges) })
  } catch (error) {
    self.postMessage({
      requestId,
      ok: false,
      error: error instanceof Error ? error.message : 'Could not lay out graph.',
    })
  }
}
