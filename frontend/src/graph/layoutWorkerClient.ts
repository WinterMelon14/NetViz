import { nodeHeight, padding } from './constants'
import { nodeCardWidth } from './nodePresentation'
import { buildLayout } from './buildLayout'
import type { LayoutPositionResult, LayoutResult } from './buildLayout'
import type { TraceEdge, TraceNode } from '../trace/types'
import type { GraphSettings } from '../settings/graphSettings'

type WorkerResponse =
  | { requestId: number; ok: true; layout: LayoutPositionResult }
  | { requestId: number; ok: false; error: string }

export class LayoutWorkerClient {
  private worker: Worker | null = null
  private requestId = 0

  layout(nodes: TraceNode[], edges: TraceEdge[], settings: GraphSettings): Promise<LayoutResult> {
    this.cancel()
    if (typeof Worker !== 'function') return buildLayout(nodes, edges, settings)
    const requestId = ++this.requestId
    let worker: Worker
    try {
      worker = new Worker(new URL('./layoutWorker.ts', import.meta.url), { type: 'module' })
    } catch {
      return buildLayout(nodes, edges, settings)
    }
    this.worker = worker

    return new Promise((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        if (event.data.requestId !== requestId || this.worker !== worker) return
        worker.terminate()
        this.worker = null
        if (!event.data.ok) {
          reject(new Error(event.data.error))
          return
        }
        const positionsById = new Map(event.data.layout.nodes.map((node) => [node.id, node]))
        resolve({
          nodes: nodes.map((node) => ({
            ...node,
            depth: positionsById.get(node.id)?.depth ?? 0,
            x: positionsById.get(node.id)?.x ?? padding,
            y: positionsById.get(node.id)?.y ?? padding,
          })),
          width: event.data.layout.width,
          height: event.data.layout.height,
        })
      }
      worker.onerror = (event) => {
        if (this.worker !== worker) return
        worker.terminate()
        this.worker = null
        if (event.message.includes('not a constructor')) {
          buildLayout(nodes, edges, settings).then(resolve, reject)
          return
        }
        reject(new Error(event.message || 'Graph layout worker failed.'))
      }
      worker.postMessage({
        requestId,
        nodes: nodes.map((node) => ({ id: node.id, width: nodeCardWidth(node), height: nodeHeight })),
        edges,
        settings,
      })
    })
  }

  cancel() {
    this.requestId += 1
    this.worker?.terminate()
    this.worker = null
  }
}
