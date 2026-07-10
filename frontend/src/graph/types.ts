import type { TraceNode } from '../trace/types'

export type PositionedTraceNode = TraceNode & {
  x: number
  y: number
}

export type GraphStageBounds = {
  width: number
  height: number
}

export type GraphView = {
  x: number
  y: number
  scale: number
}
