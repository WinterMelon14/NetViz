import type { ELK as ElkApi, ElkNode } from 'elkjs'
import type { TraceEdge, TraceNode } from '../trace/types'
import { columnGap, nodeHeight, padding, rowGap } from './constants'
import { nodeCardWidth } from './nodePresentation'

export type LayoutResult = {
  nodes: (TraceNode & {
    depth: number
    x: number
    y: number
  })[]
  width: number
  height: number
}

type PositionedNode = {
  id: string
  x: number
  y: number
  width: number
  height: number
}

function hasAlternativePath(
  source: string,
  target: string,
  ignoredEdgeId: string,
  edges: TraceEdge[],
): boolean {
  const outgoing = new Map<string, TraceEdge[]>()

  for (const edge of edges) {
    if (edge.id === ignoredEdgeId) continue

    const current = outgoing.get(edge.source) ?? []
    current.push(edge)
    outgoing.set(edge.source, current)
  }

  const queue = [source]
  const visited = new Set<string>([source])

  while (queue.length > 0) {
    const current = queue.shift()!

    for (const edge of outgoing.get(current) ?? []) {
      if (edge.target === target) return true
      if (visited.has(edge.target)) continue

      visited.add(edge.target)
      queue.push(edge.target)
    }
  }

  return false
}

function liftNodesAboveShortcutEdges(
  nodes: PositionedNode[],
  edges: TraceEdge[],
  trackGap: number,
): PositionedNode[] {
  const result = nodes.map((node) => ({ ...node }))
  const byId = new Map(result.map((node) => [node.id, node]))

  const shortcutEdges = edges.filter((edge) =>
    hasAlternativePath(edge.source, edge.target, edge.id, edges),
  )

  for (const edge of shortcutEdges) {
    const source = byId.get(edge.source)
    const target = byId.get(edge.target)

    if (!source || !target) continue

    const spanLeft = source.x + source.width
    const spanRight = target.x

    if (spanRight <= spanLeft) continue

    const nodesInsideSpan = result.filter(
      (node) =>
        node.id !== source.id &&
        node.id !== target.id &&
        node.x >= spanLeft &&
        node.x + node.width <= spanRight,
    )

    if (!nodesInsideSpan.length) continue

    /*
     * Treat the direct source → target edge as the root track.
     * Move intermediate nodes above that track.
     */
    const shortcutTop = Math.min(source.y, target.y)
    const desiredBottom = shortcutTop - trackGap

    const currentBottom = Math.max(
      ...nodesInsideSpan.map((node) => node.y + node.height),
    )

    const shift = currentBottom - desiredBottom

    if (shift <= 0) continue

    for (const node of nodesInsideSpan) {
      node.y -= shift
    }
  }

  return result
}

function normalizePositions(
  nodes: PositionedNode[],
): PositionedNode[] {
  const minX = Math.min(...nodes.map((node) => node.x))
  const minY = Math.min(...nodes.map((node) => node.y))

  const offsetX = minX < padding ? padding - minX : 0
  const offsetY = minY < padding ? padding - minY : 0

  if (!offsetX && !offsetY) return nodes

  return nodes.map((node) => ({
    ...node,
    x: node.x + offsetX,
    y: node.y + offsetY,
  }))
}

const layerTolerance = 1
let elkInstance: ElkApi | null = null

async function getElk() {
  if (elkInstance) return elkInstance

  const { default: ELK } = await import('elkjs/lib/elk.bundled.js')
  elkInstance = new ELK()
  return elkInstance
}

function fallbackLayout(): LayoutResult {
  return {
    nodes: [],
    width: padding * 2,
    height: padding * 2,
  }
}

function depthByLayer(layoutNodes: { id: string; x: number }[]) {
  const layers: number[] = []

  layoutNodes.forEach((node) => {
    const existingLayer = layers.find((layerX) => Math.abs(layerX - node.x) <= layerTolerance)
    if (existingLayer === undefined) {
      layers.push(node.x)
    }
  })

  layers.sort((left, right) => left - right)

  return new Map(
    layoutNodes.map((node) => {
      const depth = layers.findIndex((layerX) => Math.abs(layerX - node.x) <= layerTolerance)
      return [node.id, Math.max(0, depth)] as const
    }),
  )
}

function fallbackGraphSize(nodes: { x: number; y: number; width: number; height: number }[]) {
  return {
    width: Math.max(padding * 2, ...nodes.map((node) => node.x + node.width + padding)),
    height: Math.max(padding * 2, ...nodes.map((node) => node.y + node.height + padding)),
  }
}

export async function buildLayout(
  nodes: TraceNode[],
  edges: TraceEdge[],
): Promise<LayoutResult> {
  if (!nodes.length) return fallbackLayout()

  const nodeIds = new Set(nodes.map((node) => node.id))

  const validEdges = edges.filter(
    (edge) =>
      nodeIds.has(edge.source) &&
      nodeIds.has(edge.target),
  )

  const elkGraph: ElkNode = {
    id: 'trace-root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': String(rowGap),
      'elk.layered.spacing.nodeNodeBetweenLayers': String(columnGap),
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.padding':
        `[top=${padding},left=${padding},bottom=${padding},right=${padding}]`,
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: nodeCardWidth(node),
      height: nodeHeight,
    })),
    edges: validEdges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  }

  const elk = await getElk()
  const graph = await elk.layout(elkGraph)

  const childrenById = new Map(
    (graph.children ?? []).map((node) => [node.id, node]),
  )

  const elkPositionedNodes: PositionedNode[] = nodes.map((node) => {
    const layoutNode = childrenById.get(node.id)

    return {
      id: node.id,
      x: layoutNode?.x ?? padding,
      y: layoutNode?.y ?? padding,
      width: layoutNode?.width ?? nodeCardWidth(node),
      height: layoutNode?.height ?? nodeHeight,
    }
  })

  const liftedNodes = liftNodesAboveShortcutEdges(
    elkPositionedNodes,
    validEdges,
    rowGap,
  )

  const positionedNodes = normalizePositions(liftedNodes)

  const depths = depthByLayer(positionedNodes)
  const graphSize = fallbackGraphSize(positionedNodes)

  return {
    nodes: nodes.map((node, index) => ({
      ...node,
      depth: depths.get(node.id) ?? 0,
      x: positionedNodes[index]?.x ?? padding,
      y: positionedNodes[index]?.y ?? padding,
    })),
    width: graphSize.width,
    height: graphSize.height,
  }
}