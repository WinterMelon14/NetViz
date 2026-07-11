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

export async function buildLayout(nodes: TraceNode[], edges: TraceEdge[]): Promise<LayoutResult> {
  if (!nodes.length) return fallbackLayout()

  const nodeIds = new Set(nodes.map((node) => node.id))
  const elkGraph: ElkNode = {
    id: 'trace-root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': String(rowGap),
      'elk.layered.spacing.nodeNodeBetweenLayers': String(columnGap),
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.padding': `[top=${padding},left=${padding},bottom=${padding},right=${padding}]`,
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: nodeCardWidth(node),
      height: nodeHeight,
    })),
    edges: edges
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        sources: [edge.source],
        targets: [edge.target],
      })),
  }

  const elk = await getElk()
  const graph = await elk.layout(elkGraph)
  const childrenById = new Map((graph.children ?? []).map((node) => [node.id, node]))
  const positionedNodes = nodes.map((node) => {
    const layoutNode = childrenById.get(node.id)
    const width = layoutNode?.width ?? nodeCardWidth(node)
    const height = layoutNode?.height ?? nodeHeight

    return {
      id: node.id,
      x: layoutNode?.x ?? padding,
      y: layoutNode?.y ?? padding,
      width,
      height,
    }
  })
  const depths = depthByLayer(positionedNodes)
  const graphSize = fallbackGraphSize(positionedNodes)

  return {
    nodes: nodes.map((node, index) => ({
      ...node,
      depth: depths.get(node.id) ?? 0,
      x: positionedNodes[index]?.x ?? padding,
      y: positionedNodes[index]?.y ?? padding,
    })),
    width: graph.width ?? graphSize.width,
    height: graph.height ?? graphSize.height,
  }
}
