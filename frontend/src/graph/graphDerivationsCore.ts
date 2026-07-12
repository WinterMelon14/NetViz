import type { TraceEdge } from '../trace/types.ts'

type BoundedNode = { x: number; y: number }

export function edgeMap(edges: TraceEdge[] | undefined, key: 'source' | 'target') {
  const map = new Map<string, TraceEdge[]>()
  edges?.forEach((edge) => {
    const mappedEdges = map.get(edge[key])
    if (mappedEdges) mappedEdges.push(edge)
    else map.set(edge[key], [edge])
  })
  return map
}

export function calculateStageBounds<T extends BoundedNode>(
  layoutSize: { width: number; height: number } | null,
  nodes: T[],
  nodeWidth: (node: T) => number,
  nodeHeight: number,
  padding: number,
) {
  let minX = 0
  let minY = 0
  let maxX = layoutSize?.width ?? 0
  let maxY = layoutSize?.height ?? 0
  for (const node of nodes) {
    minX = Math.min(minX, node.x)
    minY = Math.min(minY, node.y)
    maxX = Math.max(maxX, node.x + nodeWidth(node))
    maxY = Math.max(maxY, node.y + nodeHeight)
  }
  return { minX, minY, maxX, maxY, padding }
}
