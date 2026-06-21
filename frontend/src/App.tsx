import { useEffect, useMemo, useState } from 'react'
import './App.css'

type TensorSummary = {
  numel?: number
  min?: number
  max?: number
  mean?: number
  std?: number
  zeros_pct?: number
  has_nan?: boolean
  has_inf?: boolean
}

type TensorValue = {
  index: number
  role: string
  shape?: number[]
  dtype?: string
  summary?: TensorSummary
  memory?: {
    human?: string
  }
  value?: unknown
}

type TraceNode = {
  id: string
  kind: string
  label: string
  fx_op: string
  target: string
  inputs: TensorValue[]
  outputs: TensorValue[]
  formula?: string
}

type TraceEdge = {
  id: string
  source: string
  target: string
  source_output: number
  target_input: number
}

type TracePayload = {
  model_name: string
  graph: {
    nodes: TraceNode[]
    edges: TraceEdge[]
  }
}

const nodeWidth = 220
const nodeHeight = 126
const columnGap = 120
const rowGap = 64
const padding = 48

function formatShape(shape?: number[]) {
  return shape ? `[${shape.join(', ')}]` : 'scalar'
}

function tensorValues(values: TensorValue[]) {
  return values.filter((value) => value.shape || value.summary)
}

function primaryInput(node: TraceNode) {
  return tensorValues(node.inputs)[0]
}

function primaryOutput(node: TraceNode) {
  return tensorValues(node.outputs)[0]
}

function nodeLabel(node: TraceNode) {
  const input = primaryInput(node)
  const output = primaryOutput(node)

  if (!input && output) {
    return `${node.label} ${formatShape(output.shape)}`
  }

  return `${node.label} ${formatShape(input?.shape)} → ${formatShape(output?.shape)}`
}

function formatNumber(value?: number, suffix = '') {
  if (typeof value !== 'number') {
    return 'n/a'
  }

  const formatted = Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3)
  return `${formatted}${suffix}`
}

function summarizeTensor(value?: TensorValue) {
  if (!value) {
    return {
      shape: 'n/a',
      mean: 'n/a',
      zeros: 'n/a',
      dtype: 'n/a',
      memory: 'n/a',
    }
  }

  return {
    shape: formatShape(value.shape),
    mean: formatNumber(value.summary?.mean),
    zeros: formatNumber(value.summary?.zeros_pct, '%'),
    dtype: value.dtype ?? value.role,
    memory: value.memory?.human ?? 'n/a',
  }
}

function buildLayout(nodes: TraceNode[], edges: TraceEdge[]) {
  const nodeIds = new Set(nodes.map((node) => node.id))
  const inbound = new Map(nodes.map((node) => [node.id, 0]))
  const outgoing = new Map<string, string[]>()

  edges.forEach((edge) => {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return
    inbound.set(edge.target, (inbound.get(edge.target) ?? 0) + 1)
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target])
  })

  const queue = nodes.filter((node) => inbound.get(node.id) === 0).map((node) => node.id)
  const depth = new Map(nodes.map((node) => [node.id, 0]))

  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index]
    const currentDepth = depth.get(current) ?? 0

    ;(outgoing.get(current) ?? []).forEach((next) => {
      depth.set(next, Math.max(depth.get(next) ?? 0, currentDepth + 1))
      inbound.set(next, (inbound.get(next) ?? 1) - 1)
      if (inbound.get(next) === 0) {
        queue.push(next)
      }
    })
  }

  const columns = new Map<number, TraceNode[]>()
  nodes.forEach((node) => {
    const column = depth.get(node.id) ?? 0
    columns.set(column, [...(columns.get(column) ?? []), node])
  })

  const layoutNodes = nodes.map((node) => {
    const column = depth.get(node.id) ?? 0
    const columnNodes = columns.get(column) ?? []
    const row = columnNodes.findIndex((candidate) => candidate.id === node.id)
    const columnHeight = columnNodes.length * nodeHeight + (columnNodes.length - 1) * rowGap

    return {
      ...node,
      x: padding + column * (nodeWidth + columnGap),
      y: padding + Math.max(0, (maxColumnHeight(columns) - columnHeight) / 2) + row * (nodeHeight + rowGap),
    }
  })

  return {
    nodes: layoutNodes,
    width: padding * 2 + (Math.max(...Array.from(columns.keys())) + 1) * nodeWidth + Math.max(0, columns.size - 1) * columnGap,
    height: padding * 2 + maxColumnHeight(columns),
  }
}

function maxColumnHeight(columns: Map<number, TraceNode[]>) {
  return Math.max(
    nodeHeight,
    ...Array.from(columns.values()).map((column) => column.length * nodeHeight + (column.length - 1) * rowGap),
  )
}

function TensorSummaryBlock({ title, value }: { title: string; value?: TensorValue }) {
  const summary = summarizeTensor(value)

  return (
    <section className="summary-block">
      <h3>{title}</h3>
      <dl>
        <div>
          <dt>Shape</dt>
          <dd>{summary.shape}</dd>
        </div>
        <div>
          <dt>Mean</dt>
          <dd>{summary.mean}</dd>
        </div>
        <div>
          <dt>Zero fraction</dt>
          <dd>{summary.zeros}</dd>
        </div>
        <div>
          <dt>Dtype</dt>
          <dd>{summary.dtype}</dd>
        </div>
        <div>
          <dt>Memory</dt>
          <dd>{summary.memory}</dd>
        </div>
      </dl>
    </section>
  )
}

function App() {
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/branchy.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load trace JSON (${response.status})`)
        }
        return response.json() as Promise<TracePayload>
      })
      .then((payload) => {
        setTrace(payload)
        setSelectedNodeId(payload.graph.nodes[0]?.id ?? null)
      })
      .catch((loadError: Error) => setError(loadError.message))
  }, [])

  const layout = useMemo(() => {
    if (!trace) return null
    return buildLayout(trace.graph.nodes, trace.graph.edges)
  }, [trace])

  const nodesById = useMemo(() => {
    return new Map(layout?.nodes.map((node) => [node.id, node]) ?? [])
  }, [layout])

  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) : undefined
  const selectedInput = selectedNode ? primaryInput(selectedNode) : undefined
  const selectedOutput = selectedNode ? primaryOutput(selectedNode) : undefined

  if (error) {
    return <main className="app-shell app-shell--message">{error}</main>
  }

  if (!trace || !layout) {
    return <main className="app-shell app-shell--message">Loading trace...</main>
  }

  return (
    <main className="app-shell">
      <section className="graph-panel" aria-label={`${trace.model_name} graph`}>
        <header className="panel-header">
          <div>
            <p className="eyebrow">PyTorch Trace</p>
            <h1>{trace.model_name}</h1>
          </div>
          <div className="graph-stats">
            <span>{trace.graph.nodes.length} nodes</span>
            <span>{trace.graph.edges.length} edges</span>
          </div>
        </header>

        <div className="graph-viewport">
          <div className="graph-stage" style={{ width: layout.width, height: layout.height }}>
            <svg className="edge-layer" width={layout.width} height={layout.height} aria-hidden="true">
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" />
                </marker>
              </defs>
              {trace.graph.edges.map((edge) => {
                const source = nodesById.get(edge.source)
                const target = nodesById.get(edge.target)
                if (!source || !target) return null

                const startX = source.x + nodeWidth
                const startY = source.y + nodeHeight / 2
                const endX = target.x
                const endY = target.y + nodeHeight / 2
                const curve = Math.max(56, (endX - startX) / 2)
                const isActive = selectedNodeId === edge.source || selectedNodeId === edge.target

                return (
                  <path
                    key={edge.id}
                    className={isActive ? 'edge edge--active' : 'edge'}
                    d={`M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`}
                    markerEnd="url(#arrow)"
                  />
                )
              })}
            </svg>

            {layout.nodes.map((node) => {
              const input = primaryInput(node)
              const output = primaryOutput(node)
              const isSelected = node.id === selectedNodeId

              return (
                <button
                  key={node.id}
                  type="button"
                  className={isSelected ? 'graph-node graph-node--selected' : 'graph-node'}
                  style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
                  onClick={() => setSelectedNodeId(node.id)}
                >
                  <span className="node-label">{nodeLabel(node)}</span>
                  <span className="node-kind">{node.kind}</span>
                  <span className="node-shapes">
                    <span>{formatShape(input?.shape)}</span>
                    <span>{formatShape(output?.shape)}</span>
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </section>

      <aside className="inspector" aria-label="Node inspector">
        {selectedNode ? (
          <>
            <header className="inspector-header">
              <p className="eyebrow">Inspector</p>
              <h2>{selectedNode.label}</h2>
              <span>{selectedNode.kind}</span>
            </header>

            <section className="formula-block">
              <h3>Formula</h3>
              <p>{selectedNode.formula ?? selectedNode.fx_op}</p>
            </section>

            <TensorSummaryBlock title="Input" value={selectedInput} />
            <TensorSummaryBlock title="Output" value={selectedOutput} />
          </>
        ) : (
          <p>Select a node to inspect it.</p>
        )}
      </aside>
    </main>
  )
}

export default App
