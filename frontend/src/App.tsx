import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, PointerEvent, ReactNode, WheelEvent } from 'react'
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
  preview?: number[]
  summary?: TensorSummary
  memory?: {
    num_bytes?: number
    human?: string
  }
  from_node?: string
  source_output?: number
  value?: unknown
}

type ParamsInfo = {
  count?: number
  shapes?: Record<string, number[]>
  dtypes?: Record<string, string>
  memory?: {
    num_bytes?: number
    human?: string
  }
}

type TraceNode = {
  id: string
  kind: string
  label: string
  fx_op: string
  target: string
  inputs: TensorValue[]
  outputs: TensorValue[]
  module?: {
    path?: string
    type?: string
    is_reused?: boolean
    reuse_count?: number
  }
  params?: ParamsInfo
  attrs?: Record<string, unknown>
  formula?: string
}

type TraceEdge = {
  id: string
  source: string
  target: string
  source_output: number
  target_input: number
}

type TraceStats = {
  total_nodes?: number
  total_edges?: number
  total_params?: number
  trainable_params?: number
  non_trainable_params?: number
  total_param_memory?: { human?: string }
  total_activation_memory?: { human?: string }
  input_specs?: {
    index: number
    name?: string
    shape?: number[]
    dtype?: string
    memory?: { human?: string }
  }[]
}

type TracePayload = {
  model_name: string
  stats?: TraceStats
  graph: {
    nodes: TraceNode[]
    edges: TraceEdge[]
  }
}

type NodePosition = {
  x: number
  y: number
}

type LayoutPositions = Record<string, NodePosition>

const nodeWidth = 244
const nodeHeight = 148
const columnGap = 128
const rowGap = 72
const padding = 64
const minScale = 0.25
const maxScale = 2.4
const whiteboardPadding = 1800

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function formatShape(shape?: number[]) {
  return shape ? `[${shape.join(', ')}]` : 'scalar'
}

function formatDtype(dtype?: string) {
  return dtype?.replace('torch.', '') ?? 'n/a'
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

  return `${node.label} ${formatShape(input?.shape)} -> ${formatShape(output?.shape)}`
}

function formatNumber(value?: number, digits = 3, suffix = '') {
  if (typeof value !== 'number') return 'n/a'
  if (value === 0) return `0${suffix}`
  return `${value.toFixed(Math.abs(value) >= 100 ? 1 : digits)}${suffix}`
}

function formatPreview(values?: number[]) {
  if (!values?.length) return 'n/a'
  const preview = values.slice(0, 4).map((value) => formatNumber(value, 4))
  return `[${preview.join(', ')}${values.length > 4 ? ', ...' : ''}]`
}

function parseTraceJson(text: string) {
  const payload = JSON.parse(text) as TracePayload
  if (!payload.graph?.nodes || !payload.graph?.edges) {
    throw new Error('JSON must include graph.nodes and graph.edges.')
  }
  return payload
}

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

function loadStoredPositions(payload: TracePayload): LayoutPositions {
  try {
    const stored = window.localStorage.getItem(layoutStorageKey(payload))
    if (!stored) return {}
    const parsed = JSON.parse(stored) as { layout?: { positions?: LayoutPositions } }
    return parsed.layout?.positions ?? {}
  } catch {
    return {}
  }
}

function saveStoredPositions(payload: TracePayload, positions: LayoutPositions) {
  const key = layoutStorageKey(payload)
  if (!Object.keys(positions).length) {
    window.localStorage.removeItem(key)
    return
  }

  window.localStorage.setItem(key, JSON.stringify({ layout: { positions } }))
}

function totalParamLabel(node: TraceNode) {
  if (node.params?.count && node.params.count > 0) {
    return `${node.params.count.toLocaleString()} params`
  }
  return node.params?.memory?.human ?? 'no params'
}

function kindBadge(node: TraceNode) {
  const labelByKind: Record<string, string> = {
    input: 'I',
    module: 'M',
    function: 'F',
    method: 'T',
  }

  return labelByKind[node.kind] ?? node.kind.slice(0, 1).toUpperCase()
}

function maxColumnHeight(columns: Map<number, TraceNode[]>) {
  return Math.max(
    nodeHeight,
    ...Array.from(columns.values()).map((column) => column.length * nodeHeight + (column.length - 1) * rowGap),
  )
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
      if (inbound.get(next) === 0) queue.push(next)
    })
  }

  const columns = new Map<number, TraceNode[]>()
  nodes.forEach((node) => {
    const column = depth.get(node.id) ?? 0
    columns.set(column, [...(columns.get(column) ?? []), node])
  })

  const tallestColumn = maxColumnHeight(columns)
  const layoutNodes = nodes.map((node) => {
    const column = depth.get(node.id) ?? 0
    const columnNodes = columns.get(column) ?? []
    const row = columnNodes.findIndex((candidate) => candidate.id === node.id)
    const columnHeight = columnNodes.length * nodeHeight + (columnNodes.length - 1) * rowGap

    return {
      ...node,
      depth: column,
      x: padding + column * (nodeWidth + columnGap),
      y: padding + Math.max(0, (tallestColumn - columnHeight) / 2) + row * (nodeHeight + rowGap),
    }
  })

  return {
    nodes: layoutNodes,
    width: padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeWidth + Math.max(0, columns.size - 1) * columnGap,
    height: padding * 2 + tallestColumn,
  }
}

function InfoRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{String(value ?? 'n/a')}</strong>
    </div>
  )
}

function CollapsibleSection({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="collapse-section" open={defaultOpen}>
      <summary>
        <h3>{title}</h3>
      </summary>
      <div className="collapse-body">{children}</div>
    </details>
  )
}

function TensorDetail({ title, value }: { title: string; value: TensorValue }) {
  return (
    <section className="detail-block">
      <h3>
        {title}
        {value.from_node ? <span>from {value.from_node}</span> : null}
      </h3>
      <InfoRow label="shape" value={formatShape(value.shape)} />
      <InfoRow label="dtype" value={formatDtype(value.dtype)} />
      <InfoRow label="preview" value={formatPreview(value.preview)} />
      {value.summary ? (
        <>
          <InfoRow label="mean" value={formatNumber(value.summary.mean)} />
          <InfoRow label="std" value={formatNumber(value.summary.std)} />
          <InfoRow label="min/max" value={`${formatNumber(value.summary.min)} / ${formatNumber(value.summary.max)}`} />
          <InfoRow label="zero fraction" value={formatNumber(value.summary.zeros_pct, 2, '%')} />
        </>
      ) : null}
    </section>
  )
}

function ParamsDetail({ params }: { params?: ParamsInfo }) {
  const shapes = Object.entries(params?.shapes ?? {})

  return (
    <section className="detail-block">
      {shapes.length ? shapes.map(([name, shape]) => <InfoRow key={name} label={name} value={formatShape(shape)} />) : <p className="empty-note">No parameter tensors</p>}
      <InfoRow label="count" value={(params?.count ?? 0).toLocaleString()} />
      <InfoRow label="memory" value={params?.memory?.human ?? '0 B'} />
    </section>
  )
}

function ModelSummary({ trace }: { trace: TracePayload }) {
  const stats = trace.stats

  return (
    <>
      <header className="inspector-header">
        <p className="eyebrow">Model Summary</p>
        <h2>{trace.model_name}</h2>
      </header>
      <CollapsibleSection title="Overview">
        <section className="metric-grid">
          <div>
            <span>Nodes</span>
            <strong>{stats?.total_nodes ?? trace.graph.nodes.length}</strong>
          </div>
          <div>
            <span>Edges</span>
            <strong>{stats?.total_edges ?? trace.graph.edges.length}</strong>
          </div>
          <div>
            <span>Params</span>
            <strong>{(stats?.total_params ?? 0).toLocaleString()}</strong>
          </div>
          <div>
            <span>Activations</span>
            <strong>{stats?.total_activation_memory?.human ?? 'n/a'}</strong>
          </div>
        </section>
      </CollapsibleSection>
      <CollapsibleSection title="Memory">
        <InfoRow label="param memory" value={stats?.total_param_memory?.human ?? 'n/a'} />
        <InfoRow label="trainable params" value={(stats?.trainable_params ?? 0).toLocaleString()} />
        <InfoRow label="non-trainable" value={(stats?.non_trainable_params ?? 0).toLocaleString()} />
      </CollapsibleSection>
      <CollapsibleSection title="Inputs">
        {stats?.input_specs?.length ? (
          stats.input_specs.map((input) => (
            <div className="input-spec" key={`${input.index}-${input.name}`}>
              <strong>{input.name ?? `input ${input.index}`}</strong>
              <span>{formatShape(input.shape)}</span>
              <span>{formatDtype(input.dtype)}</span>
              <span>{input.memory?.human ?? 'n/a'}</span>
            </div>
          ))
        ) : (
          <p className="empty-note">No input specs found in stats.</p>
        )}
      </CollapsibleSection>
    </>
  )
}

function NodeInspector({ node }: { node: TraceNode }) {
  const tensorInputs = tensorValues(node.inputs)
  const tensorOutputs = tensorValues(node.outputs)
  const attrEntries = Object.entries(node.attrs ?? {})

  return (
    <>
      <header className="inspector-header">
        <p className="eyebrow">Node Inspector</p>
        <h2>{node.label}</h2>
        <div className="node-meta">
          <span>id: {node.id}</span>
          <span>kind: {node.kind}</span>
          <span>target: {node.target}</span>
          {node.module ? (
            <span>
              shared weights: {node.module.is_reused ? 'yes' : 'no'}, reuse count {node.module.reuse_count ?? 1}
            </span>
          ) : null}
        </div>
      </header>

      {attrEntries.length ? (
        <CollapsibleSection title="Attributes">
          {attrEntries.map(([key, value]) => (
            <InfoRow key={key} label={key} value={value} />
          ))}
        </CollapsibleSection>
      ) : null}

      <CollapsibleSection title="Formula">
        <section className="formula-block">
          <p>{node.formula ?? node.fx_op}</p>
        </section>
      </CollapsibleSection>

      <CollapsibleSection title="Inputs">
        <section className="stack-block">
        {tensorInputs.length ? tensorInputs.map((input) => <TensorDetail key={input.index} title={`${input.index}`} value={input} />) : <p className="empty-note">No tensor inputs</p>}
        </section>
      </CollapsibleSection>

      <CollapsibleSection title="Output">
        <section className="stack-block">
        {tensorOutputs.length ? tensorOutputs.map((output) => <TensorDetail key={output.index} title={`${output.index}`} value={output} />) : <p className="empty-note">No tensor outputs</p>}
        </section>
      </CollapsibleSection>

      <CollapsibleSection title="Params">
        <ParamsDetail params={node.params} />
      </CollapsibleSection>
    </>
  )
}

function App() {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const dragRef = useRef({ active: false, x: 0, y: 0, moved: false })
  const nodeDragRef = useRef({ active: false, nodeId: '', x: 0, y: 0, moved: false })
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isInspectorOpen, setIsInspectorOpen] = useState(true)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [view, setView] = useState({ x: 36, y: 36, scale: 1 })
  const [error, setError] = useState<string | null>(null)
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [jsonText, setJsonText] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [layoutPositions, setLayoutPositions] = useState<LayoutPositions>({})

  useEffect(() => {
    fetch('/branchy.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load trace JSON (${response.status})`)
        return response.json() as Promise<TracePayload>
      })
      .then((payload) => {
        setTrace(payload)
        setLayoutPositions(loadStoredPositions(payload))
        setSelectedNodeId(null)
      })
      .catch((loadError: Error) => setError(loadError.message))
  }, [])

  const layout = useMemo(() => (trace ? buildLayout(trace.graph.nodes, trace.graph.edges) : null), [trace])
  const layoutNodes = useMemo(() => {
    return (
      layout?.nodes.map((node) => ({
        ...node,
        x: layoutPositions[node.id]?.x ?? node.x,
        y: layoutPositions[node.id]?.y ?? node.y,
      })) ?? []
    )
  }, [layout, layoutPositions])
  const nodesById = useMemo(() => new Map(layoutNodes.map((node) => [node.id, node])), [layoutNodes])
  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) : undefined
  const inspectorNode = selectedNode
  const stageBounds = useMemo(() => {
    const xs = layoutNodes.map((node) => node.x)
    const ys = layoutNodes.map((node) => node.y)
    const minX = Math.min(0, ...xs)
    const minY = Math.min(0, ...ys)
    const maxX = Math.max(layout?.width ?? 0, ...layoutNodes.map((node) => node.x + nodeWidth))
    const maxY = Math.max(layout?.height ?? 0, ...layoutNodes.map((node) => node.y + nodeHeight))

    return {
      width: Math.max(4000, maxX - minX + whiteboardPadding),
      height: Math.max(3000, maxY - minY + whiteboardPadding),
    }
  }, [layout?.height, layout?.width, layoutNodes])

  function fitView() {
    if (!layout || !viewportRef.current) return
    const bounds = viewportRef.current.getBoundingClientRect()
    const scale = clamp(Math.min((bounds.width - 80) / layout.width, (bounds.height - 80) / layout.height), minScale, 1.4)
    setView({
      scale,
      x: (bounds.width - layout.width * scale) / 2,
      y: (bounds.height - layout.height * scale) / 2,
    })
  }

  function resetView() {
    setView({ x: 36, y: 36, scale: 1 })
    setSelectedNodeId(null)
  }

  useEffect(() => {
    const timer = window.setTimeout(fitView, 50)
    return () => window.clearTimeout(timer)
  }, [layout?.width, layout?.height])

  useEffect(() => {
    if (!trace) return
    saveStoredPositions(trace, layoutPositions)
  }, [layoutPositions, trace])

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    dragRef.current = { active: true, x: event.clientX, y: event.clientY, moved: false }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!dragRef.current.active) return
    const dx = event.clientX - dragRef.current.x
    const dy = event.clientY - dragRef.current.y
    if (Math.abs(dx) + Math.abs(dy) > 2) dragRef.current.moved = true
    dragRef.current.x = event.clientX
    dragRef.current.y = event.clientY
    setView((current) => ({ ...current, x: current.x + dx, y: current.y + dy }))
  }

  function onPointerUp() {
    dragRef.current.active = false
  }

  function onNodePointerDown(event: PointerEvent<HTMLButtonElement>, nodeId: string) {
    event.stopPropagation()
    nodeDragRef.current = { active: true, nodeId, x: event.clientX, y: event.clientY, moved: false }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function onNodePointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (!nodeDragRef.current.active || !layout) return
    const dx = (event.clientX - nodeDragRef.current.x) / view.scale
    const dy = (event.clientY - nodeDragRef.current.y) / view.scale
    if (Math.abs(dx) + Math.abs(dy) > 1) nodeDragRef.current.moved = true
    nodeDragRef.current.x = event.clientX
    nodeDragRef.current.y = event.clientY

    const node = nodesById.get(nodeDragRef.current.nodeId)
    if (!node) return

    setLayoutPositions((current) => ({
      ...current,
      [node.id]: {
        x: node.x + dx,
        y: node.y + dy,
      },
    }))
  }

  function onNodePointerUp() {
    nodeDragRef.current.active = false
  }

  function onWheel(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1
    const nextScale = clamp(view.scale * zoomFactor, minScale, maxScale)
    const graphX = (event.clientX - rect.left - view.x) / view.scale
    const graphY = (event.clientY - rect.top - view.y) / view.scale

    setView({
      scale: nextScale,
      x: event.clientX - rect.left - graphX * nextScale,
      y: event.clientY - rect.top - graphY * nextScale,
    })
  }

  function selectNode(nodeId: string) {
    if (nodeDragRef.current.moved) return
    setSelectedNodeId(nodeId)
    setIsInspectorOpen(true)
  }

  function applyTracePayload(payload: TracePayload) {
    setTrace(payload)
    setLayoutPositions(loadStoredPositions(payload))
    setSelectedNodeId(null)
    setError(null)
    setLoadError(null)
  }

  function loadJsonFromFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    file
      .text()
      .then((text) => {
        setJsonText(text)
        applyTracePayload(parseTraceJson(text))
        setIsLoadModalOpen(false)
      })
      .catch((fileError: Error) => setLoadError(fileError.message))
      .finally(() => {
        event.target.value = ''
      })
  }

  function loadJsonFromText() {
    try {
      applyTracePayload(parseTraceJson(jsonText))
      setIsLoadModalOpen(false)
    } catch (textError) {
      setLoadError(textError instanceof Error ? textError.message : 'Could not parse JSON.')
    }
  }

  function resetLayout() {
    if (trace) {
      window.localStorage.removeItem(layoutStorageKey(trace))
    }
    setLayoutPositions({})
  }

  if (error) return <main className={`app-shell ${theme} app-shell--message`}>{error}</main>
  if (!trace || !layout) return <main className={`app-shell ${theme} app-shell--message`}>Loading trace...</main>

  return (
    <main className={`app-shell ${theme} ${isInspectorOpen ? '' : 'inspector-collapsed'}`}>
      <header className="topbar">
        <div className="brand">
          <span>PyTorch Trace</span>
          <strong>{trace.model_name}</strong>
        </div>
        <div className="toolbar">
          <button type="button" onClick={() => setIsLoadModalOpen(true)}>Load JSON</button>
          <button type="button" onClick={resetView}>Reset</button>
          <button type="button" onClick={resetLayout}>Reset Layout</button>
          <button type="button" onClick={fitView}>Fit Graph</button>
          <button type="button" onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}>{theme === 'dark' ? 'Light' : 'Dark'}</button>
        </div>
      </header>

      <section className="graph-panel" aria-label={`${trace.model_name} graph`}>
        <div
          ref={viewportRef}
          className="graph-viewport"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        >
          <div className="graph-transform" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
            <div className="graph-stage" style={{ width: stageBounds.width, height: stageBounds.height }}>
              <svg className="edge-layer" width={stageBounds.width} height={stageBounds.height} aria-hidden="true">
                {trace.graph.edges.map((edge) => {
                  const source = nodesById.get(edge.source)
                  const target = nodesById.get(edge.target)
                  if (!source || !target) return null
                  const startX = source.x + nodeWidth
                  const startY = source.y + nodeHeight / 2
                  const endX = target.x
                  const endY = target.y + nodeHeight / 2
                  const curve = Math.max(60, (endX - startX) / 2)
                  const isSelected = selectedNodeId === edge.source

                  const path = `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`

                  return (
                    <g key={edge.id}>
                      <path
                        className={`edge ${isSelected ? 'edge--active' : ''}`}
                        d={path}
                      />
                      {isSelected ? <path className="edge-streak" d={path} /> : null}
                    </g>
                  )
                })}
              </svg>

              {layoutNodes.map((node) => {
                const input = primaryInput(node)
                const output = primaryOutput(node)
                const isSelected = node.id === selectedNodeId

                return (
                  <button
                    key={node.id}
                    type="button"
                    className={`graph-node ${isSelected ? 'graph-node--selected graph-node--active' : ''}`}
                    style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
                    onPointerDown={(event) => onNodePointerDown(event, node.id)}
                    onPointerMove={onNodePointerMove}
                    onPointerUp={onNodePointerUp}
                    onPointerCancel={onNodePointerUp}
                    onClick={() => selectNode(node.id)}
                  >
                    <span className={`node-badge node-badge--${node.kind}`}>{kindBadge(node)}</span>
                    {node.module?.is_reused ? <span className="node-badge node-badge--shared">S</span> : null}
                    <span className="node-title">{node.label}</span>
                    <span className="node-label">{nodeLabel(node)}</span>
                    <span className="node-kind">{node.kind}</span>
                    <span className="node-shapes">
                      <span>{formatShape(input?.shape)}</span>
                      <span>{formatShape(output?.shape)}</span>
                    </span>
                    <span className="node-param">{totalParamLabel(node)} / {primaryOutput(node)?.memory?.human ?? '0 B'} act</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </section>

      <aside className="inspector" aria-label="Node inspector">
        {inspectorNode ? <NodeInspector node={inspectorNode} /> : <ModelSummary trace={trace} />}
      </aside>

      <button
        type="button"
        className="inspector-handle"
        aria-label={isInspectorOpen ? 'Collapse inspector' : 'Expand inspector'}
        onClick={() => setIsInspectorOpen((open) => !open)}
      >
        {isInspectorOpen ? '>' : '<'}
      </button>

      {isLoadModalOpen ? (
        <div className="modal-backdrop" role="presentation">
          <section className="load-modal" role="dialog" aria-modal="true" aria-labelledby="load-json-title">
            <header>
              <div>
                <p className="eyebrow">Trace Loader</p>
                <h2 id="load-json-title">Load JSON</h2>
              </div>
              <button type="button" className="icon-button" aria-label="Close loader" onClick={() => setIsLoadModalOpen(false)}>x</button>
            </header>
            <textarea
              value={jsonText}
              onChange={(event) => {
                setJsonText(event.target.value)
                setLoadError(null)
              }}
              spellCheck={false}
              placeholder="Paste trace JSON here..."
            />
            {loadError ? <p className="load-error">{loadError}</p> : null}
            <footer>
              <input ref={fileInputRef} type="file" accept="application/json,.json" onChange={loadJsonFromFile} />
              <button type="button" onClick={() => fileInputRef.current?.click()}>Select File</button>
              <button type="button" className="primary-button" onClick={loadJsonFromText}>Load Pasted JSON</button>
            </footer>
          </section>
        </div>
      ) : null}
    </main>
  )
}

export default App
