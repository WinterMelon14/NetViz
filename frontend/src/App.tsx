import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, PointerEvent, WheelEvent } from 'react'
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

type TraceEvent = {
  step: number
  phase: string
  event: string
  node: string
  inputs: { node: string; output: number }[]
  outputs: { node: string; output: number }[]
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
  events?: TraceEvent[]
}

const nodeWidth = 244
const nodeHeight = 148
const columnGap = 128
const rowGap = 72
const padding = 64
const minScale = 0.25
const maxScale = 2.4

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
      <h3>Params</h3>
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
      <section className="detail-block">
        <h3>Memory</h3>
        <InfoRow label="param memory" value={stats?.total_param_memory?.human ?? 'n/a'} />
        <InfoRow label="trainable params" value={(stats?.trainable_params ?? 0).toLocaleString()} />
        <InfoRow label="non-trainable" value={(stats?.non_trainable_params ?? 0).toLocaleString()} />
      </section>
      <section className="detail-block">
        <h3>Inputs</h3>
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
      </section>
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
        <section className="detail-block">
          <h3>Attributes</h3>
          {attrEntries.map(([key, value]) => (
            <InfoRow key={key} label={key} value={value} />
          ))}
        </section>
      ) : null}

      <section className="formula-block">
        <h3>Formula</h3>
        <p>{node.formula ?? node.fx_op}</p>
      </section>

      <section className="stack-block">
        <h3>Inputs</h3>
        {tensorInputs.length ? tensorInputs.map((input) => <TensorDetail key={input.index} title={`${input.index}`} value={input} />) : <p className="empty-note">No tensor inputs</p>}
      </section>

      <section className="stack-block">
        <h3>Output</h3>
        {tensorOutputs.length ? tensorOutputs.map((output) => <TensorDetail key={output.index} title={`${output.index}`} value={output} />) : <p className="empty-note">No tensor outputs</p>}
      </section>

      <ParamsDetail params={node.params} />
    </>
  )
}

function App() {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const dragRef = useRef({ active: false, x: 0, y: 0, moved: false })
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [activeStep, setActiveStep] = useState(-1)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [isInspectorOpen, setIsInspectorOpen] = useState(true)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [view, setView] = useState({ x: 36, y: 36, scale: 1 })
  const [error, setError] = useState<string | null>(null)
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [jsonText, setJsonText] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/branchy.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load trace JSON (${response.status})`)
        return response.json() as Promise<TracePayload>
      })
      .then((payload) => {
        setTrace(payload)
        setSelectedNodeId(null)
        setActiveStep(-1)
      })
      .catch((loadError: Error) => setError(loadError.message))
  }, [])

  const layout = useMemo(() => (trace ? buildLayout(trace.graph.nodes, trace.graph.edges) : null), [trace])
  const nodesById = useMemo(() => new Map(layout?.nodes.map((node) => [node.id, node]) ?? []), [layout])
  const playbackLevels = useMemo(() => {
    const levels = new Map<number, string[]>()
    layout?.nodes.forEach((node) => {
      levels.set(node.depth, [...(levels.get(node.depth) ?? []), node.id])
    })
    return Array.from(levels.entries())
      .sort(([left], [right]) => left - right)
      .map(([, nodeIds]) => nodeIds)
  }, [layout])
  const activeNodeIds = new Set(activeStep >= 0 ? playbackLevels[activeStep] ?? [] : [])
  const playbackNode = activeNodeIds.size ? nodesById.get(Array.from(activeNodeIds)[0]) : undefined
  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) : undefined
  const inspectorNode = selectedNode ?? playbackNode

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
    setActiveStep(-1)
    setIsPlaying(false)
  }

  useEffect(() => {
    if (!isPlaying || !playbackLevels.length) return undefined
    const delay = 900 / speed
    const timer = window.setInterval(() => {
      setActiveStep((step) => {
        const next = step < 0 ? 0 : step + 1
        if (next >= playbackLevels.length) {
          setIsPlaying(false)
          return playbackLevels.length - 1
        }
        return next
      })
    }, delay)

    return () => window.clearInterval(timer)
  }, [isPlaying, playbackLevels.length, speed])

  useEffect(() => {
    const timer = window.setTimeout(fitView, 50)
    return () => window.clearTimeout(timer)
  }, [layout?.width, layout?.height])

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
    setSelectedNodeId(nodeId)
    const levelIndex = playbackLevels.findIndex((level) => level.includes(nodeId))
    setActiveStep(levelIndex)
    setIsPlaying(false)
    setIsInspectorOpen(true)
  }

  function applyTracePayload(payload: TracePayload) {
    setTrace(payload)
    setSelectedNodeId(null)
    setActiveStep(-1)
    setIsPlaying(false)
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

  function setStep(step: number) {
    setSelectedNodeId(null)
    setActiveStep(clamp(step, 0, Math.max(playbackLevels.length - 1, 0)))
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
                  const curve = Math.max(60, (endX - startX) / 2)
                  const isSelected = selectedNodeId === edge.source
                  const isOutgoing = activeNodeIds.has(edge.source)

                  return (
                    <path
                      key={edge.id}
                      className={`edge ${isSelected ? 'edge--selected' : ''} ${isOutgoing ? 'edge--pulse' : ''}`}
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
                const isActive = activeNodeIds.has(node.id)

                return (
                  <button
                    key={node.id}
                    type="button"
                    className={`graph-node ${isSelected ? 'graph-node--selected' : ''} ${isActive ? 'graph-node--active' : ''}`}
                    style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
                    onPointerDown={(event) => event.stopPropagation()}
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

      <footer className="timeline">
        <div className="timeline-controls">
          <button type="button" onClick={() => { setActiveStep(-1); setSelectedNodeId(null); setIsPlaying(false) }}>Reset</button>
          <button type="button" onClick={() => setStep(activeStep <= 0 ? 0 : activeStep - 1)}>Prev</button>
          <button type="button" onClick={() => { setSelectedNodeId(null); setIsPlaying((playing) => !playing) }}>{isPlaying ? 'Pause' : 'Play'}</button>
          <button type="button" onClick={() => setStep(activeStep < 0 ? 0 : activeStep + 1)}>Next</button>
          <label>
            Speed
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={1.5}>1.5x</option>
              <option value={2}>2x</option>
            </select>
          </label>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(playbackLevels.length - 1, 0)}
          value={activeStep < 0 ? 0 : activeStep}
          onChange={(event) => setStep(Number(event.target.value))}
        />
        <div className="timeline-readout">
          <span>{activeStep < 0 ? 'summary' : `level ${activeStep}`}</span>
          <strong>{activeNodeIds.size ? Array.from(activeNodeIds).join(' + ') : 'model'}</strong>
        </div>
      </footer>

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
