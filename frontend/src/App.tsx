import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, PointerEvent } from 'react'
import './App.css'
import { CollapsibleSection } from './components/CollapsibleSection'
import { InfoRow } from './components/InfoRow'
import { RichTextView } from './components/RichTextView'
import { richTextToString } from './components/richText'
import { ShapeFlow } from './components/ShapeFlow'
import { ShapePill } from './components/ShapePill'
import { ValueDisplay } from './components/ValueDisplay'
import { explainNode } from './explanations'
import { formatDtype, formatNumber, formatPreview, formatShape, formatUnknown } from './trace/format'
import { primaryInput, primaryOutput, tensorValues } from './trace/selectors'
import type { ParamsInfo, TensorValue, TraceEdge, TraceNode, TracePayload } from './trace/types'

type NodePosition = {
  x: number
  y: number
}

type LayoutPositions = Record<string, NodePosition>
type LayoutDirection = 'left-right' | 'top-bottom'

const nodeWidth = 244
const nodeHeight = 136
const columnGap = 128
const rowGap = 72
const padding = 64
const minScale = 0.25
const maxScale = 2.4
const whiteboardPadding = 1800

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function nodeCardWidth(node: TraceNode) {
  const inputDims = primaryInput(node)?.shape?.length ?? 1
  const outputDims = primaryOutput(node)?.shape?.length ?? 1
  const shapeFlowWidth = 72 + (inputDims + outputDims) * 24
  const titleWidth = 76 + node.label.length * 8

  return Math.max(nodeWidth, shapeFlowWidth, titleWidth)
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

function maxColumnSpan(columns: Map<number, TraceNode[]>, direction: LayoutDirection) {
  const primarySize = direction === 'left-right' ? nodeHeight : nodeWidth
  return Math.max(
    primarySize,
    ...Array.from(columns.values()).map((column) => column.length * primarySize + (column.length - 1) * rowGap),
  )
}

function buildLayout(nodes: TraceNode[], edges: TraceEdge[], direction: LayoutDirection) {
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

  const widestLevel = maxColumnSpan(columns, direction)
  const layoutNodes = nodes.map((node) => {
    const column = depth.get(node.id) ?? 0
    const columnNodes = columns.get(column) ?? []
    const row = columnNodes.findIndex((candidate) => candidate.id === node.id)
    const levelSpan = columnNodes.length * (direction === 'left-right' ? nodeHeight : nodeWidth) + (columnNodes.length - 1) * rowGap

    return {
      ...node,
      depth: column,
      x:
        direction === 'left-right'
          ? padding + column * (nodeWidth + columnGap)
          : padding + Math.max(0, (widestLevel - levelSpan) / 2) + row * (nodeWidth + rowGap),
      y:
        direction === 'left-right'
          ? padding + Math.max(0, (widestLevel - levelSpan) / 2) + row * (nodeHeight + rowGap)
          : padding + column * (nodeHeight + columnGap),
    }
  })

  return {
    nodes: layoutNodes,
    width:
      direction === 'left-right'
        ? padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeWidth + Math.max(0, columns.size - 1) * columnGap
        : padding * 2 + Math.max(nodeWidth, widestLevel),
    height:
      direction === 'left-right'
        ? padding * 2 + widestLevel
        : padding * 2 + (Math.max(...Array.from(columns.keys()), 0) + 1) * nodeHeight + Math.max(0, columns.size - 1) * columnGap,
  }
}


function classifyShapeStep(from: unknown, to: unknown) {
  const fromText = formatUnknown(from)
  const toText = formatUnknown(to)

  if (fromText === toText) return 'preserved'
  if (fromText === '-' || fromText === 'n/a' || fromText === 'undefined') return 'created'
  if (toText === '-' || toText === 'n/a' || toText === 'undefined') return 'reduced'
  return 'changed'
}

function FocusButton({ nodeId, onFocusNode }: { nodeId: string; onFocusNode: (nodeId: string) => void }) {
  return (
    <button
      type="button"
      className="focus-chip"
      onClick={(event) => {
        event.stopPropagation()
        onFocusNode(nodeId)
      }}
    >
      {nodeId}
    </button>
  )
}

function TensorDetail({
  title,
  value,
  focusNodeId,
  outputTargets = [],
  onFocusNode,
}: {
  title: string
  value: TensorValue
  focusNodeId?: string
  outputTargets?: string[]
  onFocusNode: (nodeId: string) => void
}) {
  const isInteractive = Boolean(focusNodeId || outputTargets.length)

  function onTensorClick() {
    if (focusNodeId) {
      onFocusNode(focusNodeId)
    }
  }

  return (
    <section
      className={`detail-block ${isInteractive ? 'detail-block--interactive' : ''}`}
      role={focusNodeId ? 'button' : undefined}
      tabIndex={focusNodeId ? 0 : undefined}
      onKeyDown={(event) => {
        if (focusNodeId && (event.key === 'Enter' || event.key === ' ')) onTensorClick()
      }}
    >
      <h3>
        {title}
      </h3>
      {focusNodeId ? <InfoRow label="from" value={<FocusButton nodeId={focusNodeId} onFocusNode={onFocusNode} />} /> : null}
      {outputTargets.length ? (
        <div className="info-row">
          <span>to</span>
          <strong className="focus-list">{outputTargets.map((target) => <FocusButton key={target} nodeId={target} onFocusNode={onFocusNode} />)}</strong>
        </div>
      ) : null}
      <InfoRow label="shape" value={<ShapePill shape={value.shape} />} />
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
      {shapes.length ? shapes.map(([name, shape]) => <InfoRow key={name} label={name} value={<ShapePill shape={shape} />} />) : <p className="empty-note">No parameter tensors</p>}
      <InfoRow label="count" value={(params?.count ?? 0).toLocaleString()} />
      <InfoRow label="memory" value={params?.memory?.human ?? '0 B'} />
    </section>
  )
}

function TransformationDetail({ node }: { node: TraceNode }) {
  const explanation = explainNode(node)
  const inputShape = formatShape(primaryInput(node)?.shape)
  const outputShape = formatShape(primaryOutput(node)?.shape)

  if (!explanation) {
    return <p className="empty-note">No transformation metadata available.</p>
  }

  return (
    <section className="transformation-detail">
      <p className="transformation-short">
        <RichTextView value={explanation.short} />
      </p>
      <p>
        <RichTextView value={explanation.description} />
      </p>
      <InfoRow label="input shape" value={<ValueDisplay value={inputShape} />} />
      <InfoRow label="output shape" value={<ValueDisplay value={outputShape} />} />
      {explanation.shapeSteps.map((step) => (
        <section className={`shape-step shape-step--${classifyShapeStep(step.from, step.to)}`} key={step.label}>
          <h3>{step.label}</h3>
          {step.from !== undefined || step.to !== undefined ? (
            <div className="info-row">
              <span>change</span>
              <strong className="shape-change">
                <ValueDisplay value={step.from ?? 'n/a'} />
                <span className="shape-arrow">⟶</span>
                <ValueDisplay value={step.to ?? 'n/a'} />
              </strong>
            </div>
          ) : null}
          {step.reason ? <p>{step.reason}</p> : null}
          {step.substitution ? <code className="formula-code">{step.substitution}</code> : null}
        </section>
      ))}
      {explanation.formula ? (
        <section className="formula-block">
          <h3>Formula</h3>
          <code className="formula-code">{explanation.formula.display}</code>
          {explanation.formula.substitution ? <code className="formula-code formula-code--substitution">{explanation.formula.substitution}</code> : null}
        </section>
      ) : null}
    </section>
  )
}

function ModelSummary({ trace, outputNodes }: { trace: TracePayload; outputNodes: TraceNode[] }) {
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
              <span><ShapePill shape={input.shape} /></span>
              <span>{formatDtype(input.dtype)}</span>
              <span>{input.memory?.human ?? 'n/a'}</span>
            </div>
          ))
        ) : (
          <p className="empty-note">No input specs found in stats.</p>
        )}
      </CollapsibleSection>
      <CollapsibleSection title="Outputs">
        {outputNodes.length ? (
          outputNodes.map((node) => (
            <div className="input-spec" key={node.id}>
              <strong>{node.label}</strong>
              <span>{node.id}</span>
              <span><ShapePill shape={primaryOutput(node)?.shape} /></span>
              <span>{formatDtype(primaryOutput(node)?.dtype)}</span>
            </div>
          ))
        ) : (
          <p className="empty-note">No terminal output nodes found.</p>
        )}
      </CollapsibleSection>
    </>
  )
}

function NodeInspector({
  node,
  incomingEdges,
  outgoingEdges,
  onFocusNode,
}: {
  node: TraceNode
  incomingEdges: TraceEdge[]
  outgoingEdges: TraceEdge[]
  onFocusNode: (nodeId: string) => void
}) {
  const tensorInputs = tensorValues(node.inputs)
  const tensorOutputs = tensorValues(node.outputs)
  const attrEntries = Object.entries(node.attrs ?? {})

  return (
    <>
      <header className="inspector-header">
        <p className="eyebrow">Inspector</p>
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
            <InfoRow key={key} label={key} value={formatUnknown(value)} />
          ))}
        </CollapsibleSection>
      ) : null}

      <CollapsibleSection title="Transformation">
        <TransformationDetail node={node} />
      </CollapsibleSection>

      <CollapsibleSection title="Inputs">
        <section className="stack-block">
        {tensorInputs.length ? tensorInputs.map((input) => {
          const sourceNodeId = input.from ?? incomingEdges.find((edge) => edge.target_input === input.index)?.source
          return <TensorDetail key={input.index} title={`${input.index}`} value={input} focusNodeId={sourceNodeId} onFocusNode={onFocusNode} />
        }) : <p className="empty-note">No tensor inputs</p>}
        </section>
      </CollapsibleSection>

      <CollapsibleSection title="Output">
        <section className="stack-block">
        {tensorOutputs.length ? tensorOutputs.map((output) => {
          const targets = Array.from(new Set(outgoingEdges.filter((edge) => edge.source_output === output.index).map((edge) => edge.target)))
          return <TensorDetail key={output.index} title={`${output.index}`} value={output} outputTargets={targets} onFocusNode={onFocusNode} />
        }) : <p className="empty-note">No tensor outputs</p>}
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
  const inspectorRef = useRef<HTMLElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const dragRef = useRef({ active: false, x: 0, y: 0, moved: false })
  const nodeDragRef = useRef({ active: false, nodeId: '', x: 0, y: 0, startNodeX: 0, startNodeY: 0, moved: false })
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isInspectorOpen, setIsInspectorOpen] = useState(true)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [layoutDirection, setLayoutDirection] = useState<LayoutDirection>('left-right')
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

  const layout = useMemo(() => (trace ? buildLayout(trace.graph.nodes, trace.graph.edges, layoutDirection) : null), [layoutDirection, trace])
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
  const outgoingEdgesByNode = useMemo(() => {
    const map = new Map<string, TraceEdge[]>()
    trace?.graph.edges.forEach((edge) => {
      map.set(edge.source, [...(map.get(edge.source) ?? []), edge])
    })
    return map
  }, [trace])
  const incomingEdgesByNode = useMemo(() => {
    const map = new Map<string, TraceEdge[]>()
    trace?.graph.edges.forEach((edge) => {
      map.set(edge.target, [...(map.get(edge.target) ?? []), edge])
    })
    return map
  }, [trace])
  const outputNodes = useMemo(() => {
    return layoutNodes.filter((node) => node.inputs.length > 0 && !(outgoingEdgesByNode.get(node.id)?.length))
  }, [layoutNodes, outgoingEdgesByNode])
  const selectedNode = selectedNodeId ? nodesById.get(selectedNodeId) : undefined
  const inspectorNode = selectedNode
  const stageBounds = useMemo(() => {
    const xs = layoutNodes.map((node) => node.x)
    const ys = layoutNodes.map((node) => node.y)
    const minX = Math.min(0, ...xs)
    const minY = Math.min(0, ...ys)
    const maxX = Math.max(layout?.width ?? 0, ...layoutNodes.map((node) => node.x + nodeCardWidth(node)))
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

  function centerNode(nodeId: string) {
    const node = nodesById.get(nodeId)
    const viewport = viewportRef.current
    if (!node || !viewport) return

    const bounds = viewport.getBoundingClientRect()
    const nextScale = clamp(Math.max(view.scale, 1.12), minScale, maxScale)
    setView({
      scale: nextScale,
      x: bounds.width / 2 - (node.x + nodeCardWidth(node) / 2) * nextScale,
      y: bounds.height / 2 - (node.y + nodeHeight / 2) * nextScale,
    })
  }

  function focusNode(nodeId: string, options: { centerCamera?: boolean } = {}) {
    if (!nodesById.has(nodeId)) return
    setSelectedNodeId(nodeId)
    setIsInspectorOpen(true)
    if (options.centerCamera) {
      centerNode(nodeId)
    }
    inspectorRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
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
    if (dragRef.current.active && !dragRef.current.moved) {
      setSelectedNodeId(null)
    }
    dragRef.current.active = false
  }

  function onNodePointerDown(event: PointerEvent<HTMLButtonElement>, nodeId: string) {
    event.stopPropagation()
    const node = nodesById.get(nodeId)
    nodeDragRef.current = {
      active: true,
      nodeId,
      x: event.clientX,
      y: event.clientY,
      startNodeX: node?.x ?? 0,
      startNodeY: node?.y ?? 0,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function onNodePointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (!nodeDragRef.current.active || !layout) return
    const dx = (event.clientX - nodeDragRef.current.x) / view.scale
    const dy = (event.clientY - nodeDragRef.current.y) / view.scale
    if (Math.abs(dx) + Math.abs(dy) > 1) nodeDragRef.current.moved = true
    const node = nodesById.get(nodeDragRef.current.nodeId)
    if (!node) return

    setLayoutPositions((current) => ({
      ...current,
      [node.id]: {
        x: nodeDragRef.current.startNodeX + dx,
        y: nodeDragRef.current.startNodeY + dy,
      },
    }))
  }

  function onNodePointerUp() {
    nodeDragRef.current.active = false
  }

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return undefined
    const currentViewport = viewport

    function onWheel(event: WheelEvent) {
      event.preventDefault()
      const rect = currentViewport.getBoundingClientRect()
      const zoomFactor = Math.exp(-event.deltaY * 0.0035)
      const nextScale = clamp(view.scale * zoomFactor, minScale, maxScale)
      const graphX = (event.clientX - rect.left - view.x) / view.scale
      const graphY = (event.clientY - rect.top - view.y) / view.scale

      setView({
        scale: nextScale,
        x: event.clientX - rect.left - graphX * nextScale,
        y: event.clientY - rect.top - graphY * nextScale,
      })
    }

    currentViewport.addEventListener('wheel', onWheel, { passive: false })
    return () => currentViewport.removeEventListener('wheel', onWheel)
  }, [view.scale, view.x, view.y])

  function selectNode(nodeId: string) {
    if (nodeDragRef.current.moved) return
    focusNode(nodeId)
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
          <button type="button" onClick={fitView}>Fit Graph</button>
          <button
            type="button"
            className="icon-button"
            aria-label={layoutDirection === 'left-right' ? 'Switch to top to bottom layout' : 'Switch to left to right layout'}
            title={layoutDirection === 'left-right' ? 'Top to bottom' : 'Left to right'}
            onClick={() => {
              setLayoutDirection((current) => (current === 'left-right' ? 'top-bottom' : 'left-right'))
              setLayoutPositions({})
            }}
          >
            <span className={`layout-icon layout-icon--${layoutDirection}`} aria-hidden="true"></span>
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
          >
            <span className="moon-icon" aria-hidden="true"></span>
          </button>
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
        >
          <div className="graph-transform" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}>
            <div className="graph-stage" style={{ width: stageBounds.width, height: stageBounds.height }}>
              <svg className="edge-layer" width={stageBounds.width} height={stageBounds.height} aria-hidden="true">
                {trace.graph.edges.map((edge) => {
                  const source = nodesById.get(edge.source)
                  const target = nodesById.get(edge.target)
                  if (!source || !target) return null
                  const sourceWidth = nodeCardWidth(source)
                  const targetWidth = nodeCardWidth(target)
                  const startX = layoutDirection === 'left-right' ? source.x + sourceWidth : source.x + sourceWidth / 2
                  const startY = layoutDirection === 'left-right' ? source.y + nodeHeight / 2 : source.y + nodeHeight
                  const endX = layoutDirection === 'left-right' ? target.x : target.x + targetWidth / 2
                  const endY = layoutDirection === 'left-right' ? target.y + nodeHeight / 2 : target.y
                  const curve = Math.max(60, layoutDirection === 'left-right' ? (endX - startX) / 2 : (endY - startY) / 2)
                  const isSelected = selectedNodeId === edge.source

                  const path =
                    layoutDirection === 'left-right'
                      ? `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`
                      : `M ${startX} ${startY} C ${startX} ${startY + curve}, ${endX} ${endY - curve}, ${endX} ${endY}`

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
                const isOutputNode = outputNodes.some((outputNode) => outputNode.id === node.id)
                const explanation = explainNode(node)

                return (
                  <button
                    key={node.id}
                    type="button"
                    className={`graph-node ${node.kind === 'input' ? 'graph-node--input' : ''} ${isOutputNode ? 'graph-node--output' : ''} ${isSelected ? 'graph-node--selected graph-node--active' : ''}`}
                    style={{ transform: `translate(${node.x}px, ${node.y}px)`, width: nodeCardWidth(node) }}
                    onPointerDown={(event) => onNodePointerDown(event, node.id)}
                    onPointerMove={onNodePointerMove}
                    onPointerUp={onNodePointerUp}
                    onPointerCancel={onNodePointerUp}
                    onClick={() => selectNode(node.id)}
                  >
                    {node.kind !== 'input' ? <span className={`node-badge node-badge--${node.kind}`}>{kindBadge(node)}</span> : null}
                    {node.module?.is_reused ? <span className="node-badge node-badge--shared">S</span> : null}
                    <span className="node-title">
                      {node.label}
                      {explanation ? (
                        <span
                          className="shape-help"
                          aria-label={richTextToString(explanation.short)}
                        >
                          ?
                          <span className="shape-tooltip">
                            <RichTextView value={explanation.short} />
                          </span>
                        </span>
                      ) : null}
                    </span>
                    <span className="node-label">
                      <ShapeFlow input={input?.shape} output={output?.shape} />
                    </span>
                    <span className="node-kind">{node.kind}</span>
                    <span className="node-param">{totalParamLabel(node)} / {primaryOutput(node)?.memory?.human ?? '0 B'} act</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </section>

      <aside ref={inspectorRef} className="inspector" aria-label="Node inspector">
        {inspectorNode ? (
          <NodeInspector
            node={inspectorNode}
            incomingEdges={incomingEdgesByNode.get(inspectorNode.id) ?? []}
            outgoingEdges={outgoingEdgesByNode.get(inspectorNode.id) ?? []}
            onFocusNode={(nodeId) => focusNode(nodeId, { centerCamera: true })}
          />
        ) : (
          <ModelSummary trace={trace} outputNodes={outputNodes} />
        )}
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
