import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, PointerEvent } from 'react'
import './App.css'
import { RichTextView } from './components/RichTextView'
import { richTextToString } from './components/richText'
import { ShapeFlow } from './components/ShapeFlow'
import { explainNode } from './explanations'
import { buildLayout } from './graph/buildLayout'
import type { LayoutDirection } from './graph/buildLayout'
import { maxScale, minScale, nodeHeight, nodeWidth, whiteboardPadding } from './graph/constants'
import { loadStoredPositions, saveStoredPositions } from './graph/layoutStorage'
import type { LayoutPositions } from './graph/layoutStorage'
import { ModelSummary } from './inspector/ModelSummary'
import { NodeInspector } from './inspector/NodeInspector'
import { parseTraceJson } from './trace/parseTraceJson'
import { primaryInput, primaryOutput } from './trace/selectors'
import type { TraceEdge, TraceNode, TracePayload } from './trace/types'

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
