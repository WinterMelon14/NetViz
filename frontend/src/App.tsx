import { useCallback, useRef, useState } from 'react'
import './App.css'
import type { LayoutDirection } from './graph/buildLayout'
import { GraphPanel } from './graph/GraphPanel'
import { useGraphModel } from './graph/useGraphModel'
import { useGraphViewport } from './graph/useGraphViewport'
import { useNodeDrag } from './graph/useNodeDrag'
import { ModelSummary } from './inspector/ModelSummary'
import { NodeInspector } from './inspector/NodeInspector'
import { TraceLoadDialog } from './trace/TraceLoadDialog'
import { useTraceLoader } from './trace/useTraceLoader'

function App() {
  const inspectorRef = useRef<HTMLElement | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isInspectorOpen, setIsInspectorOpen] = useState(true)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [layoutDirection, setLayoutDirection] = useState<LayoutDirection>('left-right')
  const onTraceApplied = useCallback(() => setSelectedNodeId(null), [])
  const {
    trace,
    error,
    isLoadModalOpen,
    setIsLoadModalOpen,
    jsonText,
    loadError,
    layoutPositions,
    setLayoutPositions,
    onJsonTextChange,
    loadJsonFromFile,
    loadJsonFromText,
  } = useTraceLoader({ onTraceApplied })

  const {
    layout,
    layoutNodes,
    nodesById,
    incomingEdgesByNode,
    outgoingEdgesByNode,
    outputNodes,
    outputNodeIds,
    inspectorNode,
    stageBounds,
  } = useGraphModel({
    trace,
    layoutDirection,
    layoutPositions,
    selectedNodeId,
  })

  const {
    viewportRef,
    view,
    fitView,
    centerNode,
    onViewportPointerDown,
    onViewportPointerMove,
    onViewportPointerUp,
  } = useGraphViewport({
    layout,
    nodesById,
    onClearSelection: () => setSelectedNodeId(null),
  })

  const {
    onNodePointerDown,
    onNodePointerMove,
    onNodePointerUp,
    wasNodeDragged,
  } = useNodeDrag({
    layout,
    nodesById,
    scale: view.scale,
    setLayoutPositions,
  })

  function focusNode(nodeId: string, options: { centerCamera?: boolean } = {}) {
    if (!nodesById.has(nodeId)) return
    setSelectedNodeId(nodeId)
    setIsInspectorOpen(true)
    if (options.centerCamera) {
      centerNode(nodeId)
    }
    inspectorRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function selectNode(nodeId: string) {
    if (wasNodeDragged()) return
    focusNode(nodeId)
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

      <GraphPanel
        modelName={trace.model_name}
        viewportRef={viewportRef}
        nodes={layoutNodes}
        edges={trace.graph.edges}
        nodesById={nodesById}
        outputNodeIds={outputNodeIds}
        stageBounds={stageBounds}
        view={view}
        layoutDirection={layoutDirection}
        selectedNodeId={selectedNodeId}
        onViewportPointerDown={onViewportPointerDown}
        onViewportPointerMove={onViewportPointerMove}
        onViewportPointerUp={onViewportPointerUp}
        onNodePointerDown={onNodePointerDown}
        onNodePointerMove={onNodePointerMove}
        onNodePointerUp={onNodePointerUp}
        onSelectNode={selectNode}
      />

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
        <TraceLoadDialog
          jsonText={jsonText}
          loadError={loadError}
          onJsonTextChange={onJsonTextChange}
          onFileSelected={loadJsonFromFile}
          onLoadPastedJson={loadJsonFromText}
          onClose={() => setIsLoadModalOpen(false)}
        />
      ) : null}
    </main>
  )
}

export default App
