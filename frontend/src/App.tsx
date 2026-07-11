import { useCallback, useRef, useState } from 'react'
import './App.css'
import { Topbar } from './app/Topbar'
import { runKnownModelTrace } from './desktop/desktopTraceApi'
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
  const [theme] = useState<'light' | 'dark'>('light')
  const [isDesktopTraceRunning, setIsDesktopTraceRunning] = useState(false)
  const [desktopTraceError, setDesktopTraceError] = useState<string | null>(null)
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
    loadTracePayload,
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
    isLayoutPending,
    layoutError,
  } = useGraphModel({
    trace,
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
    isDraggingNode,
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

  function resetGraphPositions() {
    setLayoutPositions({})
    fitView()
  }

  function runDesktopTraceSpike() {
    setDesktopTraceError(null)
    setIsDesktopTraceRunning(true)
    runKnownModelTrace()
      .then((result) => {
        if (result.type === 'success') {
          if (result.trace.transfer === 'inline') {
            loadTracePayload(result.trace.payload)
          } else {
            setDesktopTraceError('Desktop trace returned a file transfer, which is not implemented in this spike.')
          }
          return
        }

        setDesktopTraceError(result.error.message)
      })
      .catch((traceError: unknown) => {
        setDesktopTraceError(traceError instanceof Error ? traceError.message : 'Desktop trace failed.')
      })
      .finally(() => setIsDesktopTraceRunning(false))
  }

  if (error) return <main className={`app-shell ${theme} app-shell--message`}>{error}</main>
  if (layoutError) return <main className={`app-shell ${theme} app-shell--message`}>{layoutError}</main>
  if (!trace || !layout || isLayoutPending) return <main className={`app-shell ${theme} app-shell--message`}>Loading trace...</main>

  return (
    <main className={`app-shell ${theme} ${isInspectorOpen ? '' : 'inspector-collapsed'}`}>
      <Topbar
        modelName={trace.model_name}
        onOpenLoader={() => setIsLoadModalOpen(true)}
        onFitGraph={resetGraphPositions}
        onRunDesktopTrace={runDesktopTraceSpike}
        isDesktopTraceRunning={isDesktopTraceRunning}
        desktopTraceError={desktopTraceError}
      />

      <GraphPanel
        modelName={trace.model_name}
        viewportRef={viewportRef}
        nodes={layoutNodes}
        edges={trace.graph.edges}
        nodesById={nodesById}
        outputNodeIds={outputNodeIds}
        stageBounds={stageBounds}
        view={view}
        selectedNodeId={selectedNodeId}
        isDraggingNode={isDraggingNode}
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
