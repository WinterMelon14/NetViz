import { useCallback, useRef, useState } from 'react'
import './App.css'
import { Topbar } from './app/Topbar'
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
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
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
      <Topbar
        modelName={trace.model_name}
        theme={theme}
        onOpenLoader={() => setIsLoadModalOpen(true)}
        onFitGraph={fitView}
        onToggleTheme={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
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
