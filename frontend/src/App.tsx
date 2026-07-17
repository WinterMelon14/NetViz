import { useCallback, useRef, useState } from 'react'
import './App.css'
import { Topbar } from './app/Topbar'
import { EmptyTraceState } from './app/EmptyTraceState'
import { TraceRecovery } from './app/TraceRecovery'
import { getTraceViewState } from './app/traceViewState'
import { cancelTrace, consumeTraceFile, createTraceRunId, runUserTrace, type RunTraceResponse, type TraceRunState, type TraceWorkerError } from './desktop/desktopTraceApi'
import { GraphPanel } from './graph/GraphPanel'
import { useGraphModel } from './graph/useGraphModel'
import { useGraphViewport } from './graph/useGraphViewport'
import { useNodeDrag } from './graph/useNodeDrag'
import { ModelSummary } from './inspector/ModelSummary'
import { NodeInspector } from './inspector/NodeInspector'
import { useTraceLoader } from './trace/useTraceLoader'
import { UserTracePanel } from './userTrace/UserTracePanel'
import type { UserTraceDraft } from './userTrace/UserTracePanel'
import type { TraceFailure } from './userTrace/traceErrorDetails'

function App() {
  const inspectorRef = useRef<HTMLElement | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isInspectorOpen, setIsInspectorOpen] = useState(true)
  const [theme] = useState<'light' | 'dark'>('light')
  const [desktopTraceState, setDesktopTraceState] = useState<TraceRunState>('idle')
  const activeDesktopRunId = useRef<string | null>(null)
  const cancellingDesktopRunId = useRef<string | null>(null)
  const [desktopTraceFailure, setDesktopTraceFailure] = useState<TraceFailure | null>(null)
  const [isUserTraceOpen, setIsUserTraceOpen] = useState(false)
  const onTraceApplied = useCallback(() => setSelectedNodeId(null), [])
  const {
    trace,
    layoutPositions,
    setLayoutPositions,
    loadTracePayload,
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

  function traceStateFromErrorCode(code: string): TraceRunState {
    if (code === 'cancelled') return 'cancelled'
    if (code === 'timeout') return 'timed_out'
    return 'failed'
  }

  function startDesktopTrace(
    operation: (runId: string) => Promise<RunTraceResponse>,
    onSuccess?: () => void,
  ) {
    if (activeDesktopRunId.current) return

    const runId = createTraceRunId()
    activeDesktopRunId.current = runId
    setDesktopTraceFailure(null)
    setDesktopTraceState('starting')
    window.setTimeout(() => {
      if (activeDesktopRunId.current === runId) {
        setDesktopTraceState('running')
      }
    }, 0)

    operation(runId)
      .then(async (result) => {
        if (activeDesktopRunId.current !== runId || cancellingDesktopRunId.current === runId) return

        if (result.type === 'success') {
          if (result.trace.transfer === 'inline') {
            loadTracePayload(result.trace.payload)
          } else {
            loadTracePayload(await consumeTraceFile(runId, result.trace.path))
          }
          setDesktopTraceState('succeeded')
          onSuccess?.()
          return
        }

        setDesktopTraceFailure({ runId: result.run_id ?? runId, error: result.error })
        setDesktopTraceState(traceStateFromErrorCode(result.error.code))
      })
      .catch((traceError: unknown) => {
        if (activeDesktopRunId.current !== runId || cancellingDesktopRunId.current === runId) return
        const error: TraceWorkerError = {
          code: 'desktop_trace_failed',
          title: 'Desktop trace failed',
          message: traceError instanceof Error ? traceError.message : 'Desktop trace failed.',
          stage: 'desktop_bridge',
        }
        setDesktopTraceFailure({ runId, error })
        setDesktopTraceState('failed')
      })
      .finally(() => {
        if (activeDesktopRunId.current === runId && cancellingDesktopRunId.current !== runId) {
          activeDesktopRunId.current = null
        }
      })
  }

  function runSelectedModelTrace(request: UserTraceDraft) {
    startDesktopTrace(
      (runId) => runUserTrace({ ...request, run_id: runId }),
      () => setIsUserTraceOpen(false),
    )
  }

  async function cancelDesktopTrace() {
    const runId = activeDesktopRunId.current
    if (!runId || cancellingDesktopRunId.current === runId) return

    cancellingDesktopRunId.current = runId
    setDesktopTraceFailure(null)
    setDesktopTraceState('cancelling')
    try {
      const result = await cancelTrace(runId)
      if (activeDesktopRunId.current !== runId) return
      if (result.type === 'error') {
        setDesktopTraceFailure({ runId: result.run_id ?? runId, error: result.error })
        setDesktopTraceState(traceStateFromErrorCode(result.error.code))
      } else {
        setDesktopTraceFailure({
          runId,
          error: {
            code: 'desktop_bridge_protocol_error',
            title: 'Cancellation failed',
            message: 'The desktop bridge returned an unexpected success response to cancellation.',
            stage: 'desktop_bridge',
          },
        })
        setDesktopTraceState('failed')
      }
    } catch (cancelError: unknown) {
      if (activeDesktopRunId.current !== runId) return
      setDesktopTraceFailure({
        runId,
        error: {
          code: 'desktop_bridge_unavailable',
          title: 'Cancellation failed',
          message: cancelError instanceof Error ? cancelError.message : 'Desktop trace cancellation failed.',
          stage: 'desktop_bridge',
        },
      })
      setDesktopTraceState('failed')
    } finally {
      if (activeDesktopRunId.current === runId) activeDesktopRunId.current = null
      if (cancellingDesktopRunId.current === runId) cancellingDesktopRunId.current = null
    }
  }

  function clearDesktopTraceError() {
    setDesktopTraceFailure(null)
    setDesktopTraceState((current) => (
      current === 'failed' || current === 'cancelled' || current === 'timed_out' ? 'idle' : current
    ))
  }

  function openUserTrace() {
    setDesktopTraceFailure(null)
    setDesktopTraceState((current) => (
      current === 'succeeded' || current === 'failed' || current === 'cancelled' || current === 'timed_out' ? 'idle' : current
    ))
    setIsUserTraceOpen(true)
  }

  const recoveryError = layoutError
  const viewState = getTraceViewState({
    hasTrace: Boolean(trace),
    hasLayout: Boolean(layout),
    isLayoutPending,
    hasRecoveryError: Boolean(recoveryError),
  })
  const tracePanel = (
    <UserTracePanel
      key="user-trace-panel"
      isOpen={isUserTraceOpen}
      traceState={desktopTraceState}
      traceFailure={desktopTraceFailure}
      onRun={runSelectedModelTrace}
      onCancel={cancelDesktopTrace}
      onClearError={clearDesktopTraceError}
      onClose={() => setIsUserTraceOpen(false)}
    />
  )
  let applicationView
  if (viewState === 'recovery' && recoveryError) {
    applicationView = (
      <main className={`app-shell ${theme} app-shell--recovery`}>
        <TraceRecovery
          message={recoveryError}
          onTraceModel={openUserTrace}
        />
      </main>
    )
  } else if (viewState === 'empty') {
    applicationView = <EmptyTraceState onTraceModel={openUserTrace} />
  } else if (viewState === 'layout' || !trace || !layout) {
    applicationView = <main className={`app-shell ${theme} app-shell--message`}>Preparing graph...</main>
  } else {
    applicationView = (
      <main className={`app-shell ${theme} ${isInspectorOpen ? '' : 'inspector-collapsed'}`}>
        <Topbar
          modelName={trace.model_name}
          onOpenUserTrace={openUserTrace}
          onFitGraph={resetGraphPositions}
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
      </main>
    )
  }

  return (
    <div className={`app-root app-shell ${theme}`}>
      {applicationView}
      {tracePanel}
    </div>
  )
}

export default App
