import { CollapsibleSection } from '../components/CollapsibleSection'
import { InfoRow } from '../components/InfoRow'
import { ShapePill } from '../components/ShapePill'
import { formatDtype } from '../trace/format'
import { primaryOutput } from '../trace/selectors'
import type { CPUNodeTiming, TraceNode, TracePayload } from '../trace/types'

export function ModelSummary({
  trace,
  outputNodes,
  onFocusNode,
  onIsolateNodes,
  onClearIsolation,
  isIsolationActive,
}: {
  trace: TracePayload
  outputNodes: TraceNode[]
  onFocusNode: (nodeId: string) => void
  onIsolateNodes: (nodeIds: string[]) => void
  onClearIsolation: () => void
  isIsolationActive: boolean
}) {
  const stats = trace.stats
  const profiling = trace.profiling
  const expensiveOperations = profiling?.expensive_operations.filter((operation) => operation.sample_count > 0).slice(0, 10) ?? []

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
      {profiling ? (
        <CollapsibleSection title="CPU Profiling">
          <InfoRow label="runs" value={`${profiling.config.warmup_runs} warmup / ${profiling.config.measurement_runs} measured`} />
          <InfoRow label="total median node time" value={formatMs(profiling.total_profiled_ms)} />
          <InfoRow label="critical path" value={formatMs(profiling.critical_path.total_ms)} />
          <div className="profiling-actions">
            <button type="button" onClick={() => onIsolateNodes(profiling.critical_path.node_ids)} disabled={!profiling.critical_path.node_ids.length}>Isolate Critical Path</button>
            {isIsolationActive ? <button type="button" onClick={onClearIsolation}>Show Full Graph</button> : null}
          </div>
          <div className="profiling-table" role="table" aria-label="Most expensive CPU operations">
            <div role="row">
              <span role="columnheader">Operation</span>
              <span role="columnheader">Median</span>
              <span role="columnheader">P95</span>
              <span role="columnheader">Samples</span>
            </div>
            {expensiveOperations.length ? expensiveOperations.map((operation) => (
              <button type="button" role="row" key={operation.node_id} onClick={() => onFocusNode(operation.node_id)}>
                <span role="cell" title={operation.label}>{operation.label}</span>
                <span role="cell">{formatOptionalMs(operation.median_ms)}</span>
                <span role="cell">{formatOptionalMs(operation.percentiles_ms['95'])}</span>
                <span role="cell">{operation.sample_count}</span>
              </button>
            )) : <p className="empty-note">No measured operations were recorded.</p>}
          </div>
        </CollapsibleSection>
      ) : null}
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

function formatMs(value: number) {
  if (value < 1) return `${value.toFixed(3)} ms`
  if (value < 100) return `${value.toFixed(2)} ms`
  return `${value.toFixed(1)} ms`
}

function formatOptionalMs(value: CPUNodeTiming['median_ms'] | undefined) {
  return typeof value === 'number' ? formatMs(value) : 'n/a'
}
