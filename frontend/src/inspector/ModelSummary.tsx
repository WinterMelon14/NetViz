import { CollapsibleSection } from '../components/CollapsibleSection'
import { InfoRow } from '../components/InfoRow'
import { ShapePill } from '../components/ShapePill'
import { formatDtype } from '../trace/format'
import { primaryOutput } from '../trace/selectors'
import type { TraceNode, TracePayload } from '../trace/types'

export function ModelSummary({ trace, outputNodes }: { trace: TracePayload; outputNodes: TraceNode[] }) {
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
