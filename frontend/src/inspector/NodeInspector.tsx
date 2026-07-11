import { CollapsibleSection } from '../components/CollapsibleSection'
import { InfoRow } from '../components/InfoRow'
import { formatUnknown } from '../trace/format'
import { tensorValues } from '../trace/selectors'
import type { TraceEdge, TraceNode } from '../trace/types'
import { ParamsDetail } from './ParamsDetail'
import { TensorDetail } from './TensorDetail'
import { TransformationDetail } from './TransformationDetail'

export function NodeInspector({
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
  const isInputNode = node.kind === 'input'

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

      {isInputNode ? null : (
        <CollapsibleSection title="Transformation">
          <TransformationDetail node={node} />
        </CollapsibleSection>
      )}

      {isInputNode ? null : (
        <CollapsibleSection title="Inputs">
          <section className="stack-block">
          {tensorInputs.length ? tensorInputs.map((input) => {
            const sourceNodeId = input.from ?? incomingEdges.find((edge) => edge.target_input === input.index)?.source
            return <TensorDetail key={input.index} title={`${input.index}`} value={input} focusNodeId={sourceNodeId} onFocusNode={onFocusNode} />
          }) : <p className="empty-note">No tensor inputs</p>}
          </section>
        </CollapsibleSection>
      )}

      <CollapsibleSection title="Output">
        <section className="stack-block">
        {tensorOutputs.length ? tensorOutputs.map((output) => {
          const targets = Array.from(new Set(outgoingEdges.filter((edge) => edge.source_output === output.index).map((edge) => edge.target)))
          return <TensorDetail key={output.index} title={`${output.index}`} value={output} outputTargets={targets} onFocusNode={onFocusNode} />
        }) : <p className="empty-note">No tensor outputs</p>}
        </section>
      </CollapsibleSection>

      {isInputNode ? null : (
        <CollapsibleSection title="Params">
          <ParamsDetail params={node.params} />
        </CollapsibleSection>
      )}
    </>
  )
}
