import type { PointerEvent } from 'react'
import { RichTextView } from '../components/RichTextView'
import { richTextToString } from '../components/richText'
import { ShapeFlow } from '../components/ShapeFlow'
import { ShapePill } from '../components/ShapePill'
import { explainNode } from '../explanations'
import { primaryInput, primaryOutput } from '../trace/selectors'
import { nodeCardWidth, totalParamLabel } from './nodePresentation'
import { nodeDiagnostics } from './nodeDiagnostics'
import type { PositionedTraceNode } from './types'

export function GraphNodeCard({
  node,
  timingSeverity,
  isOutputNode,
  isSelected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onSelectNode,
}: {
  node: PositionedTraceNode
  timingSeverity: 'low' | 'medium' | 'high' | null
  isOutputNode: boolean
  isSelected: boolean
  onPointerDown: (event: PointerEvent<HTMLButtonElement>, nodeId: string) => void
  onPointerMove: (event: PointerEvent<HTMLButtonElement>) => void
  onPointerUp: (event: PointerEvent<HTMLButtonElement>) => void
  onSelectNode: (nodeId: string) => void
}) {
  const input = primaryInput(node)
  const output = primaryOutput(node)
  const isInputNode = node.kind === 'input'
  const explanation = isInputNode ? null : explainNode(node)
  const diagnostics = nodeDiagnostics(node)
  const medianMs = node.profile?.median_ms

  return (
    <button
      type="button"
      className={`graph-node ${isInputNode ? 'graph-node--input' : ''} ${isOutputNode ? 'graph-node--output' : ''} ${isSelected ? 'graph-node--selected graph-node--active' : ''}`}
      style={{ transform: `translate(${node.x}px, ${node.y}px)`, width: nodeCardWidth(node) }}
      onPointerDown={(event) => onPointerDown(event, node.id)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onClick={() => onSelectNode(node.id)}
    >
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
        {isInputNode ? <ShapePill shape={output?.shape} /> : <ShapeFlow input={input?.shape} output={output?.shape} />}
      </span>
      <span className="node-status-row">
        <span className="node-kind">{node.kind}</span>
        {diagnostics.hasNan ? <span className="node-warning-badge node-warning-badge--danger" title="Observed NaN in node output">NaN</span> : null}
        {diagnostics.hasInf ? <span className="node-warning-badge node-warning-badge--danger" title="Observed infinity in node output">Inf</span> : null}
        {diagnostics.sparsePercent !== null ? <span className="node-warning-badge node-warning-badge--sparse" title={`${diagnostics.sparsePercent.toFixed(1)}% of output values are zero`}>Sparse {Math.round(diagnostics.sparsePercent)}%</span> : null}
        {medianMs !== undefined ? <span className={`node-warning-badge node-warning-badge--timing node-warning-badge--timing-${timingSeverity ?? 'low'}`} title="Median CPU duration">{formatNodeMs(medianMs)}</span> : null}
      </span>
      {isInputNode ? null : <span className="node-param">{totalParamLabel(node)} / {output?.memory?.human ?? '0 B'} act</span>}
    </button>
  )
}

function formatNodeMs(value: number) {
  if (value < 1) return `${value.toFixed(2)} ms`
  return `${value.toFixed(1)} ms`
}
