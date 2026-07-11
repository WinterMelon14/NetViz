import type { PointerEvent } from 'react'
import { RichTextView } from '../components/RichTextView'
import { richTextToString } from '../components/richText'
import { ShapeFlow } from '../components/ShapeFlow'
import { ShapePill } from '../components/ShapePill'
import { explainNode } from '../explanations'
import { primaryInput, primaryOutput } from '../trace/selectors'
import { kindBadge, nodeCardWidth, totalParamLabel } from './nodePresentation'
import type { PositionedTraceNode } from './types'

export function GraphNodeCard({
  node,
  isOutputNode,
  isSelected,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onSelectNode,
}: {
  node: PositionedTraceNode
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
        {isInputNode ? <ShapePill shape={output?.shape} /> : <ShapeFlow input={input?.shape} output={output?.shape} />}
      </span>
      <span className="node-kind">{node.kind}</span>
      {isInputNode ? null : <span className="node-param">{totalParamLabel(node)} / {output?.memory?.human ?? '0 B'} act</span>}
    </button>
  )
}
