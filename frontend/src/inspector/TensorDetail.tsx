import { InfoRow } from '../components/InfoRow'
import { ShapePill } from '../components/ShapePill'
import { formatDtype, formatNumber, formatPreview } from '../trace/format'
import type { TensorValue } from '../trace/types'
import { FocusButton } from './FocusButton'

export function TensorDetail({
  title,
  value,
  focusNodeId,
  outputTargets = [],
  onFocusNode,
}: {
  title: string
  value: TensorValue
  focusNodeId?: string
  outputTargets?: string[]
  onFocusNode: (nodeId: string) => void
}) {
  const isInteractive = Boolean(focusNodeId || outputTargets.length)

  function onTensorClick() {
    if (focusNodeId) {
      onFocusNode(focusNodeId)
    }
  }

  return (
    <section
      className={`detail-block ${isInteractive ? 'detail-block--interactive' : ''}`}
      role={focusNodeId ? 'button' : undefined}
      tabIndex={focusNodeId ? 0 : undefined}
      onKeyDown={(event) => {
        if (focusNodeId && (event.key === 'Enter' || event.key === ' ')) onTensorClick()
      }}
    >
      <h3>
        {title}
      </h3>
      {focusNodeId ? <InfoRow label="from" value={<FocusButton nodeId={focusNodeId} onFocusNode={onFocusNode} />} /> : null}
      {outputTargets.length ? (
        <div className="info-row">
          <span>to</span>
          <strong className="focus-list">{outputTargets.map((target) => <FocusButton key={target} nodeId={target} onFocusNode={onFocusNode} />)}</strong>
        </div>
      ) : null}
      <InfoRow label="shape" value={<ShapePill shape={value.shape} />} />
      <InfoRow label="dtype" value={formatDtype(value.dtype)} />
      <InfoRow label="preview" value={formatPreview(value.preview)} />
      {value.summary ? (
        <>
          <InfoRow label="mean" value={formatNumber(value.summary.mean)} />
          <InfoRow label="std" value={formatNumber(value.summary.std)} />
          <InfoRow label="min/max" value={`${formatNumber(value.summary.min)} / ${formatNumber(value.summary.max)}`} />
          <InfoRow label="zero fraction" value={formatNumber(value.summary.zeros_pct, 2, '%')} />
        </>
      ) : null}
    </section>
  )
}
