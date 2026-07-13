import { MAX_TENSOR_DIMENSIONS } from './constants.ts'
import { formatBytes, validateTensorDimensions } from './inputConfig.ts'
import type { TensorInputDraft } from './inputDrafts.ts'

export function TensorInputEditor({ draft, disabled, onChange }: { draft: TensorInputDraft; disabled: boolean; onChange: (draft: TensorInputDraft) => void }) {
  const validation = validateTensorDimensions(draft.dimensions)
  function dimensions(next: string[]) { onChange({ ...draft, dimensions: next }) }
  return (
    <section className="tensor-input-editor">
      <header><strong>{draft.parameterName}</strong><span>positional tensor</span></header>
      <div className="dimension-editor">
        {draft.dimensions.map((dimension, index) => (
          <label key={index}>Dim {index + 1}<input type="number" min="1" step="1" value={dimension} disabled={disabled} onChange={(event) => dimensions(draft.dimensions.map((item, current) => current === index ? event.target.value : item))} /></label>
        ))}
        <button type="button" aria-label={`Remove dimension from ${draft.parameterName}`} disabled={draft.dimensions.length === 1 || disabled} onClick={() => dimensions(draft.dimensions.slice(0, -1))}>-</button>
        <button type="button" aria-label={`Add dimension to ${draft.parameterName}`} disabled={draft.dimensions.length >= MAX_TENSOR_DIMENSIONS || disabled} onClick={() => dimensions([...draft.dimensions, '1'])}>+</button>
      </div>
      <div className="input-spec-summary"><span>CPU</span><span>float32</span><span>random normal</span><strong>{validation.ok ? `${validation.elementCount.toLocaleString()} elements · ${formatBytes(validation.sizeBytes)}` : validation.message}</strong></div>
    </section>
  )
}
