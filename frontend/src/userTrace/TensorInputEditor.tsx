import { DEFAULT_INTEGER_MAX_EXCLUSIVE, MAX_TENSOR_DIMENSIONS } from './constants.ts'
import { formatBytes, validateTensorDimensions } from './inputConfig.ts'
import type { TensorInputDraft } from './inputDrafts.ts'

type DimensionSource = TensorInputDraft['dimensionSources'][number]

export function TensorInputEditor({ draft, disabled, onChange }: { draft: TensorInputDraft; disabled: boolean; onChange: (draft: TensorInputDraft) => void }) {
  const validation = validateTensorDimensions(draft.dimensions, draft.dtype)

  function dimensions(next: string[], sources: DimensionSource[]) {
    onChange({ ...draft, dimensions: next, dimensionSources: sources })
  }

  function setDimension(index: number, value: string) {
    dimensions(
      draft.dimensions.map((item, current) => current === index ? value : item),
      draft.dimensionSources.map((source, current) => current === index ? 'chosen' : source),
    )
  }

  function setDtype(dtype: TensorInputDraft['dtype']) {
    onChange({
      ...draft,
      dtype,
      generator: dtype === 'int64' ? 'random_integer' : 'random_normal',
      integerMaxExclusive: dtype === 'int64' ? draft.integerMaxExclusive ?? DEFAULT_INTEGER_MAX_EXCLUSIVE : undefined,
    })
  }

  return (
    <section className="tensor-input-editor">
      <header><strong>Tensor</strong><span>{draft.parameterName}</span></header>
      <div className="dimension-editor">
        {draft.dimensions.map((dimension, index) => (
          <label key={index}>
            Dim {index + 1}
            <input type="number" min="1" step="1" value={dimension} disabled={disabled} onChange={(event) => setDimension(index, event.target.value)} />
          </label>
        ))}
        <button type="button" aria-label={'Remove dimension from ' + draft.parameterName} disabled={draft.dimensions.length === 1 || disabled} onClick={() => dimensions(draft.dimensions.slice(0, -1), draft.dimensionSources.slice(0, -1))}>-</button>
        <button type="button" aria-label={'Add dimension to ' + draft.parameterName} disabled={draft.dimensions.length >= MAX_TENSOR_DIMENSIONS || disabled} onClick={() => dimensions([...draft.dimensions, '1'], [...draft.dimensionSources, 'chosen'])}>+</button>
      </div>
      <div className="input-spec-controls">
        <label>
          Dtype
          <select value={draft.dtype} disabled={disabled} onChange={(event) => setDtype(event.target.value as TensorInputDraft['dtype'])}>
            <option value="float32">float32</option>
            <option value="int64">int64</option>
          </select>
        </label>
        {draft.dtype === 'int64' ? (
          <label>
            Values below
            <input type="number" min="1" step="1" value={draft.integerMaxExclusive ?? DEFAULT_INTEGER_MAX_EXCLUSIVE} disabled={disabled} onChange={(event) => onChange({ ...draft, integerMaxExclusive: Math.max(1, Number(event.target.value) || 1) })} />
          </label>
        ) : null}
      </div>
      <div className="input-spec-summary">
        <span>CPU</span>
        <span>{draft.dtype}</span>
        <span>{draft.generator === 'random_integer' ? 'random integer' : 'random normal'}</span>
        <strong>{validation.ok ? validation.elementCount.toLocaleString() + ' elements · ' + formatBytes(validation.sizeBytes) : validation.message}</strong>
      </div>
    </section>
  )
}
