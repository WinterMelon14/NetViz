import type { ParameterInputDraft, StructuredValueDraft, StructuredValueKind } from './structuredInputDrafts.ts'
import { createValueDraft } from './structuredInputDrafts.ts'
import { MAX_STRUCTURED_CONTAINER_ITEMS } from './constants.ts'
import { TensorInputEditor } from './TensorInputEditor.tsx'

const VALUE_LABELS: Array<[StructuredValueKind, string]> = [
  ['tensor', 'Tensor'],
  ['none', 'None'],
  ['boolean', 'Boolean'],
  ['integer', 'Integer'],
  ['float', 'Float'],
  ['string', 'String'],
  ['tuple', 'Tuple'],
  ['list', 'List'],
  ['dict', 'Dictionary'],
]

export function StructuredInputEditor({
  draft,
  disabled,
  onChange,
}: {
  draft: ParameterInputDraft
  disabled: boolean
  onChange: (draft: ParameterInputDraft) => void
}) {
  return (
    <section className="structured-parameter-editor" id={`forward-parameter-${draft.parameterName}`} tabIndex={-1}>
      <header>
        <div><strong>{draft.parameterName}</strong><small>{draft.annotationText ?? 'unannotated'}</small></div>
        <div className="structured-parameter-controls">
          {!draft.required ? (
            <label>Argument
              <select value={draft.included ? 'included' : 'omitted'} disabled={disabled} onChange={(event) => onChange({ ...draft, included: event.target.value === 'included' })}>
                <option value="omitted">Omitted</option>
                <option value="included">Provide value</option>
              </select>
            </label>
          ) : null}
          {draft.position === 'positional_or_keyword' && draft.included ? (
            <label>Placement
              <select value={draft.placement} disabled={disabled} onChange={(event) => onChange({ ...draft, placement: event.target.value as 'positional' | 'keyword' })}>
                <option value="positional">Positional</option>
                <option value="keyword">Keyword</option>
              </select>
            </label>
          ) : <span className="parameter-placement">{draft.position === 'keyword_only' ? 'keyword' : 'positional'}</span>}
        </div>
      </header>
      {draft.included ? <StructuredValueEditor value={draft.value} parameterName={draft.parameterName} disabled={disabled} depth={1} onChange={(value) => onChange({ ...draft, value })} /> : <p className="source-muted">Uses the default declared by the model.</p>}
    </section>
  )
}

function StructuredValueEditor({
  value,
  parameterName,
  disabled,
  depth,
  onChange,
}: {
  value: StructuredValueDraft
  parameterName: string
  disabled: boolean
  depth: number
  onChange: (value: StructuredValueDraft) => void
}) {
  function changeKind(kind: StructuredValueKind) {
    if (kind !== value.kind) onChange(createValueDraft(kind, parameterName))
  }

  return (
    <div className="structured-value-editor" data-depth={depth}>
      <label className="structured-kind-control">Value type
        <select value={value.kind} disabled={disabled} onChange={(event) => changeKind(event.target.value as StructuredValueKind)}>
          {VALUE_LABELS.map(([kind, label]) => <option value={kind} key={kind}>{label}</option>)}
        </select>
      </label>
      {value.kind === 'tensor' ? <TensorInputEditor draft={value.tensor} disabled={disabled} onChange={(tensor) => onChange({ ...value, tensor })} /> : null}
      {value.kind === 'boolean' ? <label className="structured-boolean"><input type="checkbox" checked={value.value} disabled={disabled} onChange={(event) => onChange({ ...value, value: event.target.checked })} />True</label> : null}
      {value.kind === 'integer' || value.kind === 'float' ? (
        <label>{value.kind === 'integer' ? 'Integer value' : 'Float value'}
          <input type="number" step={value.kind === 'integer' ? '1' : 'any'} value={value.value} disabled={disabled} onChange={(event) => onChange({ ...value, value: event.target.value })} />
        </label>
      ) : null}
      {value.kind === 'string' ? <label>String value<input type="text" value={value.value} disabled={disabled} onChange={(event) => onChange({ ...value, value: event.target.value })} /></label> : null}
      {value.kind === 'none' ? <p className="source-muted">Passes Python None explicitly.</p> : null}
      {value.kind === 'list' || value.kind === 'tuple' ? (
        <ContainerItems value={value} parameterName={parameterName} disabled={disabled} depth={depth} onChange={onChange} />
      ) : null}
      {value.kind === 'dict' ? <DictionaryEntries value={value} parameterName={parameterName} disabled={disabled} depth={depth} onChange={onChange} /> : null}
    </div>
  )
}

function ContainerItems({
  value,
  parameterName,
  disabled,
  depth,
  onChange,
}: {
  value: Extract<StructuredValueDraft, { kind: 'list' | 'tuple' }>
  parameterName: string
  disabled: boolean
  depth: number
  onChange: (value: StructuredValueDraft) => void
}) {
  return (
    <div className="structured-container">
      {value.items.map((item, index) => (
        <div className="structured-container-row" key={item.id}>
          <span>{index + 1}</span>
          <StructuredValueEditor value={item} parameterName={parameterName} disabled={disabled} depth={depth + 1} onChange={(next) => onChange({ ...value, items: value.items.map((current) => current.id === item.id ? next : current) })} />
          <button type="button" aria-label={`Remove item ${index + 1}`} disabled={disabled} onClick={() => onChange({ ...value, items: value.items.filter((current) => current.id !== item.id) })}>-</button>
        </div>
      ))}
      <button type="button" disabled={disabled || value.items.length >= MAX_STRUCTURED_CONTAINER_ITEMS} onClick={() => onChange({ ...value, items: [...value.items, createValueDraft('tensor', parameterName)] })}>+ Add item</button>
    </div>
  )
}

function DictionaryEntries({
  value,
  parameterName,
  disabled,
  depth,
  onChange,
}: {
  value: Extract<StructuredValueDraft, { kind: 'dict' }>
  parameterName: string
  disabled: boolean
  depth: number
  onChange: (value: StructuredValueDraft) => void
}) {
  return (
    <div className="structured-container">
      {value.entries.map((entry, index) => (
        <div className="structured-container-row structured-dict-row" key={entry.id}>
          <label>Key<input type="text" value={entry.key} disabled={disabled} onChange={(event) => onChange({ ...value, entries: value.entries.map((current) => current.id === entry.id ? { ...current, key: event.target.value } : current) })} /></label>
          <StructuredValueEditor value={entry.value} parameterName={parameterName} disabled={disabled} depth={depth + 1} onChange={(next) => onChange({ ...value, entries: value.entries.map((current) => current.id === entry.id ? { ...current, value: next } : current) })} />
          <button type="button" aria-label={`Remove dictionary entry ${index + 1}`} disabled={disabled} onClick={() => onChange({ ...value, entries: value.entries.filter((current) => current.id !== entry.id) })}>-</button>
        </div>
      ))}
      <button type="button" disabled={disabled || value.entries.length >= MAX_STRUCTURED_CONTAINER_ITEMS} onClick={() => onChange({ ...value, entries: [...value.entries, { id: `${value.id}-entry-${value.entries.length}`, key: '', value: createValueDraft('tensor', parameterName) }] })}>+ Add entry</button>
    </div>
  )
}
