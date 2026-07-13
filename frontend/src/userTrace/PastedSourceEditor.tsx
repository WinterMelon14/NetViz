import { useMemo, useState } from 'react'
import { MAX_SOURCE_BYTES, MAX_SOURCE_CHARS } from './constants.ts'

export function PastedSourceEditor({
  value,
  disabled,
  isInspecting,
  onChange,
  onInspect,
  onClear,
}: {
  value: string
  disabled: boolean
  isInspecting: boolean
  onChange: (value: string) => void
  onInspect: () => void
  onClear: () => void
}) {
  const [wrapLines, setWrapLines] = useState(true)
  const byteCount = useMemo(() => new TextEncoder().encode(value).byteLength, [value])
  const isOversized = value.length > MAX_SOURCE_CHARS || byteCount > MAX_SOURCE_BYTES
  const canInspect = Boolean(value.trim()) && !isOversized && !disabled && !isInspecting

  return (
    <div className="pasted-source-editor">
      <label htmlFor="pasted-python-source">Python source</label>
      <textarea
        id="pasted-python-source"
        value={value}
        disabled={disabled}
        spellCheck={false}
        wrap={wrapLines ? 'soft' : 'off'}
        onChange={(event) => onChange(event.target.value)}
      />
      <div className="pasted-source-toolbar">
        <label><input type="checkbox" checked={wrapLines} onChange={(event) => setWrapLines(event.target.checked)} />Wrap lines</label>
        <span className={isOversized ? 'source-count source-count--invalid' : 'source-count'}>
          {value.length.toLocaleString()} / {MAX_SOURCE_CHARS.toLocaleString()} characters · {byteCount.toLocaleString()} / {MAX_SOURCE_BYTES.toLocaleString()} bytes
        </span>
        <button type="button" onClick={onClear} disabled={disabled || !value}>Clear</button>
        <button type="button" onClick={onInspect} disabled={!canInspect}>{isInspecting ? 'Inspecting...' : 'Inspect Code'}</button>
      </div>
    </div>
  )
}
