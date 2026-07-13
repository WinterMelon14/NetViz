import type { PythonSource } from '../desktop/sourceApi.ts'
import type { SourceInspectionError } from '../desktop/sourceInspectionApi.ts'
import { formatBytes } from './inputConfig.ts'
import { PastedSourceEditor } from './PastedSourceEditor.tsx'

export type SourceMode = 'file' | 'paste'

export function SourceInputStep({
  mode,
  source,
  pastedText,
  inspectionError,
  isSelecting,
  isInspecting,
  disabled,
  onModeChange,
  onChooseFile,
  onInspectSource,
  onPastedTextChange,
  onInspectPaste,
  onClearPaste,
}: {
  mode: SourceMode
  source: PythonSource | null
  pastedText: string
  inspectionError: SourceInspectionError | null
  isSelecting: boolean
  isInspecting: boolean
  disabled: boolean
  onModeChange: (mode: SourceMode) => void
  onChooseFile: () => void
  onInspectSource: () => void
  onPastedTextChange: (value: string) => void
  onInspectPaste: () => void
  onClearPaste: () => void
}) {
  return (
    <section className="user-trace-section source-input-step">
      <div className="user-trace-section-heading"><div><span>1</span><strong>Model source</strong></div></div>
      <div className="source-mode-selector" role="group" aria-label="Python model source">
        <button type="button" aria-pressed={mode === 'file'} onClick={() => onModeChange('file')} disabled={disabled}>Choose Python File</button>
        <button type="button" aria-pressed={mode === 'paste'} onClick={() => onModeChange('paste')} disabled={disabled}>Paste Code</button>
      </div>
      {mode === 'file' ? (
        <>
          <button type="button" className="source-primary-action" onClick={onChooseFile} disabled={isSelecting || disabled}>
            {isSelecting ? 'Opening...' : source?.kind === 'file' ? 'Choose Different File' : 'Choose Python File'}
          </button>
          {source?.kind === 'file' ? (
            <div className="selected-file-row">
              <strong>{source.displayName}</strong>
              <span>{formatBytes(source.sizeBytes)}</span>
              <button type="button" disabled={isInspecting || disabled} onClick={onInspectSource}>Inspect Again</button>
            </div>
          ) : <p className="source-muted">Select one local .py file. Selection does not import or execute it.</p>}
        </>
      ) : (
        <PastedSourceEditor
          value={pastedText}
          disabled={disabled}
          isInspecting={isInspecting}
          onChange={onPastedTextChange}
          onInspect={onInspectPaste}
          onClear={onClearPaste}
        />
      )}
      {isInspecting ? <p className="source-muted">Inspecting source...</p> : null}
      {inspectionError ? <div className="source-error"><strong>{inspectionError.title}</strong><p>{inspectionError.message}</p></div> : null}
    </section>
  )
}
