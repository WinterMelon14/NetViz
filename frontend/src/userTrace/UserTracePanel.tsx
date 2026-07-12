import { useEffect, useRef, useState } from 'react'
import type { TraceRunState } from '../desktop/desktopTraceApi.ts'
import { inspectSelectedPythonFile, selectPythonFile } from '../desktop/selectedFileApi.ts'
import type { SelectedPythonFile } from '../desktop/selectedFileApi.ts'
import type { InspectModelSourceSuccess, ModelCandidate, SourceInspectionError } from '../desktop/sourceInspectionApi.ts'
import type { UserTraceBridgeRequest } from '../desktop/userTraceRequest.ts'
import { MAX_TENSOR_DIMENSIONS } from '../desktop/userTraceRequest.ts'
import { buildConstructorConfig, initialConstructorFields, type ConstructorFieldState } from './constructorConfig.ts'
import { TRUSTED_SOURCE_STORAGE_PREFIX } from './constants.ts'
import { formatBytes, validateTensorDimensions } from './inputConfig.ts'

export type UserTraceDraft = Omit<UserTraceBridgeRequest, 'run_id'>

function trustStorageKey(contentSha256: string) {
  return `${TRUSTED_SOURCE_STORAGE_PREFIX}${contentSha256}`
}

function readRememberedTrust(contentSha256: string) {
  try {
    return window.localStorage.getItem(trustStorageKey(contentSha256)) === 'confirmed'
  } catch (error) {
    console.warn('Could not read the trusted-source preference.', error)
    return false
  }
}

function rememberTrust(contentSha256: string) {
  try {
    window.localStorage.setItem(trustStorageKey(contentSha256), 'confirmed')
  } catch (error) {
    console.warn('Could not remember the trusted-source preference.', error)
  }
}

export function UserTracePanel({
  traceState,
  traceError,
  onRun,
  onCancel,
  onClose,
}: {
  traceState: TraceRunState
  traceError: string | null
  onRun: (request: UserTraceDraft) => void
  onCancel: () => void
  onClose: () => void
}) {
  const [selectedFile, setSelectedFile] = useState<SelectedPythonFile | null>(null)
  const [inspection, setInspection] = useState<InspectModelSourceSuccess | null>(null)
  const [inspectionError, setInspectionError] = useState<SourceInspectionError | null>(null)
  const [selectedClass, setSelectedClass] = useState<string | null>(null)
  const [constructorFields, setConstructorFields] = useState<Record<string, ConstructorFieldState>>({})
  const [trustedCodeConfirmed, setTrustedCodeConfirmed] = useState(false)
  const [dimensions, setDimensions] = useState(['1', '1'])
  const [isSelecting, setIsSelecting] = useState(false)
  const [isInspecting, setIsInspecting] = useState(false)
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const inputValidation = validateTensorDimensions(dimensions)
  const selectedCandidate = inspection?.candidates.find((candidate) => candidate.className === selectedClass) ?? null
  const constructorValidation = buildConstructorConfig(selectedCandidate?.constructor.parameters ?? [], constructorFields)
  const isTraceActive = traceState === 'starting' || traceState === 'running' || traceState === 'cancelling'

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [])

  async function chooseFile() {
    if (isSelecting || isTraceActive) return
    setIsSelecting(true)
    const result = await selectPythonFile()
    if (!mountedRef.current) return
    setIsSelecting(false)
    if (!result.ok) {
      setInspectionError(result.error)
      return
    }
    if (!result.selected) return

    setSelectedFile(result.selected)
    await inspectFile(result.selected)
  }

  async function inspectFile(file: SelectedPythonFile) {
    const requestId = ++requestIdRef.current
    setSelectedClass(null)
    setConstructorFields({})
    setTrustedCodeConfirmed(false)
    setInspection(null)
    setInspectionError(null)
    setIsInspecting(true)
    const inspectionResult = await inspectSelectedPythonFile(file.selectionId)
    if (!mountedRef.current || requestIdRef.current !== requestId) return
    setIsInspecting(false)
    if (inspectionResult.ok) {
      setInspection(inspectionResult)
      setTrustedCodeConfirmed(
        inspectionResult.sourceIdentity
          ? readRememberedTrust(inspectionResult.sourceIdentity.contentSha256)
          : false,
      )
    }
    else setInspectionError(inspectionResult.error)
  }

  function updateDimension(index: number, value: string) {
    setDimensions((current) => current.map((dimension, currentIndex) => currentIndex === index ? value : dimension))
  }

  function submitTrace() {
    const sourceIdentity = inspection?.sourceIdentity
    if (!selectedFile || !selectedClass || !sourceIdentity || !trustedCodeConfirmed || !inputValidation.ok || !constructorValidation.ok || isTraceActive) return
    rememberTrust(sourceIdentity.contentSha256)
    onRun({
      source: {
        selection_id: selectedFile.selectionId,
        class_name: selectedClass,
        content_sha256: sourceIdentity.contentSha256,
      },
      constructor: { args: constructorValidation.args, kwargs: constructorValidation.kwargs },
      inputs: [{
        kind: 'tensor',
        shape: inputValidation.shape,
        dtype: 'float32',
        generator: 'random_normal',
      }],
    })
  }

  function chooseClass(candidate: ModelCandidate) {
    setSelectedClass(candidate.className)
    setConstructorFields(initialConstructorFields(candidate.constructor.parameters))
  }

  function updateConstructorField(name: string, update: Partial<ConstructorFieldState>) {
    setConstructorFields((current) => ({
      ...current,
      [name]: { ...current[name], enabled: current[name]?.enabled ?? false, text: current[name]?.text ?? '', ...update },
    }))
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="user-trace-modal" role="dialog" aria-modal="true" aria-labelledby="user-trace-title">
        <header>
          <div>
            <p className="eyebrow">Local model trace</p>
            <h2 id="user-trace-title">Trace Python Model</h2>
          </div>
          <button type="button" onClick={onClose} disabled={isTraceActive}>Close</button>
        </header>

        <section className="user-trace-section">
          <div className="user-trace-section-heading">
            <div><span>1</span><strong>Python file</strong></div>
            <button type="button" onClick={chooseFile} disabled={isSelecting || isTraceActive}>
              {isSelecting ? 'Opening...' : selectedFile ? 'Choose Different File' : 'Choose Python File'}
            </button>
          </div>
          {selectedFile ? (
            <div className="selected-file-row">
              <strong>{selectedFile.fileName}</strong>
              <span>{formatBytes(selectedFile.sizeBytes)}</span>
              <button type="button" disabled={isInspecting || isTraceActive} onClick={() => inspectFile(selectedFile)}>Inspect Again</button>
            </div>
          ) : <p className="source-muted">Select one local .py file. Selection does not import or execute it.</p>}
          {isInspecting ? <p className="source-muted">Inspecting source...</p> : null}
          {inspectionError ? <div className="source-error"><strong>{inspectionError.title}</strong><p>{inspectionError.message}</p></div> : null}
        </section>

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>2</span><strong>Model class</strong></div></div>
          {inspection?.candidates.length ? (
            <div className="candidate-options">
              {inspection.candidates.map((candidate) => {
                return (
                  <label key={`${candidate.className}-${candidate.lineNumber}`}>
                    <input
                      type="radio"
                      name="model-class"
                      value={candidate.className}
                      checked={selectedClass === candidate.className}
                      disabled={isTraceActive}
                      onChange={() => chooseClass(candidate)}
                    />
                    <span><strong>{candidate.className}</strong><small>{candidate.confidence} · line {candidate.lineNumber}</small></span>
                    <em>{candidate.constructor.parameters.length ? `${candidate.constructor.parameters.length} constructor parameters` : 'no constructor parameters'}</em>
                  </label>
                )
              })}
            </div>
          ) : <p className="source-muted">{inspection ? 'No candidate model classes found.' : 'Choose a file to inspect model classes.'}</p>}
        </section>

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>3</span><strong>Constructor</strong></div></div>
          {selectedCandidate?.constructor.parameters.length ? (
            <div className="constructor-fields">
              {selectedCandidate.constructor.parameters.map((parameter) => {
                const field = constructorFields[parameter.name] ?? { enabled: parameter.required, text: '' }
                return (
                  <div className="constructor-field" key={parameter.name}>
                    <label>
                      {!parameter.required ? (
                        <input type="checkbox" checked={field.enabled} disabled={isTraceActive} onChange={(event) => updateConstructorField(parameter.name, { enabled: event.target.checked })} />
                      ) : null}
                      <strong>{parameter.name}</strong>
                      <small>{parameter.position.replaceAll('_', ' ')}{parameter.annotationText ? ` · ${parameter.annotationText}` : ''}</small>
                    </label>
                    <input
                      type="text"
                      value={field.text}
                      disabled={!field.enabled || isTraceActive}
                      placeholder={parameter.defaultDisplay ?? 'JSON literal'}
                      onChange={(event) => updateConstructorField(parameter.name, { text: event.target.value })}
                    />
                  </div>
                )
              })}
              {!constructorValidation.ok ? <p className="source-error">{constructorValidation.message}</p> : null}
            </div>
          ) : <p className="source-muted">This class has no inspected constructor parameters.</p>}
        </section>

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>4</span><strong>Representative input</strong></div></div>
          <div className="dimension-editor">
            {dimensions.map((dimension, index) => (
              <label key={index}>Dim {index + 1}<input type="number" min="1" step="1" value={dimension} disabled={isTraceActive} onChange={(event) => updateDimension(index, event.target.value)} /></label>
            ))}
            <button type="button" aria-label="Remove dimension" disabled={dimensions.length === 1 || isTraceActive} onClick={() => setDimensions((current) => current.slice(0, -1))}>-</button>
            <button type="button" aria-label="Add dimension" disabled={dimensions.length >= MAX_TENSOR_DIMENSIONS || isTraceActive} onClick={() => setDimensions((current) => [...current, '1'])}>+</button>
          </div>
          <div className="input-spec-summary"><span>CPU</span><span>float32</span><span>random normal</span><strong>{inputValidation.ok ? `${inputValidation.elementCount.toLocaleString()} elements · ${formatBytes(inputValidation.sizeBytes)}` : inputValidation.message}</strong></div>
        </section>

        <section className="user-trace-section trusted-code-confirmation">
          <div className="user-trace-section-heading"><div><span>5</span><strong>Execute trusted code</strong></div></div>
          <p>Tracing imports and executes this local Python file with your user permissions. The worker isolates crashes and state, but it is not a security sandbox. Static inspection does not verify that code is safe.</p>
          <label>
            <input
              type="checkbox"
              checked={trustedCodeConfirmed}
              disabled={!inspection?.sourceIdentity || isTraceActive}
              onChange={(event) => setTrustedCodeConfirmed(event.target.checked)}
            />
            I trust this exact inspected version and allow it to run locally.
          </label>
        </section>

        {traceError ? <div className="source-error"><strong>Trace failed</strong><p>{traceError}</p></div> : null}

        <footer>
          {isTraceActive ? <button type="button" onClick={onCancel} disabled={traceState === 'cancelling'}>{traceState === 'cancelling' ? 'Cancelling...' : 'Cancel'}</button> : null}
          <span className="toolbar-status">{traceState !== 'idle' ? traceState : 'ready'}</span>
          <button type="button" className="primary-button" onClick={submitTrace} disabled={!selectedFile || !selectedClass || !inspection?.sourceIdentity || !trustedCodeConfirmed || !constructorValidation.ok || !inputValidation.ok || isTraceActive}>Run Trace</button>
        </footer>
      </section>
    </div>
  )
}
