import { useEffect, useRef, useState } from 'react'
import type { TraceRunState } from '../desktop/desktopTraceApi.ts'
import { inspectSelectedPythonFile, selectPythonFile } from '../desktop/selectedFileApi.ts'
import type { SelectedPythonFile } from '../desktop/selectedFileApi.ts'
import type { InspectModelSourceSuccess, ModelCandidate, SourceInspectionError } from '../desktop/sourceInspectionApi.ts'
import type { UserTraceBridgeRequest } from '../desktop/userTraceRequest.ts'
import { MAX_TENSOR_DIMENSIONS } from '../desktop/userTraceRequest.ts'
import { formatBytes, validateTensorDimensions } from './inputConfig.ts'

export type UserTraceDraft = Omit<UserTraceBridgeRequest, 'run_id'>

function canConstruct(candidate: ModelCandidate) {
  return candidate.constructor.supportsNoArgumentConstruction === true
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
  const [dimensions, setDimensions] = useState(['1', '1'])
  const [isSelecting, setIsSelecting] = useState(false)
  const [isInspecting, setIsInspecting] = useState(false)
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const inputValidation = validateTensorDimensions(dimensions)
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

    const requestId = ++requestIdRef.current
    setSelectedFile(result.selected)
    setSelectedClass(null)
    setInspection(null)
    setInspectionError(null)
    setIsInspecting(true)
    const inspectionResult = await inspectSelectedPythonFile(result.selected.selectionId)
    if (!mountedRef.current || requestIdRef.current !== requestId) return
    setIsInspecting(false)
    if (inspectionResult.ok) setInspection(inspectionResult)
    else setInspectionError(inspectionResult.error)
  }

  function updateDimension(index: number, value: string) {
    setDimensions((current) => current.map((dimension, currentIndex) => currentIndex === index ? value : dimension))
  }

  function submitTrace() {
    if (!selectedFile || !selectedClass || !inputValidation.ok || isTraceActive) return
    onRun({
      source: { selection_id: selectedFile.selectionId, class_name: selectedClass },
      constructor: { args: [], kwargs: {} },
      inputs: [{
        kind: 'tensor',
        shape: inputValidation.shape,
        dtype: 'float32',
        generator: 'random_normal',
      }],
    })
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
                const supported = canConstruct(candidate)
                return (
                  <label key={`${candidate.className}-${candidate.lineNumber}`} className={supported ? '' : 'candidate-option--disabled'}>
                    <input
                      type="radio"
                      name="model-class"
                      value={candidate.className}
                      checked={selectedClass === candidate.className}
                      disabled={!supported || isTraceActive}
                      onChange={() => setSelectedClass(candidate.className)}
                    />
                    <span><strong>{candidate.className}</strong><small>{candidate.confidence} · line {candidate.lineNumber}</small></span>
                    <em>{supported ? 'no-argument constructor' : 'constructor arguments unsupported'}</em>
                  </label>
                )
              })}
            </div>
          ) : <p className="source-muted">{inspection ? 'No candidate model classes found.' : 'Choose a file to inspect model classes.'}</p>}
        </section>

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>3</span><strong>Representative input</strong></div></div>
          <div className="dimension-editor">
            {dimensions.map((dimension, index) => (
              <label key={index}>Dim {index + 1}<input type="number" min="1" step="1" value={dimension} disabled={isTraceActive} onChange={(event) => updateDimension(index, event.target.value)} /></label>
            ))}
            <button type="button" aria-label="Remove dimension" disabled={dimensions.length === 1 || isTraceActive} onClick={() => setDimensions((current) => current.slice(0, -1))}>-</button>
            <button type="button" aria-label="Add dimension" disabled={dimensions.length >= MAX_TENSOR_DIMENSIONS || isTraceActive} onClick={() => setDimensions((current) => [...current, '1'])}>+</button>
          </div>
          <div className="input-spec-summary"><span>CPU</span><span>float32</span><span>random normal</span><strong>{inputValidation.ok ? `${inputValidation.elementCount.toLocaleString()} elements · ${formatBytes(inputValidation.sizeBytes)}` : inputValidation.message}</strong></div>
        </section>

        {traceError ? <div className="source-error"><strong>Trace failed</strong><p>{traceError}</p></div> : null}

        <footer>
          {isTraceActive ? <button type="button" onClick={onCancel} disabled={traceState === 'cancelling'}>{traceState === 'cancelling' ? 'Cancelling...' : 'Cancel'}</button> : null}
          <span className="toolbar-status">{traceState !== 'idle' ? traceState : 'ready'}</span>
          <button type="button" className="primary-button" onClick={submitTrace} disabled={!selectedFile || !selectedClass || !inputValidation.ok || isTraceActive}>Run Trace</button>
        </footer>
      </section>
    </div>
  )
}
