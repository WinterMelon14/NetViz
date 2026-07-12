import { useEffect, useRef, useState } from 'react'
import {
  inspectModelSource,
  type FunctionParameter,
  type InspectModelSourceSuccess,
  type ModelCandidate,
  type SourceInspectionState,
} from '../desktop/sourceInspectionApi'

function formatDefault(parameter: FunctionParameter) {
  if (Object.prototype.hasOwnProperty.call(parameter, 'defaultValue')) {
    return ` = ${JSON.stringify(parameter.defaultValue)}`
  }
  if (parameter.defaultDisplay) {
    return ` = ${parameter.defaultDisplay}`
  }
  return ''
}

function formatParameter(parameter: FunctionParameter) {
  const annotation = parameter.annotationText ? `: ${parameter.annotationText}` : ''
  const required = parameter.required ? 'required' : 'optional'
  return `${parameter.name}${annotation}${formatDefault(parameter)} (${parameter.position}, ${required})`
}

function formatNoArgSupport(value: ModelCandidate['constructor']['supportsNoArgumentConstruction']) {
  if (value === true) return 'yes'
  if (value === false) return 'no'
  return 'unknown'
}

function CandidateList({ result }: { result: InspectModelSourceSuccess }) {
  if (result.candidates.length === 0) {
    return <p className="empty-note">No candidate model classes found by static inspection.</p>
  }

  return (
    <div className="source-candidates">
      {result.candidates.map((candidate) => (
        <section key={`${candidate.className}-${candidate.lineNumber}`} className="source-candidate">
          <header>
            <div>
              <h3>{candidate.className}</h3>
              <p>line {candidate.lineNumber} · {candidate.confidence} static candidate</p>
            </div>
            <span>{formatNoArgSupport(candidate.constructor.supportsNoArgumentConstruction)} no-arg constructor</span>
          </header>

          {candidate.bases.length > 0 ? (
            <p className="source-muted">Bases: {candidate.bases.join(', ')}</p>
          ) : null}

          {candidate.constructor.parameters.length > 0 ? (
            <div>
              <strong>Constructor parameters</strong>
              <ul>
                {candidate.constructor.parameters.map((parameter) => (
                  <li key={`${candidate.className}-init-${parameter.name}`}>{formatParameter(parameter)}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {candidate.forward ? (
            <div>
              <strong>Forward signature</strong>
              {candidate.forward.isAsync ? <p className="source-warning">async forward detected</p> : null}
              {candidate.forward.parameters.length > 0 ? (
                <ul>
                  {candidate.forward.parameters.map((parameter) => (
                    <li key={`${candidate.className}-forward-${parameter.name}`}>{formatParameter(parameter)}</li>
                  ))}
                </ul>
              ) : (
                <p className="source-muted">No named forward parameters beyond self.</p>
              )}
              {candidate.forward.hasVarArgs ? <p className="source-muted">Includes *{candidate.forward.varArgName ?? 'args'}</p> : null}
              {candidate.forward.hasVarKwargs ? <p className="source-muted">Includes **{candidate.forward.varKwargName ?? 'kwargs'}</p> : null}
            </div>
          ) : (
            <p className="source-warning">No directly declared forward method found.</p>
          )}
        </section>
      ))}
    </div>
  )
}

export function SourceInspectionPanel({ onClose }: { onClose: () => void }) {
  const [sourceText, setSourceText] = useState('')
  const [inspectionState, setInspectionState] = useState<SourceInspectionState>({ status: 'idle' })
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const isInspecting = inspectionState.status === 'inspecting'

  useEffect(() => () => {
    mountedRef.current = false
    requestIdRef.current += 1
  }, [])

  function runInspection() {
    if (isInspecting) return
    const requestId = ++requestIdRef.current
    setInspectionState({ status: 'inspecting' })
    inspectModelSource(sourceText)
      .then((result) => {
        if (!mountedRef.current || requestIdRef.current !== requestId) return
        if (result.ok) {
          setInspectionState({ status: 'succeeded', result })
        } else {
          setInspectionState({ status: 'failed', error: result.error })
        }
      })
      .catch((error: unknown) => {
        if (!mountedRef.current || requestIdRef.current !== requestId) return
        setInspectionState({
          status: 'failed',
          error: {
            code: 'source_inspection_failed',
            title: 'Source inspection failed',
            message: error instanceof Error ? error.message : 'The source inspection request failed unexpectedly.',
            stage: 'source_inspection_bridge',
          },
        })
      })
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="source-modal" aria-label="Model source inspection">
        <header>
          <div>
            <p className="eyebrow">Static inspection</p>
            <h2>Model Source</h2>
          </div>
          <button type="button" onClick={onClose}>Close</button>
        </header>

        <textarea
          value={sourceText}
          onChange={(event) => setSourceText(event.target.value)}
          placeholder="Paste a self-contained PyTorch model source file"
          spellCheck={false}
        />

        {inspectionState.status === 'failed' ? (
          <div className="source-error">
            <strong>{inspectionState.error.title}</strong>
            <p>{inspectionState.error.message}</p>
            {typeof inspectionState.error.details?.line === 'number' ? (
              <p>Line {inspectionState.error.details.line}, column {String(inspectionState.error.details.column ?? '?')}</p>
            ) : null}
            {typeof inspectionState.error.details?.sourceLine === 'string' ? (
              <code>{inspectionState.error.details.sourceLine}</code>
            ) : null}
          </div>
        ) : null}

        <footer>
          <button type="button" className="primary-button" onClick={runInspection} disabled={isInspecting}>
            {isInspecting ? 'Inspecting...' : 'Inspect Source'}
          </button>
        </footer>

        {inspectionState.status === 'succeeded' ? (
          <div className="source-results">
            {inspectionState.result.warnings.length > 0 ? (
              <div className="source-warning-list">
                {inspectionState.result.warnings.map((warning) => (
                  <p key={`${warning.code}-${warning.lineNumber ?? warning.message}`}>{warning.message}</p>
                ))}
              </div>
            ) : null}
            <CandidateList result={inspectionState.result} />
          </div>
        ) : inspectionState.status === 'idle' ? (
          <p className="empty-note">Inspection uses Python AST only. It does not run, import, instantiate, or trace pasted source.</p>
        ) : null}
      </section>
    </div>
  )
}
