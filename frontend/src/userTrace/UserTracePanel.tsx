import { useEffect, useRef, useState } from 'react'
import type { TraceRunState } from '../desktop/desktopTraceApi.ts'
import { inspectPythonSource, registerInlinePythonSource, releasePythonSource, selectPythonFile } from '../desktop/sourceApi.ts'
import type { PythonSource } from '../desktop/sourceApi.ts'
import type { InspectModelSourceSuccess, ModelCandidate, SourceInspectionError } from '../desktop/sourceInspectionApi.ts'
import type { UserTraceBridgeRequest } from '../desktop/userTraceRequest.ts'
import { buildConstructorConfig, initialConstructorFields, type ConstructorFieldState } from './constructorConfig.ts'
import { TRUSTED_SOURCE_STORAGE_PREFIX } from './constants.ts'
import { formatBytes } from './inputConfig.ts'
import { LoadingIndicator } from '../components/LoadingIndicator.tsx'
import { TraceErrorCard } from './TraceErrorCard.tsx'
import type { TraceFailure } from './traceErrorDetails.ts'
import { createInputDrafts, validateInputDrafts, type TensorInputDraft } from './inputDrafts.ts'
import { TensorInputEditor } from './TensorInputEditor.tsx'
import { MAX_TOTAL_INPUT_BYTES } from './constants.ts'
import { suggestFromTraceError } from './errorInputSuggestions.ts'
import { SourceInputStep, type SourceMode } from './SourceInputStep.tsx'

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
  traceFailure,
  onRun,
  onCancel,
  onClearError,
  onClose,
}: {
  traceState: TraceRunState
  traceFailure: TraceFailure | null
  onRun: (request: UserTraceDraft) => void
  onCancel: () => void
  onClearError: () => void
  onClose: () => void
}) {
  const [sourceMode, setSourceMode] = useState<SourceMode>('file')
  const [source, setSource] = useState<PythonSource | null>(null)
  const [pastedText, setPastedText] = useState('')
  const [inspection, setInspection] = useState<InspectModelSourceSuccess | null>(null)
  const [inspectionError, setInspectionError] = useState<SourceInspectionError | null>(null)
  const [selectedClass, setSelectedClass] = useState<string | null>(null)
  const [constructorFields, setConstructorFields] = useState<Record<string, ConstructorFieldState>>({})
  const [trustedCodeConfirmed, setTrustedCodeConfirmed] = useState(false)
  const [inputDrafts, setInputDrafts] = useState<TensorInputDraft[]>([])
  const [inputSignatureError, setInputSignatureError] = useState<string | null>(null)
  const [useProviderInputs, setUseProviderInputs] = useState(false)
  const [isSelecting, setIsSelecting] = useState(false)
  const [isInspecting, setIsInspecting] = useState(false)
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const sourceRef = useRef<PythonSource | null>(null)
  const failureCode = traceFailure?.error.code
  const isSourceInvalidated = failureCode === 'source_changed' || failureCode === 'source_reinspection_required'
  const isSelectionInvalidated = failureCode === 'source_unavailable' || failureCode === 'selected_file_unavailable'
  const activeSource = isSelectionInvalidated ? null : source
  const activeInspection = isSourceInvalidated || isSelectionInvalidated ? null : inspection
  const selectedCandidate = activeInspection?.candidates.find((candidate) => candidate.className === selectedClass) ?? null
  const constructorValidation = buildConstructorConfig(selectedCandidate?.constructor.parameters ?? [], constructorFields)
  const inputValidation = validateInputDrafts(inputDrafts, MAX_TOTAL_INPUT_BYTES)
  const errorInputSuggestion = traceFailure ? suggestFromTraceError(traceFailure.error.message, inputDrafts) : null
  const isTraceActive = traceState === 'starting' || traceState === 'running' || traceState === 'cancelling'

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
      const currentSource = sourceRef.current
      if (currentSource) void releasePythonSource(currentSource.sourceId)
    }
  }, [])

  useEffect(() => {
    sourceRef.current = source
  }, [source])

  function resetSourceConfiguration() {
    requestIdRef.current += 1
    setSelectedClass(null)
    setConstructorFields({})
    setInputDrafts([])
    setInputSignatureError(null)
    setUseProviderInputs(false)
    setTrustedCodeConfirmed(false)
    setInspection(null)
    setInspectionError(null)
    setIsInspecting(false)
    onClearError()
  }

  async function releaseSourceHandle(handle: PythonSource | null) {
    if (!handle) return true
    const result = await releasePythonSource(handle.sourceId)
    if (!result.ok && mountedRef.current) {
      setInspectionError(result.error)
      return false
    }
    return true
  }

  async function chooseFile() {
    if (isSelecting || isTraceActive) return
    onClearError()
    setIsSelecting(true)
    const result = await selectPythonFile()
    if (!mountedRef.current) return
    setIsSelecting(false)
    if (!result.ok) {
      setInspectionError(result.error)
      return
    }
    if (!result.source) return
    if (source && !window.confirm('Discard the current model source and configuration?')) {
      await releaseSourceHandle(result.source)
      return
    }
    if (!await releaseSourceHandle(source)) {
      await releaseSourceHandle(result.source)
      return
    }
    setSource(result.source)
    await inspectSource(result.source)
  }

  async function inspectSource(nextSource: PythonSource) {
    const requestId = ++requestIdRef.current
    onClearError()
    setSelectedClass(null)
    setConstructorFields({})
    setInputDrafts([])
    setInputSignatureError(null)
    setUseProviderInputs(false)
    setTrustedCodeConfirmed(false)
    setInspection(null)
    setInspectionError(null)
    setIsInspecting(true)
    const inspectionResult = await inspectPythonSource(nextSource.sourceId)
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

  async function inspectPaste() {
    if (isInspecting || isTraceActive) return
    resetSourceConfiguration()
    setIsInspecting(true)
    const registered = await registerInlinePythonSource(pastedText)
    if (!mountedRef.current) return
    if (!registered.ok) {
      setIsInspecting(false)
      setInspectionError(registered.error)
      return
    }
    setSource(registered.source)
    await inspectSource(registered.source)
  }

  function changePastedText(value: string) {
    setPastedText(value)
    if (source || inspection) {
      const previous = source
      setSource(null)
      resetSourceConfiguration()
      void releaseSourceHandle(previous)
    }
  }

  function clearPaste() {
    const previous = source
    setPastedText('')
    setSource(null)
    resetSourceConfiguration()
    void releaseSourceHandle(previous)
  }

  async function changeSourceMode(nextMode: SourceMode) {
    if (nextMode === sourceMode) return
    if ((source || pastedText || inspection) && !window.confirm('Discard the current model source and configuration?')) return
    if (!await releaseSourceHandle(source)) return
    setSourceMode(nextMode)
    setSource(null)
    setPastedText('')
    resetSourceConfiguration()
  }

  function updateInputDraft(nextDraft: TensorInputDraft) {
    if (!isSourceInvalidated && !isSelectionInvalidated) onClearError()
    setInputDrafts((current) => current.map((draft) => draft.id === nextDraft.id ? nextDraft : draft))
  }

  function applyErrorInputSuggestion() {
    if (!errorInputSuggestion?.draftId || errorInputSuggestion.dimensionIndex === undefined || errorInputSuggestion.value === undefined) return
    setInputDrafts((current) => current.map((draft) => draft.id === errorInputSuggestion.draftId ? {
      ...draft,
      dimensions: draft.dimensions.map((dimension, index) => index === errorInputSuggestion.dimensionIndex ? errorInputSuggestion.value! : dimension),
    } : draft))
    onClearError()
  }

  function submitTrace() {
    const sourceIdentity = activeInspection?.sourceIdentity
    if (!activeSource || !selectedClass || !sourceIdentity || !trustedCodeConfirmed || (!useProviderInputs && (Boolean(inputSignatureError) || !inputValidation.ok)) || !constructorValidation.ok || isTraceActive) return
    const configuredInputs = inputValidation.ok ? inputValidation.inputs.map(({ draft, validation }) => ({
      kind: 'tensor' as const, parameter_name: draft.parameterName, shape: validation.shape, dtype: draft.dtype, generator: draft.generator,
    })) : []
    onClearError()
    rememberTrust(sourceIdentity.contentSha256)
    onRun({
      source: {
        source_id: activeSource.sourceId,
        class_name: selectedClass,
        content_sha256: sourceIdentity.contentSha256,
      },
      constructor: { args: constructorValidation.args, kwargs: constructorValidation.kwargs },
      inputs: useProviderInputs ? [] : configuredInputs,
      input_provider: useProviderInputs ? { function_name: 'netviz_example_inputs', parameter_names: inputDrafts.map((draft) => draft.parameterName) } : null,
    })
  }

  function chooseClass(candidate: ModelCandidate) {
    if (!isSourceInvalidated && !isSelectionInvalidated) onClearError()
    setSelectedClass(candidate.className)
    setConstructorFields(initialConstructorFields(candidate.constructor.parameters))
    const drafts = createInputDrafts(candidate.forward)
    setInputDrafts(drafts.ok ? drafts.drafts : [])
    setInputSignatureError(drafts.ok ? null : drafts.message)
    setUseProviderInputs(false)
  }

  function updateConstructorField(name: string, update: Partial<ConstructorFieldState>) {
    if (!isSourceInvalidated && !isSelectionInvalidated) onClearError()
    setConstructorFields((current) => ({
      ...current,
      [name]: { ...current[name], enabled: current[name]?.enabled ?? false, text: current[name]?.text ?? '', ...update },
    }))
  }

  const progressText = traceState === 'starting'
    ? 'Starting trace...'
    : traceState === 'cancelling'
      ? 'Cancelling...'
      : 'Tracing model...'

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

        {isTraceActive ? (
          <section className="trace-progress" aria-busy="true" aria-live="polite">
            <LoadingIndicator />
            <div>
              <h3>{selectedClass ? `Tracing ${selectedClass}` : 'Tracing model'}</h3>
              <p>{progressText}</p>
              <span>Executing the model locally...</span>
            </div>
            <button type="button" onClick={onCancel} disabled={traceState === 'cancelling'}>
              {traceState === 'cancelling' ? 'Cancelling...' : 'Cancel'}
            </button>
          </section>
        ) : (
          <>
        <SourceInputStep
          mode={sourceMode}
          source={activeSource}
          pastedText={pastedText}
          inspectionError={inspectionError}
          isSelecting={isSelecting}
          isInspecting={isInspecting}
          disabled={isTraceActive}
          onModeChange={(mode) => void changeSourceMode(mode)}
          onChooseFile={() => void chooseFile()}
          onInspectSource={() => { if (activeSource) void inspectSource(activeSource) }}
          onPastedTextChange={changePastedText}
          onInspectPaste={() => void inspectPaste()}
          onClearPaste={clearPaste}
        />

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>2</span><strong>Model class</strong></div></div>
          {activeInspection?.candidates.length ? (
            <div className="candidate-options">
              {activeInspection.candidates.map((candidate) => {
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
          ) : <p className="source-muted">{activeInspection ? 'No candidate model classes found.' : 'Choose or paste source to inspect model classes.'}</p>}
        </section>

        <section className="user-trace-section">
          <div className="user-trace-section-heading"><div><span>3</span><strong>Constructor</strong></div></div>
          {selectedCandidate?.constructor.parameters.length ? (
            <div className="constructor-fields">
              {selectedCandidate.constructor.parameters.map((parameter) => {
                const field = constructorFields[parameter.name] ?? { enabled: true, text: '' }
                return (
                  <div className="constructor-field" key={parameter.name}>
                    <label>
                      <strong>{parameter.name}</strong>
                      <small>{parameter.typeText ?? parameter.annotationText ?? ''}</small>
                    </label>
                    <input
                      type="text"
                      value={field.text}
                      disabled={!field.enabled || isTraceActive}
                      placeholder={field.enabled ? 'JSON literal' : 'Uses source default'}
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
          <div className="user-trace-section-heading"><div><span>4</span><strong>Representative inputs</strong></div></div>
          {activeInspection?.exampleInputProvider ? <label className="provider-input-toggle"><input type="checkbox" checked={useProviderInputs} onChange={(event) => { onClearError(); setUseProviderInputs(event.target.checked) }} />Use tensors from netviz_example_inputs()</label> : null}
          {useProviderInputs ? <p className="source-muted">The trusted worker will execute and validate the model-provided tensor sequence.</p> : null}
          {!useProviderInputs && inputSignatureError ? <div className="source-error"><strong>Unsupported forward signature</strong><p>{inputSignatureError}</p></div> : null}
          {!useProviderInputs && !inputSignatureError && inputDrafts.length === 0 ? <p className="source-muted">This model has no required positional inputs.</p> : null}
          {!useProviderInputs ? inputDrafts.map((draft) => <TensorInputEditor key={draft.id} draft={draft} disabled={isTraceActive} onChange={updateInputDraft} />) : null}
          {!useProviderInputs && !inputSignatureError ? <div className="input-total"><span>Total input allocation</span><strong>{inputValidation.ok ? formatBytes(inputValidation.totalBytes) : inputValidation.message}</strong></div> : null}
          {errorInputSuggestion ? <div className="input-recovery-suggestion"><span>{errorInputSuggestion.message}</span>{errorInputSuggestion.value ? <button type="button" onClick={applyErrorInputSuggestion}>Apply Suggestion</button> : null}</div> : null}
        </section>

        <section className="user-trace-section trusted-code-confirmation">
          <div className="user-trace-section-heading"><div><span>5</span><strong>Execute trusted code</strong></div></div>
          <p>Tracing imports and executes this local Python source with your user permissions. The worker isolates crashes and state, but it is not a security sandbox. Static inspection does not verify that code is safe.</p>
          <label>
            <input
              type="checkbox"
              checked={trustedCodeConfirmed}
              disabled={!activeInspection?.sourceIdentity || isTraceActive}
              onChange={(event) => {
                onClearError()
                setTrustedCodeConfirmed(event.target.checked)
              }}
            />
            I trust this exact inspected version and allow it to run locally.
          </label>
        </section>

        {traceFailure ? (
          <TraceErrorCard
            failure={traceFailure}
            canInspectAgain={Boolean(source || (sourceMode === 'paste' && pastedText))}
            onRetry={submitTrace}
            onInspectAgain={() => {
              if (source) void inspectSource(source)
              else if (sourceMode === 'paste') void inspectPaste()
            }}
            onChooseFile={() => sourceMode === 'file' ? void chooseFile() : void inspectPaste()}
            onClose={onClose}
          />
        ) : null}

        <footer>
          <span className="toolbar-status">{traceState !== 'idle' ? traceState : 'ready'}</span>
          <button type="button" className="primary-button" onClick={submitTrace} disabled={!activeSource || !selectedClass || !activeInspection?.sourceIdentity || !trustedCodeConfirmed || (!useProviderInputs && (Boolean(inputSignatureError) || !inputValidation.ok)) || !constructorValidation.ok || isTraceActive}>Run Trace</button>
        </footer>
          </>
        )}
      </section>
    </div>
  )
}
