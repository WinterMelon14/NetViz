import { useState } from 'react'
import { friendlyTraceStage, technicalErrorDetails, type TraceFailure } from './traceErrorDetails.ts'

export function TraceErrorCard({
  failure,
  canInspectAgain,
  onRetry,
  onInspectAgain,
  onChooseFile,
  onClose,
}: {
  failure: TraceFailure
  canInspectAgain: boolean
  onRetry: () => void
  onInspectAgain: () => void
  onChooseFile: () => void
  onClose: () => void
}) {
  const [copyStatus, setCopyStatus] = useState<string | null>(null)
  const isSourceFailure = failure.error.code === 'source_changed' || failure.error.code === 'source_reinspection_required'
  const isSelectionFailure = failure.error.code === 'selected_file_unavailable'
  const details = technicalErrorDetails(failure, import.meta.env.DEV)

  async function copyDetails() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(details, null, 2))
      setCopyStatus('Copied')
    } catch (error) {
      console.warn('Could not copy trace error details.', error)
      setCopyStatus('Copy failed')
    }
  }

  return (
    <section className="trace-error-card" aria-live="polite">
      <div>
        <p className="eyebrow">{failure.error.code === 'cancelled' ? 'Trace cancelled' : failure.error.code === 'timeout' ? 'Trace timed out' : 'Trace failed'}</p>
        <h3>{failure.error.title}</h3>
        <p>{failure.error.message}</p>
        <span>{friendlyTraceStage(failure.error.stage)}</span>
      </div>
      <div className="trace-error-actions">
        {isSourceFailure && canInspectAgain ? <button type="button" onClick={onInspectAgain}>Inspect Again</button> : null}
        {isSelectionFailure ? <button type="button" onClick={onChooseFile}>Choose Different File</button> : null}
        {!isSourceFailure && !isSelectionFailure ? <button type="button" onClick={onRetry}>Try Again</button> : null}
        <button type="button" onClick={onClose}>Close</button>
      </div>
      <details className="trace-error-details">
        <summary>Technical details</summary>
        <pre>{JSON.stringify(details, null, 2)}</pre>
        <button type="button" onClick={copyDetails}>Copy Details</button>
        {copyStatus ? <span role="status">{copyStatus}</span> : null}
      </details>
    </section>
  )
}
