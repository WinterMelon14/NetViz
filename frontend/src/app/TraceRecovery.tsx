export function TraceRecovery({
  message,
  onRetry,
  onOpenLoader,
}: {
  message: string
  onRetry: () => void
  onOpenLoader: () => void
}) {
  return (
    <section className="trace-recovery" aria-label="Trace loading error">
      <p className="eyebrow">Trace unavailable</p>
      <h2>Could not display the graph</h2>
      <p>{message}</p>
      <div>
        <button type="button" onClick={onRetry}>Retry</button>
        <button type="button" className="primary-button" onClick={onOpenLoader}>Load Trace</button>
      </div>
    </section>
  )
}
