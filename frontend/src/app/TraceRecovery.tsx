export function TraceRecovery({
  message,
  onTraceModel,
}: {
  message: string
  onTraceModel: () => void
}) {
  return (
    <section className="trace-recovery" aria-label="Trace loading error">
      <p className="eyebrow">Trace unavailable</p>
      <h2>Could not display the graph</h2>
      <p>{message}</p>
      <div>
        <button type="button" className="primary-button" onClick={onTraceModel}>Trace Model</button>
      </div>
    </section>
  )
}
