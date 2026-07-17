import type { ReactNode } from 'react'

export function EmptyTraceState({
  onTraceModel,
  children,
}: {
  onTraceModel: () => void
  children?: ReactNode
}) {
  return (
    <main className="app-shell empty-trace-shell light">
      <header className="empty-trace-header">
        <strong>NetViz</strong>
      </header>
      <section className="empty-trace-state" aria-labelledby="empty-trace-title">
        <p className="eyebrow">NetViz</p>
        <h1 id="empty-trace-title">No model traced yet</h1>
        <p>Choose a local Python model to begin.</p>
        <button type="button" className="primary-button" onClick={onTraceModel}>Trace Model</button>
      </section>
      {children}
    </main>
  )
}
