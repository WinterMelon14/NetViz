export function Topbar({
  modelName,
  onOpenUserTrace,
  onFitGraph,
  onLoadFixture,
}: {
  modelName: string
  onOpenUserTrace: () => void
  onFitGraph: () => void
  onLoadFixture?: () => void
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <span>NetViz</span>
        <strong>{modelName}</strong>
      </div>
      <div className="toolbar">
        {onLoadFixture ? <button type="button" onClick={onLoadFixture}>Load Trace Fixture</button> : null}
        <button type="button" onClick={onOpenUserTrace}>Trace Model</button>
        <button type="button" aria-label="Fit graph" title="Fit graph" onClick={onFitGraph}><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 32 32">
  <path d="M0 0h32v32H0z" fill="none" />
  <path fill="currentColor" d="M18 28A12 12 0 1 0 6 16v6.2l-3.6-3.6L1 20l6 6l6-6l-1.4-1.4L8 22.2V16a10 10 0 1 1 10 10Z" />
</svg></button>
      
      </div>
    </header>
  )
}
