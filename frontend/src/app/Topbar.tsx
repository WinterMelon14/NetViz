export function Topbar({
  modelName,
  onOpenUserTrace,
  onFitGraph,
  onOpenSettings,
}: {
  modelName: string
  onOpenUserTrace: () => void
  onFitGraph: () => void
  onOpenSettings: () => void
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <span>NetViz</span>
        <strong>{modelName}</strong>
      </div>
      <div className="toolbar">
        <button type="button" onClick={onOpenUserTrace}>Trace Model</button>
        <button type="button" aria-label="Fit graph" title="Fit graph" onClick={onFitGraph}><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 32 32">
  <path d="M0 0h32v32H0z" fill="none" />
  <path fill="currentColor" d="M18 28A12 12 0 1 0 6 16v6.2l-3.6-3.6L1 20l6 6l6-6l-1.4-1.4L8 22.2V16a10 10 0 1 1 10 10Z" />
</svg></button>
        <button type="button" className="icon-button" aria-label="Settings" title="Settings" onClick={onOpenSettings}>
          <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" aria-hidden="true">
            <path fill="currentColor" d="m19.4 13.5.1-1.5-.1-1.5 2-1.5-2-3.5-2.4 1a8 8 0 0 0-2.6-1.5L14 2h-4l-.4 2.5A8 8 0 0 0 7 6L4.6 5 2.6 8.5l2 1.5-.1 1.5.1 1.5-2 1.5 2 3.5 2.4-1a8 8 0 0 0 2.6 1.5L10 22h4l.4-2.5A8 8 0 0 0 17 18l2.4 1 2-3.5zM12 15.5A3.5 3.5 0 1 1 12 8a3.5 3.5 0 0 1 0 7.5m0-2A1.5 1.5 0 1 0 12 10a1.5 1.5 0 0 0 0 3.5" />
          </svg>
        </button>
      </div>
    </header>
  )
}
