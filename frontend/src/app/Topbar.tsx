export function Topbar({
  modelName,
  onOpenLoader,
  onFitGraph,
}: {
  modelName: string
  onOpenLoader: () => void
  onFitGraph: () => void
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <span>PyTorch Trace</span>
        <strong>{modelName}</strong>
      </div>
      <div className="toolbar">
        <button type="button" onClick={onOpenLoader}>Load JSON</button>
        <button type="button" onClick={onFitGraph}><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 32 32">
  <path d="M0 0h32v32H0z" fill="none" />
  <path fill="currentColor" d="M18 28A12 12 0 1 0 6 16v6.2l-3.6-3.6L1 20l6 6l6-6l-1.4-1.4L8 22.2V16a10 10 0 1 1 10 10Z" />
</svg></button>
      
      </div>
    </header>
  )
}
