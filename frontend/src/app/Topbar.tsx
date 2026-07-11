export function Topbar({
  modelName,
  theme,
  onOpenLoader,
  onFitGraph,
  onToggleTheme,
}: {
  modelName: string
  theme: 'dark' | 'light'
  onOpenLoader: () => void
  onFitGraph: () => void
  onToggleTheme: () => void
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <span>PyTorch Trace</span>
        <strong>{modelName}</strong>
      </div>
      <div className="toolbar">
        <button type="button" onClick={onOpenLoader}>Load JSON</button>
        <button type="button" onClick={onFitGraph}>Fit Graph</button>
      </div>
    </header>
  )
}
