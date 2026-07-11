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
        <button
          type="button"
          className="icon-button"
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
          onClick={onToggleTheme}
        >
          <span className="moon-icon" aria-hidden="true"></span>
        </button>
      </div>
    </header>
  )
}
