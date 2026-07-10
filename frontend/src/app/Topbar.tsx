import type { LayoutDirection } from '../graph/buildLayout'

export function Topbar({
  modelName,
  layoutDirection,
  theme,
  onOpenLoader,
  onFitGraph,
  onToggleLayout,
  onToggleTheme,
}: {
  modelName: string
  layoutDirection: LayoutDirection
  theme: 'dark' | 'light'
  onOpenLoader: () => void
  onFitGraph: () => void
  onToggleLayout: () => void
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
          aria-label={layoutDirection === 'left-right' ? 'Switch to top to bottom layout' : 'Switch to left to right layout'}
          title={layoutDirection === 'left-right' ? 'Top to bottom' : 'Left to right'}
          onClick={onToggleLayout}
        >
          <span className={`layout-icon layout-icon--${layoutDirection}`} aria-hidden="true"></span>
        </button>
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
