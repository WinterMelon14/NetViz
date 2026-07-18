import { graphLayeringStrategies, type GraphSettings, type GraphLayeringStrategy } from './graphSettings'

function strategyLabel(strategy: GraphLayeringStrategy) {
  if (strategy === 'DF_MODEL_ORDER') return 'DF Model Order'
  if (strategy === 'NETWORK_SIMPLEX') return 'Network Simplex'
  return 'Longest Path Source'
}

export function SettingsModal({
  isOpen,
  settings,
  onChange,
  onClose,
}: {
  isOpen: boolean
  settings: GraphSettings
  onChange: (settings: GraphSettings) => void
  onClose: () => void
}) {
  if (!isOpen) return null

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="settings-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <h2 id="settings-title">Settings</h2>
          <button type="button" className="icon-button" aria-label="Close settings" title="Close settings" onClick={onClose}>
            <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" aria-hidden="true">
              <path fill="currentColor" d="m13.4 12 5.3-5.3-1.4-1.4-5.3 5.3-5.3-5.3-1.4 1.4 5.3 5.3-5.3 5.3 1.4 1.4 5.3-5.3 5.3 5.3 1.4-1.4z" />
            </svg>
          </button>
        </header>

        <section className="settings-section" aria-labelledby="graph-settings-title">
          <h3 id="graph-settings-title">Graph</h3>
          <div className="settings-field">
            <span>Graph Arrangement Pattern</span>
            <div className="segmented-control" role="radiogroup" aria-label="Graph Arrangement Pattern">
              {graphLayeringStrategies.map((strategy) => (
                <button
                  key={strategy}
                  type="button"
                  role="radio"
                  aria-checked={settings.layeringStrategy === strategy}
                  onClick={() => onChange({ ...settings, layeringStrategy: strategy })}
                >
                  {strategyLabel(strategy)}
                </button>
              ))}
            </div>
          </div>
        </section>
      </section>
    </div>
  )
}
