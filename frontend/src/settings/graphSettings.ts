export const graphLayeringStrategies = [
  'DF_MODEL_ORDER',
  'NETWORK_SIMPLEX',
  'LONGEST_PATH_SOURCE',
] as const

export type GraphLayeringStrategy = typeof graphLayeringStrategies[number]

export type GraphSettings = {
  layeringStrategy: GraphLayeringStrategy
}

export const defaultGraphSettings: GraphSettings = {
  layeringStrategy: 'DF_MODEL_ORDER',
}

const storageKey = 'netviz:graph-settings'

function warnSettingsStorageFailure(error: unknown) {
  if (!import.meta.env?.DEV) return
  console.warn('Graph settings could not be persisted.', error)
}

export function isGraphLayeringStrategy(value: unknown): value is GraphLayeringStrategy {
  return typeof value === 'string'
    && graphLayeringStrategies.includes(value as GraphLayeringStrategy)
}

export function normalizeGraphSettings(value: unknown): GraphSettings {
  if (!value || typeof value !== 'object') return defaultGraphSettings
  const candidate = value as Record<string, unknown>
  return {
    layeringStrategy: isGraphLayeringStrategy(candidate.layeringStrategy)
      ? candidate.layeringStrategy
      : defaultGraphSettings.layeringStrategy,
  }
}

export function loadGraphSettings(): GraphSettings {
  try {
    if (typeof window === 'undefined') return defaultGraphSettings
    const stored = window.localStorage.getItem(storageKey)
    if (!stored) return defaultGraphSettings
    return normalizeGraphSettings(JSON.parse(stored))
  } catch (error) {
    warnSettingsStorageFailure(error)
    return defaultGraphSettings
  }
}

export function saveGraphSettings(settings: GraphSettings) {
  try {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(storageKey, JSON.stringify(settings))
  } catch (error) {
    warnSettingsStorageFailure(error)
    // Settings persistence is best-effort; the in-memory selection remains active.
  }
}
