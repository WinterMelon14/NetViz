import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const frontendRoot = fileURLToPath(new URL('../', import.meta.url))
const vite = await createServer({ root: frontendRoot, appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })

const tracePayload = {
  model_name: 'SettingsModel',
  graph: {
    nodes: [
      { id: 'input', label: 'input' },
      { id: 'linear', label: 'Linear' },
    ],
    edges: [
      { source: 'input', target: 'linear', source_output: 0, target_input: 0 },
    ],
  },
}

try {
  const {
    defaultGraphSettings,
    normalizeGraphSettings,
  } = await vite.ssrLoadModule('/src/settings/graphSettings.ts')
  const { layoutOptionsFor } = await vite.ssrLoadModule('/src/graph/buildLayout.ts')
  const { layoutStorageKey } = await vite.ssrLoadModule('/src/graph/layoutStorage.ts')

  assert.deepEqual(normalizeGraphSettings(null), defaultGraphSettings)
  assert.deepEqual(normalizeGraphSettings({ layeringStrategy: 'INVALID' }), defaultGraphSettings)
  assert.deepEqual(
    normalizeGraphSettings({ layeringStrategy: 'NETWORK_SIMPLEX' }),
    { layeringStrategy: 'NETWORK_SIMPLEX' },
  )

  assert.equal(
    layoutOptionsFor({ layeringStrategy: 'LONGEST_PATH_SOURCE' })['elk.layered.layering.strategy'],
    'LONGEST_PATH_SOURCE',
  )

  assert.notEqual(
    layoutStorageKey(tracePayload, { layeringStrategy: 'DF_MODEL_ORDER' }),
    layoutStorageKey(tracePayload, { layeringStrategy: 'NETWORK_SIMPLEX' }),
  )

  console.log('Graph settings validation, layout options, and storage key checks passed.')
} finally {
  await vite.close()
}
