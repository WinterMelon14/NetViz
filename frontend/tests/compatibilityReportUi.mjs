import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const frontendRoot = fileURLToPath(new URL('../', import.meta.url))
const fixturePath = fileURLToPath(new URL('../../tests/fixtures/compatibility_report_v1.json', import.meta.url))
const vite = await createServer({ root: frontendRoot, appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })

try {
  const { CompatibilityReportPanel } = await vite.ssrLoadModule('/src/userTrace/CompatibilityReportPanel.tsx')
  const { parseCompatibilityReport } = await vite.ssrLoadModule('/src/desktop/sourceInspectionApi.ts')
  const fixture = JSON.parse(await readFile(fixturePath, 'utf8'))
  assert.equal(parseCompatibilityReport(fixture).schemaVersion, 1)
  assert.throws(() => parseCompatibilityReport({ ...fixture, schemaVersion: 2 }), /Unsupported compatibility report version/)

  const statuses = ['supported', 'configuration_required', 'unsupported', 'unknown']
  const report = {
    schemaVersion: 1,
    className: 'MixedModel',
    findings: statuses.map((status, index) => ({
      category: ['runtime', 'constructor', 'forward', 'fx'][index],
      code: `fixture_${status}`,
      status,
      title: `${status} finding`,
      explanation: `${status} explanation`,
      origin: index === 0 ? 'runtime' : index === 3 ? 'heuristic' : 'source',
      evidence: [{ kind: 'source', text: status, lineNumber: index + 1 }],
      remediation: status === 'configuration_required' ? 'Complete the field.' : undefined,
      target: status === 'configuration_required' ? { kind: 'constructor_parameter', name: 'width' } : undefined,
    })),
    tracingOutcome: { status: 'unknown', statement: 'Symbolic tracing outcome is unknown until execution.' },
  }
  const markup = renderToStaticMarkup(React.createElement(CompatibilityReportPanel, {
    report,
    resolvedFindingKeys: new Set(),
  }))
  for (const status of statuses) assert.match(markup, new RegExp(`compatibility-status--${status}`))
  assert.doesNotMatch(markup, /Go to configuration/)
  assert.doesNotMatch(markup, /Runtime constraints/)
  assert.match(markup, /aria-labelledby="compatibility-report-title"/)
  assert.match(markup, /Symbolic tracing outcome is unknown until execution/)
  console.log('Compatibility report parser and mixed-status UI checks passed.')
} finally {
  await vite.close()
}
