import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const frontendRoot = fileURLToPath(new URL('../', import.meta.url))
const vite = await createServer({ root: frontendRoot, appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })

const output = (summary) => ({ index: 0, role: 'output', summary })
const node = (label, outputs) => ({ id: label, kind: 'module', label, fx_op: 'call_module', target: label, inputs: [], outputs })

try {
  const { nodeDiagnostics } = await vite.ssrLoadModule('/src/graph/nodeDiagnostics.ts')
  const { GraphNodeCard } = await vite.ssrLoadModule('/src/graph/GraphNodeCard.tsx')

  assert.deepEqual(nodeDiagnostics(node('Linear', [])), { hasNan: false, hasInf: false, sparsePercent: null })
  assert.deepEqual(nodeDiagnostics(node('Linear', [output({ has_nan: true }), output({ has_inf: true })])), { hasNan: true, hasInf: true, sparsePercent: null })
  assert.equal(nodeDiagnostics(node('Linear', [output({ zeros_pct: 5 })])).sparsePercent, null)
  assert.equal(nodeDiagnostics(node('Linear', [output({ zeros_pct: 5.1 })])).sparsePercent, 5.1)
  assert.equal(nodeDiagnostics(node('ReLU', [output({ zeros_pct: 90 })])).sparsePercent, null)
  assert.equal(nodeDiagnostics(node('ReLU', [output({ zeros_pct: 90.1 })])).sparsePercent, 90.1)
  assert.equal(nodeDiagnostics(node('pad', [output({ zeros_pct: 100 })])).sparsePercent, null)
  assert.equal(nodeDiagnostics(node('Linear', [output({ zeros_pct: 12 }), output({ zeros_pct: 34 })])).sparsePercent, 34)

  const markup = renderToStaticMarkup(React.createElement(GraphNodeCard, {
    node: { ...node('ReLU', [output({ has_nan: true, has_inf: true, zeros_pct: 90.6 })]), x: 0, y: 0, depth: 0 },
    timingSeverity: null,
    isOutputNode: false,
    isSelected: false,
    onPointerDown() {},
    onPointerMove() {},
    onPointerUp() {},
    onSelectNode() {},
  }))
  assert.match(markup, />NaN</)
  assert.match(markup, />Inf</)
  assert.match(markup, />Sparse 91%</)

  console.log('Node NaN, Inf, and strict sparsity threshold checks passed.')
} finally {
  await vite.close()
}
