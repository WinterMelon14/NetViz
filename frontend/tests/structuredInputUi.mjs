import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const frontendRoot = fileURLToPath(new URL('../', import.meta.url))
const fixturePath = fileURLToPath(new URL('../../tests/fixtures/structured_input_call_v2.json', import.meta.url))
const fixture = JSON.parse(await readFile(fixturePath, 'utf8'))
const vite = await createServer({ root: frontendRoot, appType: 'custom', logLevel: 'silent', server: { middlewareMode: true } })

try {
  const draftsModule = await vite.ssrLoadModule('/src/userTrace/structuredInputDrafts.ts')
  const { StructuredInputEditor } = await vite.ssrLoadModule('/src/userTrace/StructuredInputEditor.tsx')
  const tensor = {
    id: 'mask-tensor',
    kind: 'tensor',
    tensor: {
      id: 'mask-tensor', parameterName: 'mask', dimensions: ['2', '2'], dimensionSources: ['chosen', 'chosen'],
      dtype: 'int64', generator: 'random_integer', integerMaxExclusive: 2,
    },
  }
  const drafts = [
    {
      id: 'pair', parameterName: 'pair', position: 'positional_only', required: true, included: true, placement: 'positional',
      value: { id: 'pair-value', kind: 'tuple', items: [{ id: 'three', kind: 'integer', value: '3' }, { id: 'none', kind: 'none' }] },
    },
    { id: 'mask', parameterName: 'mask', position: 'keyword_only', required: true, included: true, placement: 'keyword', value: tensor },
    {
      id: 'settings', parameterName: 'settings', position: 'positional_or_keyword', required: false, included: true, placement: 'keyword',
      value: { id: 'settings-value', kind: 'dict', entries: [
        { id: 'enabled', key: 'enabled', value: { id: 'enabled-value', kind: 'boolean', value: true } },
        { id: 'name', key: 'name', value: { id: 'name-value', kind: 'string', value: 'demo' } },
      ] },
    },
    { id: 'optional', parameterName: 'optional', position: 'keyword_only', required: false, included: false, placement: 'keyword', value: { id: 'optional-none', kind: 'none' } },
  ]
  const validation = draftsModule.validateStructuredInputDrafts(drafts)
  assert.equal(validation.ok, true)
  assert.deepEqual({ input_schema_version: 2, args: validation.args, kwargs: validation.kwargs }, fixture)
  assert.equal(validation.tensorCount, 1)
  assert.equal(validation.totalBytes, 32)
  assert.equal('optional' in validation.kwargs, false)

  const explicitNone = drafts.map((draft) => draft.id === 'optional' ? { ...draft, included: true } : draft)
  const explicitNoneValidation = draftsModule.validateStructuredInputDrafts(explicitNone)
  assert.equal(explicitNoneValidation.ok, true)
  assert.deepEqual(explicitNoneValidation.kwargs.optional, { kind: 'none' })

  const markup = renderToStaticMarkup(React.createElement(StructuredInputEditor, { draft: drafts[3], disabled: false, onChange() {} }))
  assert.match(markup, /Omitted/)
  assert.match(markup, /Provide value/)
  assert.match(markup, /keyword/)
  console.log('Structured input fixture, omission, keyword, nesting, allocation, and UI checks passed.')
} finally {
  await vite.close()
}
