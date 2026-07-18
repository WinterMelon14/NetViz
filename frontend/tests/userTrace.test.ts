import assert from 'node:assert/strict'
import test from 'node:test'

import { validateTensorDimensions } from '../src/userTrace/inputConfig.ts'
import { buildConstructorConfig, initialConstructorFields } from '../src/userTrace/constructorConfig.ts'
import type { FunctionParameter } from '../src/desktop/sourceInspectionApi.ts'
import { getTraceViewState } from '../src/app/traceViewState.ts'
import { friendlyTraceStage, technicalErrorDetails } from '../src/userTrace/traceErrorDetails.ts'
import { createInputDrafts, validateInputDrafts } from '../src/userTrace/inputDrafts.ts'
import { MAX_TOTAL_INPUT_BYTES } from '../src/userTrace/constants.ts'
import { applyErrorInputSuggestion, suggestFromTraceError } from '../src/userTrace/errorInputSuggestions.ts'
import { parsePythonSource, parseRegisterInlineSourceResponse } from '../src/desktop/sourceApi.ts'

test('application starts empty and distinguishes graph preparation', () => {
  assert.equal(getTraceViewState({ hasTrace: false, hasLayout: false, isLayoutPending: false, hasRecoveryError: false }), 'empty')
  assert.equal(getTraceViewState({ hasTrace: true, hasLayout: false, isLayoutPending: true, hasRecoveryError: false }), 'layout')
  assert.equal(getTraceViewState({ hasTrace: true, hasLayout: true, isLayoutPending: false, hasRecoveryError: false }), 'ready')
  assert.equal(getTraceViewState({ hasTrace: false, hasLayout: false, isLayoutPending: false, hasRecoveryError: true }), 'recovery')
})

test('trace failures use friendly stages and redact copied sensitive details', () => {
  assert.equal(friendlyTraceStage('module_import'), 'Loading Python module')
  assert.equal(friendlyTraceStage('forward_trace'), 'Executing and tracing model')
  assert.equal(friendlyTraceStage('unknown_stage'), 'Running model trace')

  const details = technicalErrorDetails({
    runId: 'run-1',
    error: {
      code: 'module_import_failed',
      title: 'Import failed',
      message: 'Could not import model.',
      stage: 'module_import',
      details: { file_path: 'C:/secret/model.py', exit_code: 1, nested: { stderr: 'secret' } },
      traceback: 'development traceback',
    },
  }, false)

  assert.deepEqual(details, {
    code: 'module_import_failed',
    stage: 'module_import',
    runId: 'run-1',
    details: { file_path: '[redacted]', exit_code: 1, nested: { stderr: '[redacted]' } },
  })
})

test('user tensor dimensions are structured and bounded', () => {
  const valid = validateTensorDimensions(['1', '3', '224', '224'])
  assert.equal(valid.ok, true)
  assert.deepEqual(valid.ok ? valid.shape : null, [1, 3, 224, 224])

  for (const dimensions of [['0'], ['-1'], ['1.5'], [''], ['16777217']]) {
    assert.equal(validateTensorDimensions(dimensions).ok, false, dimensions.join(','))
  }
  assert.equal(validateTensorDimensions(Array(9).fill('1')).ok, false)
})

test('forward signatures create ordered required input drafts', () => {
  const forward = {
    parameters: [
      { name: 'image', position: 'positional_only' as const, required: true },
      { name: 'metadata', position: 'positional_or_keyword' as const, required: true },
      { name: 'optional', position: 'positional_or_keyword' as const, required: false },
    ],
    hasVarArgs: false,
    hasVarKwargs: false,
    lineNumber: 1,
  }
  const result = createInputDrafts(forward)
  assert.equal(result.ok, true)
  assert.deepEqual(result.ok ? result.drafts.map((draft) => draft.parameterName) : [], ['image', 'metadata'])
  assert.equal(createInputDrafts({ ...forward, parameters: [] }).ok, true)
  assert.equal(createInputDrafts({ ...forward, hasVarArgs: true }).ok, false)
  assert.equal(createInputDrafts({ ...forward, parameters: [{ name: 'flag', position: 'keyword_only', required: true }] }).ok, false)
})

test('input drafts validate independently and enforce combined memory', () => {
  const created = createInputDrafts({
    parameters: [
      { name: 'left', position: 'positional_or_keyword', required: true },
      { name: 'right', position: 'positional_or_keyword', required: true },
    ],
    hasVarArgs: false,
    hasVarKwargs: false,
    lineNumber: 1,
  })
  assert.equal(created.ok, true)
  if (!created.ok) return
  created.drafts[0].dimensions = ['10000000']
  created.drafts[1].dimensions = ['10000000']
  assert.equal(validateInputDrafts(created.drafts, MAX_TOTAL_INPUT_BYTES).ok, false)
  created.drafts[0].dimensions = ['1', '2']
  created.drafts[1].dimensions = ['1', '3']
  const valid = validateInputDrafts(created.drafts, MAX_TOTAL_INPUT_BYTES)
  assert.equal(valid.ok, true)
  if (valid.ok) assert.equal(valid.totalBytes, 20)
})

test('known runtime errors produce advisory input suggestions', () => {
  const created = createInputDrafts({ parameters: [{ name: 'x', position: 'positional_or_keyword', required: true }], hasVarArgs: false, hasVarKwargs: false, lineNumber: 1 })
  assert.equal(created.ok, true)
  if (!created.ok) return
  const failure = {
    runId: 'run-1',
    error: {
      code: 'trace_execution_failed',
      title: 'Model trace failed',
      message: 'mat1 and mat2 shapes cannot be multiplied (1x4 and 8x2)',
      stage: 'forward_trace',
      details: { inputs: [{ index: 0, parameter_name: 'x', shape: [1, 4], dtype: 'float32', generator: 'random_normal', estimated_bytes: 16 }] },
      traceback: null,
    },
  }
  created.drafts[0].dimensions = ['1', '4']
  const suggestion = suggestFromTraceError(failure, created.drafts)
  assert.equal(suggestion?.value, '8')
  assert.deepEqual(suggestion ? applyErrorInputSuggestion(created.drafts, suggestion)[0].dimensions : null, ['1', '8'])
  assert.equal(suggestFromTraceError({ ...failure, error: { ...failure.error, message: 'unknown failure' } }, created.drafts), null)
})

test('embedding errors propose an explicit dtype-only edit', () => {
  const created = createInputDrafts({ parameters: [{ name: 'tokens', position: 'positional_or_keyword', required: true }], hasVarArgs: false, hasVarKwargs: false, lineNumber: 1 })
  assert.equal(created.ok, true)
  if (!created.ok) return
  const failure = {
    runId: 'run-2',
    error: {
      code: 'trace_execution_failed',
      title: 'Model trace failed',
      message: 'Expected tensor for argument indices to have scalar type Long but got Float for embedding',
      stage: 'forward_trace',
      details: { inputs: [{ index: 0, parameter_name: 'tokens', shape: [1, 128], dtype: 'float32', generator: 'random_normal', estimated_bytes: 512 }] },
      traceback: null,
    },
  }
  const suggestion = suggestFromTraceError(failure, created.drafts)
  const updated = suggestion ? applyErrorInputSuggestion(created.drafts, suggestion) : created.drafts
  assert.equal(updated[0].dtype, 'int64')
  assert.equal(updated[0].generator, 'random_integer')
  assert.deepEqual(updated[0].dimensions, created.drafts[0].dimensions)
})

test('channel, rank, and LayerNorm recovery target submitted inputs', () => {
  const created = createInputDrafts({
    parameters: [
      { name: 'metadata', position: 'positional_or_keyword', required: true },
      { name: 'image', position: 'positional_or_keyword', required: true },
    ],
    hasVarArgs: false,
    hasVarKwargs: false,
    lineNumber: 1,
  })
  assert.equal(created.ok, true)
  if (!created.ok) return
  created.drafts[0].dimensions = ['1', '16']
  created.drafts[1].dimensions = ['1', '1', '32', '32']
  const baseFailure = {
    runId: 'run-3',
    error: {
      code: 'trace_execution_failed',
      title: 'Model trace failed',
      stage: 'forward_trace',
      details: {
        inputs: [
          { index: 0, parameter_name: 'metadata', shape: [1, 16], dtype: 'float32', generator: 'random_normal', estimated_bytes: 64 },
          { index: 1, parameter_name: 'image', shape: [1, 1, 32, 32], dtype: 'float32', generator: 'random_normal', estimated_bytes: 4096 },
        ],
      },
      traceback: null,
    },
  }
  const channel = suggestFromTraceError({
    ...baseFailure,
    error: { ...baseFailure.error, message: 'Given groups=1, weight of size [8, 3, 3, 3], expected input[1, 1, 32, 32] to have 3 channels, but got 1 channels instead' },
  }, created.drafts)
  assert.equal(channel?.draftId, created.drafts[1].id)
  const channelApplied = channel ? applyErrorInputSuggestion(created.drafts, channel) : created.drafts
  assert.deepEqual(channelApplied[0].dimensions, ['1', '16'])
  assert.deepEqual(channelApplied[1].dimensions, ['1', '3', '32', '32'])

  const rank = suggestFromTraceError({
    ...baseFailure,
    error: { ...baseFailure.error, message: 'Expected 4-dimensional input for 4-dimensional weight, but got 2-dimensional input' },
  }, created.drafts)
  assert.equal(rank?.draftId, created.drafts[0].id)
  assert.equal(rank?.field, 'shape')

  const normalized = suggestFromTraceError({
    ...baseFailure,
    error: { ...baseFailure.error, message: 'Given normalized_shape=[8], expected input with shape [*, 8], but got input of size[1, 16]' },
  }, created.drafts)
  assert.equal(normalized?.draftId, created.drafts[0].id)
  assert.deepEqual(normalized?.value, ['1', '8'])
})

test('static suggestions populate practical defaults and integer bounds', () => {
  const created = createInputDrafts({
    parameters: [{ name: 'tokens', position: 'positional_or_keyword', required: true }],
    hasVarArgs: false,
    hasVarKwargs: false,
    lineNumber: 1,
    inputSuggestions: [{
      parameterName: 'tokens',
      shapeTemplate: [1, null],
      dimensionSources: ['default', 'unknown'],
      dtypeCategory: 'integer',
      confidence: 'high',
      evidence: ['self.embedding resolves to Embedding.'],
      presetKind: 'sequence',
      integerRange: { min: 0, maxExclusive: 100 },
    }],
  })
  assert.equal(created.ok, true)
  if (!created.ok) return
  assert.deepEqual(created.drafts[0].dimensions, ['1', '128'])
  assert.equal(created.drafts[0].dtype, 'int64')
  assert.equal(created.drafts[0].integerMaxExclusive, 100)
  const validation = validateInputDrafts(created.drafts, MAX_TOTAL_INPUT_BYTES)
  assert.equal(validation.ok, true)
  if (validation.ok) assert.equal(validation.totalBytes, 1024)
})

test('image suggestions populate the default spatial dimensions', () => {
  const created = createInputDrafts({
    parameters: [{ name: 'image', position: 'positional_or_keyword', required: true }],
    hasVarArgs: false,
    hasVarKwargs: false,
    lineNumber: 1,
    inputSuggestions: [{
      parameterName: 'image',
      shapeTemplate: [1, 3, null, null],
      dimensionSources: ['default', 'inferred', 'unknown', 'unknown'],
      confidence: 'high',
      evidence: ['self.conv resolves to Conv2d.'],
      presetKind: 'image',
    }],
  })
  assert.equal(created.ok, true)
  if (!created.ok) return
  assert.deepEqual(created.drafts[0].dimensions, ['1', '3', '224', '224'])
})

test('constructor parameters become positional and keyword JSON literals', () => {
  const parameters: FunctionParameter[] = [
    { name: 'width', position: 'positional_only', required: true },
    { name: 'config', position: 'positional_or_keyword', required: true },
    { name: 'enabled', position: 'keyword_only', required: false, defaultValue: true },
  ]
  const fields = initialConstructorFields(parameters)
  fields.width.text = '4'
  fields.config.text = '{"layers":[2,3],"name":"demo"}'
  const result = buildConstructorConfig(parameters, fields)

  assert.equal(result.ok, true)
  if (result.ok) {
    assert.deepEqual(result.args, [4, { layers: [2, 3], name: 'demo' }])
    assert.deepEqual(result.kwargs, { enabled: true })
  }
})

test('constructor values after an omitted optional positional parameter use keywords', () => {
  const parameters: FunctionParameter[] = [
    { name: 'width', position: 'positional_or_keyword', required: false, defaultValue: 4 },
    { name: 'label', position: 'positional_or_keyword', required: false, defaultValue: 'default' },
  ]
  const fields = initialConstructorFields(parameters)
  fields.width.enabled = false
  fields.label.enabled = true
  fields.label.text = '"custom"'

  const result = buildConstructorConfig(parameters, fields)

  assert.deepEqual(result, { ok: true, args: [], kwargs: { label: 'custom' } })
})

test('constructor defaults are populated when safe and source expressions remain omitted', () => {
  const parameters: FunctionParameter[] = [
    { name: 'd_model', position: 'positional_or_keyword', required: true, typeText: 'int' },
    { name: 'expansion', position: 'positional_or_keyword', required: false, defaultValue: 2, typeText: 'int' },
    { name: 'activation', position: 'positional_or_keyword', required: false, defaultDisplay: 'torch.nn.ReLU()' },
  ]
  const fields = initialConstructorFields(parameters)

  assert.deepEqual(fields.expansion, { enabled: true, text: '2' })
  assert.deepEqual(fields.activation, { enabled: false, text: 'torch.nn.ReLU()' })
  fields.d_model.text = '4'

  assert.deepEqual(buildConstructorConfig(parameters, fields), { ok: true, args: [4, 2], kwargs: {} })
})

test('constructor configuration rejects expressions and excessive depth', () => {
  const parameters: FunctionParameter[] = [
    { name: 'value', position: 'positional_or_keyword', required: true },
  ]
  const fields = initialConstructorFields(parameters)
  fields.value.text = 'torch.device("cpu")'
  assert.equal(buildConstructorConfig(parameters, fields).ok, false)

  fields.value.text = '[[[[[[[[[null]]]]]]]]]'
  assert.equal(buildConstructorConfig(parameters, fields).ok, false)
})

test('source bridge descriptors reject malformed responses', () => {
  assert.deepEqual(parsePythonSource({ sourceId: 'source-1', kind: 'inline', displayName: 'pasted_model.py', sizeBytes: 12 }), {
    sourceId: 'source-1', kind: 'inline', displayName: 'pasted_model.py', sizeBytes: 12,
  })
  assert.equal(parsePythonSource({ sourceId: 'source-1', kind: 'inline', displayName: 'pasted_model.py', sizeBytes: -1 }), null)
  assert.equal(parseRegisterInlineSourceResponse({ ok: true, source: { sourceId: 42 } }).ok, false)
  assert.equal(parseRegisterInlineSourceResponse({ ok: false, error: { code: 'source_too_large', message: 'Too large.' } }).ok, false)
})
