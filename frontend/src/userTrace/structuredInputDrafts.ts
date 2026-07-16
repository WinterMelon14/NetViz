import type { StructuredInputSpec } from '../desktop/userTraceRequest.ts'
import type { ForwardSignature, FunctionParameter, InputSuggestion, SerializableLiteral } from '../desktop/sourceInspectionApi.ts'
import type { TensorInputDraft } from './inputDrafts.ts'
import { validateTensorDimensions } from './inputConfig.ts'
import {
  DEFAULT_IMAGE_SPATIAL_SIZE,
  DEFAULT_INTEGER_MAX_EXCLUSIVE,
  DEFAULT_SEQUENCE_LENGTH,
  MAX_INPUT_SERIALIZED_BYTES,
  MAX_INPUT_STRING_CHARS,
  MAX_STRUCTURED_CONTAINER_ITEMS,
  MAX_STRUCTURED_INPUT_DEPTH,
  MAX_STRUCTURED_INPUT_VALUES,
  MAX_TOTAL_INPUT_BYTES,
  MAX_USER_INPUTS,
} from './constants.ts'

export type StructuredValueKind = 'tensor' | 'none' | 'boolean' | 'integer' | 'float' | 'string' | 'list' | 'tuple' | 'dict'
export type StructuredValueDraft =
  | { id: string; kind: 'tensor'; tensor: TensorInputDraft }
  | { id: string; kind: 'none' }
  | { id: string; kind: 'boolean'; value: boolean }
  | { id: string; kind: 'integer'; value: string }
  | { id: string; kind: 'float'; value: string }
  | { id: string; kind: 'string'; value: string }
  | { id: string; kind: 'list'; items: StructuredValueDraft[] }
  | { id: string; kind: 'tuple'; items: StructuredValueDraft[] }
  | { id: string; kind: 'dict'; entries: Array<{ id: string; key: string; value: StructuredValueDraft }> }

export type ParameterInputDraft = {
  id: string
  parameterName: string
  position: FunctionParameter['position']
  required: boolean
  included: boolean
  placement: 'positional' | 'keyword'
  annotationText?: string
  value: StructuredValueDraft
}

export type StructuredDraftResult =
  | { ok: true; drafts: ParameterInputDraft[]; variadicNotice: string | null }
  | { ok: false; message: string }

let nextDraftId = 0

function draftId(prefix: string) {
  nextDraftId += 1
  return `${prefix}-${nextDraftId}`
}

export function createStructuredInputDrafts(forward: ForwardSignature | null): StructuredDraftResult {
  if (!forward) return { ok: false, message: 'No directly declared forward method was found.' }
  if (forward.parameters.length > MAX_USER_INPUTS) return { ok: false, message: `Models may have at most ${MAX_USER_INPUTS} configurable forward parameters.` }
  const drafts = forward.parameters.map((parameter, index): ParameterInputDraft => {
    const suggestion = forward.inputSuggestions?.find((item) => item.parameterName === parameter.name)
    return {
      id: `${parameter.name}-${index}`,
      parameterName: parameter.name,
      position: parameter.position,
      required: parameter.required,
      included: parameter.required,
      placement: parameter.position === 'keyword_only' || !parameter.required ? 'keyword' : 'positional',
      annotationText: parameter.annotationText,
      value: initialValue(parameter, suggestion),
    }
  })
  const variadic = [forward.hasVarArgs ? `*${forward.varArgName ?? 'args'}` : '', forward.hasVarKwargs ? `**${forward.varKwargName ?? 'kwargs'}` : ''].filter(Boolean)
  return { ok: true, drafts, variadicNotice: variadic.length ? `${variadic.join(' and ')} are not populated automatically.` : null }
}

function initialValue(parameter: FunctionParameter, suggestion?: InputSuggestion): StructuredValueDraft {
  if (suggestion) return tensorValue(parameter.name, suggestion)
  const annotation = (parameter.annotationText ?? '').replace(/\s/g, '').toLowerCase()
  if (annotationContains(annotation, 'bool')) return { id: draftId(parameter.name), kind: 'boolean', value: literalBoolean(parameter.defaultValue) }
  if (annotationContains(annotation, 'int')) return { id: draftId(parameter.name), kind: 'integer', value: literalText(parameter.defaultValue, '0') }
  if (annotationContains(annotation, 'float')) return { id: draftId(parameter.name), kind: 'float', value: literalText(parameter.defaultValue, '1') }
  if (annotationContains(annotation, 'str')) return { id: draftId(parameter.name), kind: 'string', value: typeof parameter.defaultValue === 'string' ? parameter.defaultValue : '' }
  if (annotationContains(annotation, 'tuple')) return { id: draftId(parameter.name), kind: 'tuple', items: [] }
  if (annotationContains(annotation, 'list')) return { id: draftId(parameter.name), kind: 'list', items: [] }
  if (annotationContains(annotation, 'dict')) return { id: draftId(parameter.name), kind: 'dict', entries: [] }
  if (!parameter.required && parameter.defaultKind === 'none') return { id: draftId(parameter.name), kind: 'none' }
  if (!parameter.required && parameter.defaultValue !== undefined) {
    const value = valueFromLiteral(parameter.name, parameter.defaultValue)
    return parameter.defaultKind === 'tuple' && value.kind === 'list' ? { ...value, kind: 'tuple' } : value
  }
  return tensorValue(parameter.name)
}

function annotationContains(annotation: string, typeName: string) {
  return new RegExp(`(^|[.\\[|,])${typeName}($|[\\]|,])`).test(annotation)
}

function valueFromLiteral(name: string, value: SerializableLiteral): StructuredValueDraft {
  const id = draftId(name)
  if (value === null) return { id, kind: 'none' }
  if (typeof value === 'boolean') return { id, kind: 'boolean', value }
  if (typeof value === 'number') return { id, kind: Number.isInteger(value) ? 'integer' : 'float', value: String(value) }
  if (typeof value === 'string') return { id, kind: 'string', value }
  if (Array.isArray(value)) return { id, kind: 'list', items: value.map((item) => valueFromLiteral(name, item)) }
  return { id, kind: 'dict', entries: Object.entries(value).map(([key, item]) => ({ id: draftId(`${name}-entry`), key, value: valueFromLiteral(name, item) })) }
}

function tensorValue(parameterName: string, suggestion?: InputSuggestion): StructuredValueDraft {
  const suggestedDimensions = suggestion?.shapeTemplate.map((dimension, index) => {
    if (dimension !== null) return String(dimension)
    if (suggestion.presetKind === 'image' && index >= suggestion.shapeTemplate.length - 2) return String(DEFAULT_IMAGE_SPATIAL_SIZE)
    if (suggestion.presetKind === 'sequence') return String(DEFAULT_SEQUENCE_LENGTH)
    return ''
  })
  const id = draftId(parameterName)
  const integer = suggestion?.dtypeCategory === 'integer'
  return {
    id,
    kind: 'tensor',
    tensor: {
      id,
      parameterName,
      dimensions: suggestedDimensions ?? ['1', '1'],
      dimensionSources: suggestion?.dimensionSources.map((source, index) => source === 'unknown' && suggestedDimensions?.[index] ? 'default' : source) ?? ['default', 'default'],
      dtype: integer ? 'int64' : 'float32',
      generator: integer ? 'random_integer' : 'random_normal',
      integerMaxExclusive: integer ? suggestion?.integerRange?.maxExclusive ?? DEFAULT_INTEGER_MAX_EXCLUSIVE : undefined,
    },
  }
}

export function createValueDraft(kind: StructuredValueKind, parameterName: string): StructuredValueDraft {
  const id = draftId(parameterName)
  if (kind === 'tensor') return tensorValue(parameterName)
  if (kind === 'none') return { id, kind }
  if (kind === 'boolean') return { id, kind, value: false }
  if (kind === 'integer') return { id, kind, value: '0' }
  if (kind === 'float') return { id, kind, value: '1' }
  if (kind === 'string') return { id, kind, value: '' }
  if (kind === 'dict') return { id, kind, entries: [] }
  return { id, kind, items: [] }
}

function literalText(value: SerializableLiteral | undefined, fallback: string) {
  return typeof value === 'number' ? String(value) : fallback
}

function literalBoolean(value: SerializableLiteral | undefined) {
  return typeof value === 'boolean' ? value : false
}

type Budget = { values: number; tensors: number; totalBytes: number }
export type StructuredInputValidation =
  | { ok: true; args: StructuredInputSpec[]; kwargs: Record<string, StructuredInputSpec>; totalBytes: number; valueCount: number; tensorCount: number }
  | { ok: false; message: string; path: string }

export function validateStructuredInputDrafts(drafts: ParameterInputDraft[]): StructuredInputValidation {
  const budget: Budget = { values: 0, tensors: 0, totalBytes: 0 }
  const args: StructuredInputSpec[] = []
  const kwargs: Record<string, StructuredInputSpec> = {}
  let positionalClosed = false
  try {
    for (const draft of drafts) {
      if (!draft.included) {
        if (draft.position !== 'keyword_only') positionalClosed = true
        continue
      }
      const spec = validateValue(draft.value, draft.parameterName, 1, budget)
      if (draft.placement === 'positional') {
        if (draft.position === 'keyword_only') throw validationError(draft.parameterName, 'Keyword-only parameters cannot be positional.')
        if (positionalClosed) throw validationError(draft.parameterName, 'Use keyword placement because an earlier positional parameter is omitted or keyword-based.')
        args.push(spec)
      } else {
        if (draft.position === 'positional_only') throw validationError(draft.parameterName, 'Positional-only parameters cannot use keyword placement.')
        positionalClosed = true
        kwargs[draft.parameterName] = spec
      }
    }
    const serializedBytes = new TextEncoder().encode(JSON.stringify({ args, kwargs })).byteLength
    if (serializedBytes > MAX_INPUT_SERIALIZED_BYTES) throw validationError('inputs', `Structured inputs exceed ${MAX_INPUT_SERIALIZED_BYTES.toLocaleString()} serialized bytes.`)
    return { ok: true, args, kwargs, totalBytes: budget.totalBytes, valueCount: budget.values, tensorCount: budget.tensors }
  } catch (error) {
    const result = error as { path?: string; message?: string }
    return { ok: false, path: result.path ?? 'inputs', message: result.message ?? 'Structured inputs are invalid.' }
  }
}

function validateValue(value: StructuredValueDraft, path: string, depth: number, budget: Budget): StructuredInputSpec {
  if (depth > MAX_STRUCTURED_INPUT_DEPTH) throw validationError(path, `Maximum nesting depth is ${MAX_STRUCTURED_INPUT_DEPTH}.`)
  budget.values += 1
  if (budget.values > MAX_STRUCTURED_INPUT_VALUES) throw validationError(path, `Maximum value count is ${MAX_STRUCTURED_INPUT_VALUES}.`)
  if (value.kind === 'tensor') {
    const tensor = value.tensor
    const validation = validateTensorDimensions(tensor.dimensions, tensor.dtype)
    if (!validation.ok) throw validationError(path, validation.message)
    if (tensor.dtype === 'int64' && (!Number.isSafeInteger(tensor.integerMaxExclusive) || (tensor.integerMaxExclusive ?? 0) < 1)) throw validationError(path, 'Integer tensors need an upper bound of at least 1.')
    budget.tensors += 1
    budget.totalBytes += validation.sizeBytes
    if (budget.tensors > MAX_USER_INPUTS) throw validationError(path, `Maximum tensor count is ${MAX_USER_INPUTS}.`)
    if (budget.totalBytes > MAX_TOTAL_INPUT_BYTES) throw validationError(path, 'Combined tensor allocation exceeds 64 MiB.')
    return { kind: 'tensor', shape: validation.shape, dtype: tensor.dtype, generator: tensor.generator, ...(tensor.dtype === 'int64' ? { integer_max_exclusive: tensor.integerMaxExclusive } : {}) }
  }
  if (value.kind === 'none') return { kind: 'none' }
  if (value.kind === 'boolean') return { kind: 'boolean', value: value.value }
  if (value.kind === 'integer') {
    const number = Number(value.value)
    if (!Number.isSafeInteger(number)) throw validationError(path, 'Enter a safe integer.')
    return { kind: 'integer', value: number }
  }
  if (value.kind === 'float') {
    const number = Number(value.value)
    if (!Number.isFinite(number)) throw validationError(path, 'Enter a finite number.')
    return { kind: 'float', value: number }
  }
  if (value.kind === 'string') {
    if (value.value.length > MAX_INPUT_STRING_CHARS) throw validationError(path, `Strings are limited to ${MAX_INPUT_STRING_CHARS.toLocaleString()} characters.`)
    return { kind: 'string', value: value.value }
  }
  if (value.kind === 'list' || value.kind === 'tuple') {
    if (value.items.length > MAX_STRUCTURED_CONTAINER_ITEMS) throw validationError(path, `Containers are limited to ${MAX_STRUCTURED_CONTAINER_ITEMS} items.`)
    return { kind: value.kind, items: value.items.map((item, index) => validateValue(item, `${path}[${index}]`, depth + 1, budget)) }
  }
  if (value.entries.length > MAX_STRUCTURED_CONTAINER_ITEMS) throw validationError(path, `Dictionaries are limited to ${MAX_STRUCTURED_CONTAINER_ITEMS} entries.`)
  const keys = new Set<string>()
  const entries = value.entries.map((entry, index) => {
    const entryPath = `${path}.${entry.key || `[${index}]`}`
    if (entry.key.length > MAX_INPUT_STRING_CHARS) throw validationError(entryPath, 'Dictionary key is too long.')
    if (keys.has(entry.key)) throw validationError(entryPath, 'Dictionary keys must be unique.')
    keys.add(entry.key)
    return { key: entry.key, value: validateValue(entry.value, entryPath, depth + 1, budget) }
  })
  return { kind: 'dict', entries }
}

function validationError(path: string, message: string) {
  return { path, message }
}

export function topLevelTensorDrafts(drafts: ParameterInputDraft[]) {
  return drafts.flatMap((draft) => draft.included && draft.value.kind === 'tensor' ? [draft.value.tensor] : [])
}

export function replaceTopLevelTensorDrafts(drafts: ParameterInputDraft[], tensors: TensorInputDraft[]) {
  const byId = new Map(tensors.map((tensor) => [tensor.id, tensor]))
  return drafts.map((draft) => draft.value.kind === 'tensor' && byId.has(draft.value.tensor.id)
    ? { ...draft, value: { ...draft.value, tensor: byId.get(draft.value.tensor.id)! } }
    : draft)
}
