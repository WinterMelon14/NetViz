import type { ForwardSignature } from '../desktop/sourceInspectionApi.ts'
import {
  DEFAULT_IMAGE_SPATIAL_SIZE,
  DEFAULT_INTEGER_MAX_EXCLUSIVE,
  DEFAULT_SEQUENCE_LENGTH,
  MAX_USER_INPUTS,
} from './constants.ts'
import { validateTensorDimensions, type TensorInputValidation } from './inputConfig.ts'

export type TensorInputDraft = {
  id: string
  parameterName: string
  dimensions: string[]
  dimensionSources: Array<'inferred' | 'default' | 'chosen' | 'unknown'>
  dtype: 'float32' | 'int64'
  generator: 'random_normal' | 'random_integer'
  integerMaxExclusive?: number
}

export type InputDraftResult =
  | { ok: true; drafts: TensorInputDraft[] }
  | { ok: false; message: string }

export function createInputDrafts(forward: ForwardSignature | null): InputDraftResult {
  if (!forward) return { ok: false, message: 'No directly declared forward method was found.' }
  if (forward.hasVarArgs || forward.hasVarKwargs) return { ok: false, message: 'Variadic forward signatures are not supported yet.' }
  if (forward.parameters.some((parameter) => parameter.required && parameter.position === 'keyword_only')) {
    return { ok: false, message: 'Required keyword-only forward parameters are not supported yet.' }
  }
  const parameters = forward.parameters.filter((parameter) => parameter.required && parameter.position !== 'keyword_only')
  if (parameters.length > MAX_USER_INPUTS) return { ok: false, message: `Models may have at most ${MAX_USER_INPUTS} required tensor inputs.` }
  return {
    ok: true,
    drafts: parameters.map((parameter, index) => {
      const suggestion = forward.inputSuggestions?.find((item) => item.parameterName === parameter.name)
      const suggestedDimensions = suggestion?.shapeTemplate.map((dimension, dimensionIndex) => {
        if (dimension !== null) return String(dimension)
        if (suggestion.presetKind === 'image' && dimensionIndex >= suggestion.shapeTemplate.length - 2) {
          return String(DEFAULT_IMAGE_SPATIAL_SIZE)
        }
        if (suggestion.presetKind === 'sequence') return String(DEFAULT_SEQUENCE_LENGTH)
        return ''
      })
      return {
        id: `${parameter.name}-${index}`,
        parameterName: parameter.name,
        dimensions: suggestedDimensions ?? ['1', '1'],
        dimensionSources: suggestion?.dimensionSources.map((source, dimensionIndex) => (
          source === 'unknown' && suggestedDimensions?.[dimensionIndex] ? 'default' : source
        )) ?? ['default', 'default'],
        dtype: suggestion?.dtypeCategory === 'integer' ? 'int64' : 'float32',
        generator: suggestion?.dtypeCategory === 'integer' ? 'random_integer' : 'random_normal',
        integerMaxExclusive: suggestion?.dtypeCategory === 'integer' ? suggestion.integerRange?.maxExclusive ?? DEFAULT_INTEGER_MAX_EXCLUSIVE : undefined,
      }
    }),
  }
}

export type InputDraftValidation =
  | { ok: true; inputs: Array<{ draft: TensorInputDraft; validation: Extract<TensorInputValidation, { ok: true }> }>; totalBytes: number }
  | { ok: false; message: string; validations: Map<string, TensorInputValidation> }

export function validateInputDrafts(drafts: TensorInputDraft[], maxTotalBytes: number): InputDraftValidation {
  const validations = new Map<string, TensorInputValidation>()
  let totalBytes = 0
  for (const draft of drafts) {
    if (draft.dtype === 'int64' && (!Number.isSafeInteger(draft.integerMaxExclusive) || (draft.integerMaxExclusive ?? 0) < 1)) {
      return { ok: false, message: draft.parameterName + ': integer values need an upper bound of at least 1.', validations }
    }
    const validation = validateTensorDimensions(draft.dimensions, draft.dtype)
    validations.set(draft.id, validation)
    if (!validation.ok) return { ok: false, message: `${draft.parameterName}: ${validation.message}`, validations }
    if (totalBytes > maxTotalBytes - validation.sizeBytes) {
      return { ok: false, message: 'Combined inputs exceed the 64 MiB memory limit.', validations }
    }
    totalBytes += validation.sizeBytes
  }
  return { ok: true, inputs: drafts.map((draft) => ({ draft, validation: validations.get(draft.id) as Extract<TensorInputValidation, { ok: true }> })), totalBytes }
}
