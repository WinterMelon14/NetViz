import type { ForwardSignature } from '../desktop/sourceInspectionApi.ts'
import { MAX_USER_INPUTS } from './constants.ts'
import { validateTensorDimensions, type TensorInputValidation } from './inputConfig.ts'

export type TensorInputDraft = {
  id: string
  parameterName: string
  dimensions: string[]
  dtype: 'float32'
  generator: 'random_normal'
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
    drafts: parameters.map((parameter, index) => ({
      id: `${parameter.name}-${index}`,
      parameterName: parameter.name,
      dimensions: ['1', '1'],
      dtype: 'float32',
      generator: 'random_normal',
    })),
  }
}

export type InputDraftValidation =
  | { ok: true; inputs: Array<{ draft: TensorInputDraft; validation: Extract<TensorInputValidation, { ok: true }> }>; totalBytes: number }
  | { ok: false; message: string; validations: Map<string, TensorInputValidation> }

export function validateInputDrafts(drafts: TensorInputDraft[], maxTotalBytes: number): InputDraftValidation {
  const validations = new Map<string, TensorInputValidation>()
  let totalBytes = 0
  for (const draft of drafts) {
    const validation = validateTensorDimensions(draft.dimensions)
    validations.set(draft.id, validation)
    if (!validation.ok) return { ok: false, message: `${draft.parameterName}: ${validation.message}`, validations }
    if (totalBytes > maxTotalBytes - validation.sizeBytes) {
      return { ok: false, message: 'Combined inputs exceed the 64 MiB memory limit.', validations }
    }
    totalBytes += validation.sizeBytes
  }
  return { ok: true, inputs: drafts.map((draft) => ({ draft, validation: validations.get(draft.id) as Extract<TensorInputValidation, { ok: true }> })), totalBytes }
}

