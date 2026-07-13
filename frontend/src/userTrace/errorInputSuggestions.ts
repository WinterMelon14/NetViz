import type { TensorInputDraft } from './inputDrafts.ts'
import { suppliedInputDetails, type TraceFailure } from './traceErrorDetails.ts'
import { DEFAULT_INTEGER_MAX_EXCLUSIVE } from './constants.ts'

export type ErrorInputSuggestion =
  | { message: string; draftId: string; field: 'dimension'; dimensionIndex: number; value: string }
  | { message: string; draftId: string; field: 'shape'; value: string[] }
  | { message: string; draftId: string; field: 'dtype'; value: 'int64' }

function identifiableDraft(
  failure: TraceFailure,
  drafts: TensorInputDraft[],
  matches: (shape: number[], dtype: string) => boolean,
) {
  const submitted = suppliedInputDetails(failure).filter((input) => matches(input.shape, input.dtype))
  if (submitted.length === 1) {
    return drafts.find((draft) => draft.parameterName === submitted[0].parameterName) ?? null
  }
  const matchingDrafts = drafts.filter((draft) => matches(draft.dimensions.map(Number), draft.dtype))
  if (matchingDrafts.length === 1) return matchingDrafts[0]
  return drafts.length === 1 ? drafts[0] : null
}

function parsedShape(text: string) {
  const values = text.match(/\d+/g)?.map(Number) ?? []
  return values.every((value) => Number.isSafeInteger(value) && value > 0) ? values : []
}

export function suggestFromTraceError(failure: TraceFailure, drafts: TensorInputDraft[]): ErrorInputSuggestion | null {
  const message = failure.error.message
  const matrix = message.match(/mat1 and mat2 shapes cannot be multiplied \([^x]+x(\d+) and (\d+)x/)
  if (matrix) {
    const actual = Number(matrix[1])
    const draft = identifiableDraft(failure, drafts, (shape) => shape.at(-1) === actual)
    if (draft) return { message: draft.parameterName + "'s final dimension may need to be " + matrix[2] + '.', draftId: draft.id, field: 'dimension', dimensionIndex: draft.dimensions.length - 1, value: matrix[2] }
  }

  const channels = message.match(/expected input(?:\[[^\]]+\])?.*to have (\d+) channels, but got (\d+) channels/i)
    ?? message.match(/expected input.*to have (\d+) channels/i)
  if (channels) {
    const actual = channels[2] ? Number(channels[2]) : null
    const draft = identifiableDraft(failure, drafts, (shape) => actual === null ? shape.length >= 2 : shape[1] === actual)
    if (draft) return { message: draft.parameterName + "'s channel dimension may need to be " + channels[1] + '.', draftId: draft.id, field: 'dimension', dimensionIndex: 1, value: channels[1] }
  }

  const rank = message.match(/Expected (\d+)[- ]dimensional input.*got (\d+)[- ]dimensional input/i)
    ?? message.match(/Expected (\d+)D input.*got (\d+)D input/i)
  if (rank) {
    const expected = Number(rank[1])
    const actual = Number(rank[2])
    const draft = identifiableDraft(failure, drafts, (shape) => shape.length === actual)
    if (draft) {
      const value = draft.dimensions.slice(0, expected)
      while (value.length < expected) value.push('1')
      return { message: draft.parameterName + ' may need rank ' + expected + '.', draftId: draft.id, field: 'shape', value }
    }
  }

  if (/Long|Int64|integer/i.test(message) && /dtype|scalar type|embedding|indices/i.test(message)) {
    const draft = identifiableDraft(failure, drafts, (_shape, dtype) => dtype !== 'int64')
    if (draft) return { message: draft.parameterName + ' may need the int64 dtype.', draftId: draft.id, field: 'dtype', value: 'int64' }
  }

  const normalized = message.match(/normalized_shape=\[([^\]]+)\].*got input of size\[([^\]]+)\]/i)
  if (normalized) {
    const trailing = parsedShape(normalized[1])
    const actual = parsedShape(normalized[2])
    const draft = identifiableDraft(failure, drafts, (shape) => shape.join(',') === actual.join(','))
    if (draft && trailing.length && draft.dimensions.length >= trailing.length) {
      const value = [...draft.dimensions]
      value.splice(value.length - trailing.length, trailing.length, ...trailing.map(String))
      return { message: draft.parameterName + "'s trailing dimensions may need to match LayerNorm's normalized shape.", draftId: draft.id, field: 'shape', value }
    }
  }
  return null
}

export function applyErrorInputSuggestion(drafts: TensorInputDraft[], suggestion: ErrorInputSuggestion) {
  return drafts.map((draft) => {
    if (draft.id !== suggestion.draftId) return draft
    if (suggestion.field === 'dimension') {
      return {
        ...draft,
        dimensions: draft.dimensions.map((dimension, index) => index === suggestion.dimensionIndex ? suggestion.value : dimension),
        dimensionSources: draft.dimensionSources.map((source, index) => index === suggestion.dimensionIndex ? 'chosen' as const : source),
      }
    }
    if (suggestion.field === 'shape') {
      return { ...draft, dimensions: suggestion.value, dimensionSources: suggestion.value.map(() => 'chosen' as const) }
    }
    return {
      ...draft,
      dtype: 'int64' as const,
      generator: 'random_integer' as const,
      integerMaxExclusive: draft.integerMaxExclusive ?? DEFAULT_INTEGER_MAX_EXCLUSIVE,
    }
  })
}
