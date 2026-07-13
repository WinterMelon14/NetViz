import type { TensorInputDraft } from './inputDrafts.ts'

export type ErrorInputSuggestion = { message: string; draftId?: string; dimensionIndex?: number; value?: string }

export function suggestFromTraceError(message: string, drafts: TensorInputDraft[]): ErrorInputSuggestion | null {
  const first = drafts[0]
  const matrix = message.match(/mat1 and mat2 shapes cannot be multiplied \([^x]+x(\d+) and (\d+)x/)
  if (matrix && first) return { message: `${first.parameterName}'s final dimension may need to be ${matrix[2]}.`, draftId: first.id, dimensionIndex: first.dimensions.length - 1, value: matrix[2] }
  const channels = message.match(/expected input.*to have (\d+) channels/i)
  if (channels && first) return { message: `${first.parameterName}'s channel dimension may need to be ${channels[1]}.`, draftId: first.id, dimensionIndex: 1, value: channels[1] }
  const rank = message.match(/Expected (\d+)-dimensional input.*got (\d+)-dimensional input/i)
  if (rank && first) return { message: `${first.parameterName} may need rank ${rank[1]}.` }
  if (/Long|Int/.test(message) && /dtype|scalar type|embedding/i.test(message)) return { message: 'This operation may require an integer tensor dtype, which is not supported by the current input editor.' }
  return null
}

