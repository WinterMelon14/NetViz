import {
  FLOAT32_BYTES,
  INT64_BYTES,
  MAX_TENSOR_DIMENSIONS,
  MAX_TENSOR_ELEMENTS,
  MAX_TOTAL_INPUT_BYTES,
} from '../desktop/userTraceRequest.ts'

export type TensorInputValidation =
  | { ok: true; shape: number[]; elementCount: number; sizeBytes: number }
  | { ok: false; message: string }

export function validateTensorDimensions(dimensions: string[], dtype: 'float32' | 'int64' = 'float32'): TensorInputValidation {
  if (dimensions.length < 1 || dimensions.length > MAX_TENSOR_DIMENSIONS) {
    return { ok: false, message: `Shape must contain 1 to ${MAX_TENSOR_DIMENSIONS} dimensions.` }
  }
  const shape: number[] = []
  for (let index = 0; index < dimensions.length; index += 1) {
    const text = dimensions[index].trim()
    if (!/^\d+$/.test(text)) return { ok: false, message: `Dimension ${index + 1} must be a whole number.` }
    const value = Number(text)
    if (!Number.isSafeInteger(value) || value < 1) {
      return { ok: false, message: `Dimension ${index + 1} must be at least 1.` }
    }
    shape.push(value)
  }
  const elementCount = shape.reduce((total, dimension) => total * dimension, 1)
  const sizeBytes = elementCount * (dtype === 'int64' ? INT64_BYTES : FLOAT32_BYTES)
  if (!Number.isSafeInteger(elementCount) || elementCount > MAX_TENSOR_ELEMENTS) {
    return { ok: false, message: `Input exceeds the ${MAX_TENSOR_ELEMENTS.toLocaleString()} element limit.` }
  }
  if (sizeBytes > MAX_TOTAL_INPUT_BYTES) {
    return { ok: false, message: 'Input exceeds the 64 MiB memory limit.' }
  }
  return { ok: true, shape, elementCount, sizeBytes }
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KiB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`
}
