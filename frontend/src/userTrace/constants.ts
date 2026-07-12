// Bounds representative tensor inputs before a trace request reaches Python.
export const MAX_TENSOR_DIMENSIONS = 8
export const MAX_TENSOR_ELEMENTS = 16_777_216
export const MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
export const FLOAT32_BYTES = 4

// Bounds JSON-safe constructor values consistently with worker validation.
export const MAX_CONSTRUCTOR_LITERAL_DEPTH = 8
export const MAX_CONSTRUCTOR_LITERAL_VALUES = 1_024
export const MAX_CONSTRUCTOR_STRING_CHARS = 64 * 1024
export const MAX_CONSTRUCTOR_SERIALIZED_BYTES = 256 * 1024

// Namespaces remembered trust decisions by exact inspected source content.
export const TRUSTED_SOURCE_STORAGE_PREFIX = 'tensor-trace:trusted-source:'

