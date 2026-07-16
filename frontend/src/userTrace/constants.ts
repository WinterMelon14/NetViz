// Bounds representative tensor inputs before a trace request reaches Python.
export const MAX_TENSOR_DIMENSIONS = 8
export const MAX_TENSOR_ELEMENTS = 16_777_216
export const MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
export const FLOAT32_BYTES = 4
export const INT64_BYTES = 8
export const DEFAULT_INTEGER_MAX_EXCLUSIVE = 10
export const DEFAULT_IMAGE_SPATIAL_SIZE = 224
export const DEFAULT_SEQUENCE_LENGTH = 128
export const MAX_USER_INPUTS = 8
export const MAX_STRUCTURED_INPUT_DEPTH = 8
export const MAX_STRUCTURED_CONTAINER_ITEMS = 128
export const MAX_STRUCTURED_INPUT_VALUES = 1_024
export const MAX_INPUT_STRING_CHARS = 64 * 1024
export const MAX_INPUT_SERIALIZED_BYTES = 256 * 1024

// Bounds JSON-safe constructor values consistently with worker validation.
export const MAX_CONSTRUCTOR_LITERAL_DEPTH = 8
export const MAX_CONSTRUCTOR_LITERAL_VALUES = 1_024
export const MAX_CONSTRUCTOR_STRING_CHARS = 64 * 1024
export const MAX_CONSTRUCTOR_SERIALIZED_BYTES = 256 * 1024

// Mirrors host limits so paste controls can reject oversized source before bridge submission.
export const MAX_SOURCE_CHARS = 200_000
export const MAX_SOURCE_BYTES = 200_000

// Namespaces remembered trust decisions by exact inspected source content.
export const TRUSTED_SOURCE_STORAGE_PREFIX = 'tensor-trace:trusted-source:'
