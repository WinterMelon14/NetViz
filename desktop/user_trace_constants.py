"""Tunable safety and lifecycle limits for local user-model tracing."""

# Bounds representative tensor allocation before PyTorch is invoked.
MAX_USER_INPUTS = 1
MAX_TENSOR_DIMENSIONS = 8
MAX_TENSOR_ELEMENTS = 16_777_216
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
FLOAT32_BYTES = 4

# Bounds JSON-safe values passed to a model constructor.
MAX_CONSTRUCTOR_LITERAL_DEPTH = 8
MAX_CONSTRUCTOR_LITERAL_VALUES = 1_024
MAX_CONSTRUCTOR_STRING_CHARS = 64 * 1024
MAX_CONSTRUCTOR_SERIALIZED_BYTES = 256 * 1024

# Bounds selected source inspection before reading the file into memory.
MAX_SOURCE_FILE_BYTES = 200_000

# Bounds graceful worker termination before a forced kill.
CANCEL_TERMINATION_TIMEOUT_SECONDS = 2
MAX_REMEMBERED_CANCELLED_RUNS = 128
