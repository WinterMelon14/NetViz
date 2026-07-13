"""Tunable safety and lifecycle limits for local user-model tracing."""

# Supported generated tensor configuration.
SUPPORTED_TENSOR_DTYPES = frozenset({"float32"})
SUPPORTED_TENSOR_GENERATORS = frozenset({"random_normal"})

# Bounds representative tensor allocation before PyTorch is invoked.
MAX_USER_INPUTS = 8
MAX_TENSOR_DIMENSIONS = 8
MAX_TENSOR_ELEMENTS = 16_777_216
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
FLOAT32_BYTES = 4

# Bounds JSON-safe values passed to a model constructor.
MAX_CONSTRUCTOR_LITERAL_DEPTH = 8
MAX_CONSTRUCTOR_LITERAL_VALUES = 1_024
MAX_CONSTRUCTOR_STRING_CHARS = 64 * 1024
MAX_CONSTRUCTOR_SERIALIZED_BYTES = 256 * 1024

# Bounds selected and pasted source inspection before parsing.
MAX_SOURCE_FILE_BYTES = 200_000
MAX_SOURCE_CHARS = 200_000

# Bounds worker protocol, diagnostics, and trace-result transport.
MAX_INLINE_TRACE_BYTES = 1 * 1024 * 1024
MAX_TRACE_FILE_BYTES = 256 * 1024 * 1024
MAX_PROTOCOL_OUTPUT_BYTES = MAX_INLINE_TRACE_BYTES + 64 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
TRACE_FILE_TTL_SECONDS = 5 * 60

# Bounds trace execution, graceful termination, and lifecycle bookkeeping.
DEFAULT_TRACE_TIMEOUT_SECONDS = 600
CANCEL_TERMINATION_TIMEOUT_SECONDS = 2
MAX_REMEMBERED_CANCELLED_RUNS = 128
