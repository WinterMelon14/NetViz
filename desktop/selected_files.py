"""Compatibility import for callers that still use the original file-specific name."""

from desktop.source_handles import SourceHandles
from desktop.user_trace_constants import MAX_SOURCE_FILE_BYTES

SelectedPythonFiles = SourceHandles

__all__ = ["MAX_SOURCE_FILE_BYTES", "SelectedPythonFiles"]
