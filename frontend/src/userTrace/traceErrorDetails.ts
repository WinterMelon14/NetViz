import type { TraceWorkerError } from '../desktop/desktopTraceApi.ts'

export type TraceFailure = {
  runId: string | null
  error: TraceWorkerError
}

const FRIENDLY_STAGES: Record<string, string> = {
  module_import: 'Loading Python module',
  model_resolution: 'Finding model class',
  model_construction: 'Constructing model',
  input_construction: 'Creating model inputs',
  trace_execution: 'Executing and tracing model',
  forward_trace: 'Executing and tracing model',
  worker_execution: 'Executing trace worker',
  worker_transport: 'Returning trace result',
  host_transport: 'Loading trace result',
  source_identity: 'Checking model source',
  host_selection: 'Checking selected file',
  worker_timeout: 'Executing and tracing model',
  worker_cancelled: 'Cancelling trace',
}

const SENSITIVE_DETAIL_KEYS = ['path', 'stderr', 'traceback']

function redactDetails(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactDetails)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    SENSITIVE_DETAIL_KEYS.some((sensitive) => key.toLowerCase().includes(sensitive)) ? '[redacted]' : redactDetails(item),
  ]))
}

export function friendlyTraceStage(stage: string) {
  return FRIENDLY_STAGES[stage] ?? 'Running model trace'
}

export function technicalErrorDetails(failure: TraceFailure, includeTraceback: boolean) {
  return {
    code: failure.error.code,
    stage: failure.error.stage,
    runId: failure.runId,
    details: redactDetails(failure.error.details ?? {}),
    ...(includeTraceback && failure.error.traceback ? { traceback: failure.error.traceback } : {}),
  }
}

