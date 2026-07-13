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
  host_source: 'Checking model source',
  source_registration: 'Preparing pasted source',
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

export type SuppliedInputDetail = { index: number; parameterName: string; shape: number[]; dtype: string; generator: string; estimatedBytes: number }

export function suppliedInputDetails(failure: TraceFailure): SuppliedInputDetail[] {
  const inputs = failure.error.details?.inputs
  if (!Array.isArray(inputs)) return []
  return inputs.flatMap((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const item = value as Record<string, unknown>
    if (typeof item.index !== 'number' || typeof item.parameter_name !== 'string' || !Array.isArray(item.shape) || !item.shape.every((dimension) => typeof dimension === 'number')) return []
    return [{
      index: item.index,
      parameterName: item.parameter_name,
      shape: item.shape,
      dtype: typeof item.dtype === 'string' ? item.dtype : 'unknown',
      generator: typeof item.generator === 'string' ? item.generator : 'unknown',
      estimatedBytes: typeof item.estimated_bytes === 'number' ? item.estimated_bytes : 0,
    }]
  })
}
