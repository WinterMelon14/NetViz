import type { TracePayload } from '../trace/types.ts'
import { validateTracePayload } from '../trace/validateTracePayload.ts'
import type { UserTraceBridgeRequest } from './userTraceRequest.ts'

const protocolVersion = 1

export type TraceRunState = 'idle' | 'starting' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out'

type TraceTransfer =
  | { transfer: 'inline'; payload: TracePayload }
  | { transfer: 'file'; path: string; size_bytes: number }

export type TraceWorkerError = {
  code: string
  title: string
  message: string
  stage: string
  details?: Record<string, unknown>
  traceback?: string | null
}

export type RunTraceResponse =
  | {
      protocol_version: 1
      type: 'success'
      run_id: string
      trace: TraceTransfer
      warnings: string[]
    }
  | {
      protocol_version: 1
      type: 'error'
      run_id: string | null
      error: TraceWorkerError
    }

declare global {
  interface Window {
    pywebview?: {
      api?: {
        runUserTrace?: (request: UserTraceBridgeRequest) => Promise<unknown>
        runSelectedUserTrace?: (request: UserTraceBridgeRequest) => Promise<unknown>
        selectPythonFile?: () => Promise<unknown>
        inspectSelectedPythonFile?: (selectionId: string) => Promise<unknown>
        registerInlinePythonSource?: (request: { sourceText: string; displayName?: string }) => Promise<unknown>
        inspectPythonSource?: (sourceId: string) => Promise<unknown>
        releasePythonSource?: (sourceId: string) => Promise<unknown>
        cancelTrace?: (runId: string) => Promise<unknown>
        consumeTraceFile?: (runId: string, path: string) => Promise<unknown>
      }
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeError(value: unknown): TraceWorkerError {
  if (!isRecord(value)) {
    return {
      code: 'desktop_bridge_error',
      title: 'Desktop trace failed',
      message: 'The desktop trace bridge returned an unknown error.',
      stage: 'desktop_bridge',
    }
  }

  return {
    code: typeof value.code === 'string' ? value.code : 'desktop_bridge_error',
    title: typeof value.title === 'string' ? value.title : 'Desktop trace failed',
    message: typeof value.message === 'string' ? value.message : 'The desktop trace bridge returned an error.',
    stage: typeof value.stage === 'string' ? value.stage : 'desktop_bridge',
    details: isRecord(value.details) ? value.details : undefined,
    traceback: typeof value.traceback === 'string' || value.traceback === null ? value.traceback : undefined,
  }
}

export function parseRunTraceResponse(value: unknown): RunTraceResponse {
  if (!isRecord(value) || value.protocol_version !== protocolVersion) {
    return {
      protocol_version: protocolVersion,
      type: 'error',
      run_id: null,
      error: {
        code: 'desktop_bridge_protocol_error',
        title: 'Unexpected desktop trace response',
        message: 'The desktop bridge returned a response that does not match the trace protocol.',
        stage: 'desktop_bridge',
      },
    }
  }

  if (value.type === 'success' && isRecord(value.trace)) {
    const transfer = value.trace.transfer
    const payload = value.trace.payload
    if (transfer === 'inline') {
      let validatedPayload: TracePayload
      try {
        validatedPayload = validateTracePayload(payload)
      } catch (error) {
        return {
          protocol_version: protocolVersion,
          type: 'error',
          run_id: typeof value.run_id === 'string' ? value.run_id : null,
          error: {
            code: 'desktop_bridge_protocol_error',
            title: 'Invalid trace payload',
            message: error instanceof Error ? error.message : 'The desktop bridge returned an invalid trace payload.',
            stage: 'desktop_bridge',
          },
        }
      }
      return {
        protocol_version: protocolVersion,
        type: 'success',
        run_id: typeof value.run_id === 'string' ? value.run_id : '',
        trace: { transfer, payload: validatedPayload },
        warnings: Array.isArray(value.warnings) ? value.warnings.filter((warning): warning is string => typeof warning === 'string') : [],
      }
    }
    if (
      transfer === 'file'
      && typeof value.trace.path === 'string'
      && value.trace.path.length > 0
      && typeof value.trace.size_bytes === 'number'
      && Number.isFinite(value.trace.size_bytes)
      && value.trace.size_bytes >= 0
    ) {
      return {
        protocol_version: protocolVersion,
        type: 'success',
        run_id: typeof value.run_id === 'string' ? value.run_id : '',
        trace: { transfer, path: value.trace.path, size_bytes: value.trace.size_bytes },
        warnings: Array.isArray(value.warnings) ? value.warnings.filter((warning): warning is string => typeof warning === 'string') : [],
      }
    }
  }

  if (value.type === 'error') {
    return {
      protocol_version: protocolVersion,
      type: 'error',
      run_id: typeof value.run_id === 'string' ? value.run_id : null,
      error: normalizeError(value.error),
    }
  }

  return {
    protocol_version: protocolVersion,
    type: 'error',
    run_id: null,
    error: {
      code: 'desktop_bridge_protocol_error',
      title: 'Unsupported desktop trace response',
      message: 'The desktop bridge did not return an inline successful trace or structured error.',
      stage: 'desktop_bridge',
    },
  }
}

function hasTraceApi() {
  return Boolean(
    window.pywebview?.api?.runUserTrace
    && window.pywebview.api.cancelTrace
    && window.pywebview.api.consumeTraceFile,
  )
}

function waitForPywebviewReady() {
  if (hasTraceApi()) return Promise.resolve()

  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      window.removeEventListener('pywebviewready', onReady)
      reject(new Error('pywebview API is not available. Run the app through the desktop host.'))
    }, 3000)

    function onReady() {
      window.clearTimeout(timeout)
      if (hasTraceApi()) {
        resolve()
      } else {
        reject(new Error('pywebview trace API is not available. Run the app through the desktop host.'))
      }
    }

    window.addEventListener('pywebviewready', onReady, { once: true })
  })
}

export function createTraceRunId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID()
  }

  return `trace-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export async function runUserTrace(request: UserTraceBridgeRequest) {
  try {
    await waitForPywebviewReady()
    const result = await window.pywebview?.api?.runUserTrace?.(request)
    return parseRunTraceResponse(result)
  } catch (error) {
    return {
      protocol_version: protocolVersion,
      type: 'error',
      run_id: request.run_id,
      error: {
        code: 'desktop_bridge_unavailable',
        title: 'Desktop bridge unavailable',
        message: error instanceof Error ? error.message : 'The desktop bridge could not be reached.',
        stage: 'desktop_bridge',
      },
    } satisfies RunTraceResponse
  }
}

export async function cancelTrace(runId: string) {
  try {
    await waitForPywebviewReady()
    const result = await window.pywebview?.api?.cancelTrace?.(runId)
    return parseRunTraceResponse(result)
  } catch (error) {
    return {
      protocol_version: protocolVersion,
      type: 'error',
      run_id: runId,
      error: {
        code: 'desktop_bridge_unavailable',
        title: 'Desktop bridge unavailable',
        message: error instanceof Error ? error.message : 'The desktop bridge could not be reached.',
        stage: 'desktop_bridge',
      },
    } satisfies RunTraceResponse
  }
}

export async function consumeTraceFile(runId: string, path: string): Promise<TracePayload> {
  await waitForPywebviewReady()
  const result = await window.pywebview?.api?.consumeTraceFile?.(runId, path)
  if (!isRecord(result) || result.ok !== true) {
    const error = isRecord(result) && result.type === 'error' ? normalizeError(result.error) : null
    throw new Error(error?.message ?? 'The desktop trace file could not be loaded.')
  }
  return validateTracePayload(result.payload)
}
