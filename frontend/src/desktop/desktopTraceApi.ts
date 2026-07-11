import type { TracePayload } from '../trace/types'

const protocolVersion = 1

type TraceTransfer =
  | { transfer: 'inline'; payload: TracePayload }
  | { transfer: 'file'; path: string }

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
        runKnownModelTrace?: () => Promise<unknown>
      }
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function isTracePayload(value: unknown): value is TracePayload {
  return isRecord(value)
    && isRecord(value.graph)
    && Array.isArray(value.graph.nodes)
    && Array.isArray(value.graph.edges)
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

function parseRunTraceResponse(value: unknown): RunTraceResponse {
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
    if (transfer === 'inline' && isTracePayload(payload)) {
      return {
        protocol_version: protocolVersion,
        type: 'success',
        run_id: typeof value.run_id === 'string' ? value.run_id : '',
        trace: { transfer, payload },
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

function waitForPywebviewReady() {
  if (window.pywebview?.api?.runKnownModelTrace) return Promise.resolve()

  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      window.removeEventListener('pywebviewready', onReady)
      reject(new Error('pywebview API is not available. Run the app through the desktop host.'))
    }, 3000)

    function onReady() {
      window.clearTimeout(timeout)
      resolve()
    }

    window.addEventListener('pywebviewready', onReady, { once: true })
  })
}

export async function runKnownModelTrace() {
  try {
    await waitForPywebviewReady()
    const result = await window.pywebview?.api?.runKnownModelTrace?.()
    return parseRunTraceResponse(result)
  } catch (error) {
    return {
      protocol_version: protocolVersion,
      type: 'error',
      run_id: null,
      error: {
        code: 'desktop_bridge_unavailable',
        title: 'Desktop bridge unavailable',
        message: error instanceof Error ? error.message : 'The desktop bridge could not be reached.',
        stage: 'desktop_bridge',
      },
    } satisfies RunTraceResponse
  }
}
