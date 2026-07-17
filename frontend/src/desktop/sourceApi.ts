import { parseInspectModelSourceResult } from './sourceInspectionApi.ts'
import type { InspectModelSourceResult, SourceInspectionError } from './sourceInspectionApi.ts'

export type PythonSource = {
  sourceId: string
  kind: 'file' | 'inline'
  displayName: string
  sizeBytes: number
  projectRootDisplay?: string
}

export type SelectPythonFileResult =
  | { ok: true; source: PythonSource | null }
  | { ok: false; error: SourceInspectionError }

export type RegisterInlineSourceResult =
  | { ok: true; source: PythonSource }
  | { ok: false; error: SourceInspectionError }

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function bridgeError(value: unknown, fallback: string): SourceInspectionError {
  const error = isRecord(value) && isRecord(value.error) ? value.error : null
  return {
    code: error && typeof error.code === 'string' ? error.code : 'source_bridge_failed',
    title: error && typeof error.title === 'string' ? error.title : 'Python source operation failed',
    message: error && typeof error.message === 'string' ? error.message : fallback,
    stage: error && typeof error.stage === 'string' ? error.stage : 'source_bridge',
    details: error && isRecord(error.details) ? error.details : undefined,
  }
}

export function parsePythonSource(value: unknown): PythonSource | null {
  if (!isRecord(value)) return null
  const { sourceId, kind, displayName, sizeBytes } = value
  if (typeof sourceId !== 'string' || (kind !== 'file' && kind !== 'inline') || typeof displayName !== 'string' || typeof sizeBytes !== 'number' || !Number.isFinite(sizeBytes) || sizeBytes < 0) return null
  return { sourceId, kind, displayName, sizeBytes, projectRootDisplay: typeof value.projectRootDisplay === 'string' ? value.projectRootDisplay : undefined }
}

export function parseRegisterInlineSourceResponse(value: unknown): RegisterInlineSourceResult {
  if (!isRecord(value) || value.ok !== true) return { ok: false, error: bridgeError(value, 'Pasted source registration failed.') }
  const source = parsePythonSource(value.source)
  return source ? { ok: true, source } : { ok: false, error: bridgeError(null, 'The desktop host returned an invalid source descriptor.') }
}

function hasSourceApi() {
  return Boolean(
    window.pywebview?.api?.selectPythonFile
    && window.pywebview.api.registerInlinePythonSource
    && window.pywebview.api.inspectPythonSource
    && window.pywebview.api.releasePythonSource,
  )
}

async function waitForSourceApi() {
  if (hasSourceApi()) return
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      window.removeEventListener('pywebviewready', ready)
      reject(new Error('Desktop source APIs are unavailable. Run the app through the desktop host.'))
    }, 3000)
    function ready() {
      window.clearTimeout(timeout)
      if (hasSourceApi()) resolve()
      else reject(new Error('Desktop source APIs are unavailable.'))
    }
    window.addEventListener('pywebviewready', ready, { once: true })
  })
}

export async function selectPythonFile(): Promise<SelectPythonFileResult> {
  try {
    await waitForSourceApi()
    const result = await window.pywebview?.api?.selectPythonFile?.()
    if (!isRecord(result) || result.ok !== true) return { ok: false, error: bridgeError(result, 'The Python file picker failed.') }
    if (result.selected === null) return { ok: true, source: null }
    const source = parsePythonSource(result.selected)
    return source ? { ok: true, source } : { ok: false, error: bridgeError(null, 'The desktop host returned an invalid source descriptor.') }
  } catch (error) {
    return { ok: false, error: bridgeError(null, error instanceof Error ? error.message : 'File selection failed.') }
  }
}

export async function registerInlinePythonSource(sourceText: string): Promise<RegisterInlineSourceResult> {
  try {
    await waitForSourceApi()
    const result = await window.pywebview?.api?.registerInlinePythonSource?.({ sourceText, displayName: 'pasted_model.py' })
    return parseRegisterInlineSourceResponse(result)
  } catch (error) {
    return { ok: false, error: bridgeError(null, error instanceof Error ? error.message : 'Pasted source registration failed.') }
  }
}

export async function inspectPythonSource(sourceId: string): Promise<InspectModelSourceResult> {
  try {
    await waitForSourceApi()
    return parseInspectModelSourceResult(await window.pywebview?.api?.inspectPythonSource?.(sourceId))
  } catch (error) {
    return { ok: false, error: bridgeError(null, error instanceof Error ? error.message : 'Source inspection failed.') }
  }
}

export async function releasePythonSource(sourceId: string): Promise<{ ok: true; released: boolean } | { ok: false; error: SourceInspectionError }> {
  try {
    await waitForSourceApi()
    const result = await window.pywebview?.api?.releasePythonSource?.(sourceId)
    if (isRecord(result) && result.ok === true && typeof result.released === 'boolean') return { ok: true, released: result.released }
    return { ok: false, error: bridgeError(result, 'The Python source could not be released.') }
  } catch (error) {
    return { ok: false, error: bridgeError(null, error instanceof Error ? error.message : 'The Python source could not be released.') }
  }
}
