import { parseInspectModelSourceResult } from './sourceInspectionApi.ts'
import type { InspectModelSourceResult, SourceInspectionError } from './sourceInspectionApi.ts'

export type SelectedPythonFile = {
  selectionId: string
  fileName: string
  sizeBytes: number
}

export type SelectPythonFileResult =
  | { ok: true; selected: SelectedPythonFile | null }
  | { ok: false; error: SourceInspectionError }

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function bridgeError(value: unknown, fallback: string): SourceInspectionError {
  const error = isRecord(value) && isRecord(value.error) ? value.error : null
  return {
    code: error && typeof error.code === 'string' ? error.code : 'file_selection_failed',
    title: error && typeof error.title === 'string' ? error.title : 'Python file selection failed',
    message: error && typeof error.message === 'string' ? error.message : fallback,
    stage: error && typeof error.stage === 'string' ? error.stage : 'file_selection',
  }
}

function hasSelectedFileApi() {
  return Boolean(
    window.pywebview?.api?.selectPythonFile
    && window.pywebview.api.inspectSelectedPythonFile,
  )
}

async function waitForSelectedFileApi() {
  if (hasSelectedFileApi()) return
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      window.removeEventListener('pywebviewready', ready)
      reject(new Error('Desktop file selection is unavailable. Run the app through the desktop host.'))
    }, 3000)
    function ready() {
      window.clearTimeout(timeout)
      if (hasSelectedFileApi()) resolve()
      else reject(new Error('Desktop file selection API is unavailable.'))
    }
    window.addEventListener('pywebviewready', ready, { once: true })
  })
}

export async function selectPythonFile(): Promise<SelectPythonFileResult> {
  try {
    await waitForSelectedFileApi()
    const result = await window.pywebview?.api?.selectPythonFile?.()
    if (!isRecord(result) || result.ok !== true) {
      return { ok: false, error: bridgeError(result, 'The Python file picker failed.') }
    }
    if (result.selected === null) return { ok: true, selected: null }
    if (!isRecord(result.selected)) {
      return { ok: false, error: bridgeError(null, 'The desktop host returned an invalid file descriptor.') }
    }
    const { selectionId, fileName, sizeBytes } = result.selected
    if (typeof selectionId !== 'string' || typeof fileName !== 'string' || typeof sizeBytes !== 'number') {
      return { ok: false, error: bridgeError(null, 'The desktop host returned an invalid file descriptor.') }
    }
    return { ok: true, selected: { selectionId, fileName, sizeBytes } }
  } catch (error) {
    return { ok: false, error: bridgeError(null, error instanceof Error ? error.message : 'File selection failed.') }
  }
}

export async function inspectSelectedPythonFile(selectionId: string): Promise<InspectModelSourceResult> {
  try {
    await waitForSelectedFileApi()
    const result = await window.pywebview?.api?.inspectSelectedPythonFile?.(selectionId)
    return parseInspectModelSourceResult(result)
  } catch (error) {
    return {
      ok: false,
      error: bridgeError(null, error instanceof Error ? error.message : 'Selected file inspection failed.'),
    }
  }
}
