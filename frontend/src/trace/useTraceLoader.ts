import { useCallback, useEffect, useRef, useState } from 'react'
import { loadStoredPositions, saveStoredPositions } from '../graph/layoutStorage'
import type { LayoutPositions } from '../graph/layoutStorage'
import { parseTraceJson } from './parseTraceJson'
import type { TracePayload } from './types'
import { validateTracePayload } from './validateTracePayload'

const LAYOUT_PERSIST_DEBOUNCE_MS = 300

export function useTraceLoader({ onTraceApplied }: { onTraceApplied: () => void }) {
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [jsonText, setJsonText] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [layoutPositions, setLayoutPositions] = useState<LayoutPositions>({})
  const loadRequestRef = useRef(0)

  const commitTracePayload = useCallback((value: unknown) => {
    const payload = validateTracePayload(value)
    setTrace(payload)
    setLayoutPositions(loadStoredPositions(payload))
    setError(null)
    setLoadError(null)
    onTraceApplied()
  }, [onTraceApplied])

  const applyTracePayload = useCallback((value: unknown) => {
    loadRequestRef.current += 1
    commitTracePayload(value)
  }, [commitTracePayload])

  const loadDefaultTrace = useCallback(() => {
    const requestId = ++loadRequestRef.current
    fetch('/branchy.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load trace JSON (${response.status})`)
        return response.text()
      })
      .then((text) => {
        if (loadRequestRef.current !== requestId) return
        commitTracePayload(parseTraceJson(text))
      })
      .catch((loadError: unknown) => {
        if (loadRequestRef.current !== requestId) return
        setError(loadError instanceof Error ? loadError.message : 'Unable to load the default trace.')
      })
  }, [commitTracePayload])

  useEffect(() => {
    loadDefaultTrace()
  }, [loadDefaultTrace])

  useEffect(() => {
    if (!trace) return
    const timer = window.setTimeout(
      () => saveStoredPositions(trace, layoutPositions),
      LAYOUT_PERSIST_DEBOUNCE_MS,
    )
    return () => window.clearTimeout(timer)
  }, [layoutPositions, trace])

  function onJsonTextChange(text: string) {
    setJsonText(text)
    setLoadError(null)
  }

  function loadJsonFromFile(file: File) {
    file
      .text()
      .then((text) => {
        setJsonText(text)
        applyTracePayload(parseTraceJson(text))
        setIsLoadModalOpen(false)
      })
      .catch((fileError: Error) => setLoadError(fileError.message))
  }

  function loadJsonFromText() {
    try {
      applyTracePayload(parseTraceJson(jsonText))
      setIsLoadModalOpen(false)
    } catch (textError) {
      setLoadError(textError instanceof Error ? textError.message : 'Could not parse JSON.')
    }
  }

  return {
    trace,
    error,
    isLoadModalOpen,
    setIsLoadModalOpen,
    jsonText,
    loadError,
    layoutPositions,
    setLayoutPositions,
    loadTracePayload: applyTracePayload,
    retryDefaultTrace: loadDefaultTrace,
    onJsonTextChange,
    loadJsonFromFile,
    loadJsonFromText,
  }
}
