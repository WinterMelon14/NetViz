import { useCallback, useEffect, useState } from 'react'
import { loadStoredPositions, saveStoredPositions } from '../graph/layoutStorage'
import type { LayoutPositions } from '../graph/layoutStorage'
import { parseTraceJson } from './parseTraceJson'
import type { TracePayload } from './types'
import { validateTracePayload } from './validateTracePayload'

export function useTraceLoader({ onTraceApplied }: { onTraceApplied: () => void }) {
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [jsonText, setJsonText] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [layoutPositions, setLayoutPositions] = useState<LayoutPositions>({})

  const applyTracePayload = useCallback((value: unknown) => {
    const payload = validateTracePayload(value)
    setTrace(payload)
    setLayoutPositions(loadStoredPositions(payload))
    setError(null)
    setLoadError(null)
    onTraceApplied()
  }, [onTraceApplied])

  useEffect(() => {
    fetch('/branchy.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load trace JSON (${response.status})`)
        return response.text()
      })
      .then((text) => {
        applyTracePayload(parseTraceJson(text))
      })
      .catch((loadError: Error) => setError(loadError.message))
  }, [applyTracePayload])

  useEffect(() => {
    if (!trace) return
    saveStoredPositions(trace, layoutPositions)
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
    onJsonTextChange,
    loadJsonFromFile,
    loadJsonFromText,
  }
}
