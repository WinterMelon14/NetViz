import { useCallback, useEffect, useRef, useState } from 'react'
import { loadStoredPositions, saveStoredPositions } from '../graph/layoutStorage'
import type { LayoutPositions } from '../graph/layoutStorage'
import type { GraphSettings } from '../settings/graphSettings'
import type { TracePayload } from './types'
import { validateTracePayload } from './validateTracePayload'

const LAYOUT_PERSIST_DEBOUNCE_MS = 300

export function useTraceLoader({
  onTraceApplied,
  settings,
}: {
  onTraceApplied: () => void
  settings: GraphSettings
}) {
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [layoutPositions, setLayoutPositions] = useState<LayoutPositions>({})
  const loadRequestRef = useRef(0)

  const commitTracePayload = useCallback((value: unknown) => {
    const payload = validateTracePayload(value)
    setTrace(payload)
    setLayoutPositions(loadStoredPositions(payload, settings))
    onTraceApplied()
  }, [onTraceApplied, settings])

  const applyTracePayload = useCallback((value: unknown) => {
    loadRequestRef.current += 1
    commitTracePayload(value)
  }, [commitTracePayload])

  useEffect(() => {
    if (!trace) return
    const timer = window.setTimeout(
      () => saveStoredPositions(trace, settings, layoutPositions),
      LAYOUT_PERSIST_DEBOUNCE_MS,
    )
    return () => window.clearTimeout(timer)
  }, [layoutPositions, settings, trace])

  return {
    trace,
    layoutPositions,
    setLayoutPositions,
    loadTracePayload: applyTracePayload,
  }
}
