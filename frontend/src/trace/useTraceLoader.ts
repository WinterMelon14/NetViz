import { useCallback, useEffect, useRef, useState } from 'react'
import { loadStoredPositions, saveStoredPositions } from '../graph/layoutStorage'
import type { LayoutPositions } from '../graph/layoutStorage'
import type { TracePayload } from './types'
import { validateTracePayload } from './validateTracePayload'

const LAYOUT_PERSIST_DEBOUNCE_MS = 300

export function useTraceLoader({ onTraceApplied }: { onTraceApplied: () => void }) {
  const [trace, setTrace] = useState<TracePayload | null>(null)
  const [layoutPositions, setLayoutPositions] = useState<LayoutPositions>({})
  const loadRequestRef = useRef(0)

  const commitTracePayload = useCallback((value: unknown) => {
    const payload = validateTracePayload(value)
    setTrace(payload)
    setLayoutPositions(loadStoredPositions(payload))
    onTraceApplied()
  }, [onTraceApplied])

  const applyTracePayload = useCallback((value: unknown) => {
    loadRequestRef.current += 1
    commitTracePayload(value)
  }, [commitTracePayload])

  useEffect(() => {
    if (!trace) return
    const timer = window.setTimeout(
      () => saveStoredPositions(trace, layoutPositions),
      LAYOUT_PERSIST_DEBOUNCE_MS,
    )
    return () => window.clearTimeout(timer)
  }, [layoutPositions, trace])

  return {
    trace,
    layoutPositions,
    setLayoutPositions,
    loadTracePayload: applyTracePayload,
  }
}
