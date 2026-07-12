export type TraceViewState = 'empty' | 'layout' | 'ready' | 'recovery'

export function getTraceViewState({
  hasTrace,
  hasLayout,
  isLayoutPending,
  hasRecoveryError,
}: {
  hasTrace: boolean
  hasLayout: boolean
  isLayoutPending: boolean
  hasRecoveryError: boolean
}): TraceViewState {
  if (hasRecoveryError) return 'recovery'
  if (!hasTrace) return 'empty'
  if (isLayoutPending || !hasLayout) return 'layout'
  return 'ready'
}

