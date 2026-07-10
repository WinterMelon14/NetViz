import type { ReactNode } from 'react'

export function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value ?? 'n/a'}</strong>
    </div>
  )
}
