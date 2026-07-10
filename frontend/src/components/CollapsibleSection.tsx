import type { ReactNode } from 'react'

export function CollapsibleSection({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="collapse-section" open={defaultOpen}>
      <summary>
        <h3>{title}</h3>
      </summary>
      <div className="collapse-body">{children}</div>
    </details>
  )
}
