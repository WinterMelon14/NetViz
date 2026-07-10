import { InfoRow } from '../components/InfoRow'
import { ShapePill } from '../components/ShapePill'
import type { ParamsInfo } from '../trace/types'

export function ParamsDetail({ params }: { params?: ParamsInfo }) {
  const shapes = Object.entries(params?.shapes ?? {})

  return (
    <section className="detail-block">
      {shapes.length ? shapes.map(([name, shape]) => <InfoRow key={name} label={name} value={<ShapePill shape={shape} />} />) : <p className="empty-note">No parameter tensors</p>}
      <InfoRow label="count" value={(params?.count ?? 0).toLocaleString()} />
      <InfoRow label="memory" value={params?.memory?.human ?? '0 B'} />
    </section>
  )
}
