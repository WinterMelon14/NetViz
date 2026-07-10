import { formatShape, shapeParts } from '../trace/format'

export function ShapePill({ shape }: { shape?: number[] }) {
  return (
    <span className="shape-pill" aria-label={formatShape(shape)}>
      {shapeParts(shape).map((part, index) => (
        <span className="shape-dim" key={`${part}-${index}`}>
          {part}
        </span>
      ))}
    </span>
  )
}
