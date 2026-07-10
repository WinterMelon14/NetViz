import { isShapeString } from '../trace/format'
import { ShapePill } from './ShapePill'

export function ValueDisplay({ value }: { value: unknown }) {
  if (Array.isArray(value) && value.every((item) => typeof item === 'number')) {
    return <ShapePill shape={value} />
  }

  if (typeof value === 'string' && isShapeString(value)) {
    if (value === 'scalar') return <ShapePill />
    return <ShapePill shape={value.slice(1, -1).split(',').map((part) => Number(part.trim()))} />
  }

  return <>{String(value ?? 'n/a')}</>
}
