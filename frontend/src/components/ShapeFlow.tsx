import { ShapePill } from './ShapePill'

export function ShapeFlow({ input, output }: { input?: number[]; output?: number[] }) {
  return (
    <span className="shape-flow">
      <ShapePill shape={input} />
      <span className="shape-arrow">⟶</span>
      <ShapePill shape={output} />
    </span>
  )
}
