import { InfoRow } from '../components/InfoRow'
import { RichTextView } from '../components/RichTextView'
import { ValueDisplay } from '../components/ValueDisplay'
import { explainNode } from '../explanations'
import { formatShape, formatUnknown } from '../trace/format'
import { primaryInput, primaryOutput } from '../trace/selectors'
import type { TraceNode } from '../trace/types'

function classifyShapeStep(from: unknown, to: unknown) {
  const fromText = formatUnknown(from)
  const toText = formatUnknown(to)

  if (fromText === toText) return 'preserved'
  if (fromText === '-' || fromText === 'n/a' || fromText === 'undefined') return 'created'
  if (toText === '-' || toText === 'n/a' || toText === 'undefined') return 'reduced'
  return 'changed'
}

export function TransformationDetail({ node }: { node: TraceNode }) {
  const explanation = explainNode(node)
  const inputShape = formatShape(primaryInput(node)?.shape)
  const outputShape = formatShape(primaryOutput(node)?.shape)

  if (!explanation) {
    return <p className="empty-note">No transformation metadata available.</p>
  }

  return (
    <section className="transformation-detail">
      <p className="transformation-short">
        <RichTextView value={explanation.short} />
      </p>
      <p>
        <RichTextView value={explanation.description} />
      </p>
      <InfoRow label="input shape" value={<ValueDisplay value={inputShape} />} />
      <InfoRow label="output shape" value={<ValueDisplay value={outputShape} />} />
      {explanation.shapeSteps.map((step) => (
        <section className={`shape-step shape-step--${classifyShapeStep(step.from, step.to)}`} key={step.label}>
          <h3>{step.label}</h3>
          {step.from !== undefined || step.to !== undefined ? (
            <div className="info-row">
              <span>change</span>
              <strong className="shape-change">
                <ValueDisplay value={step.from ?? 'n/a'} />
                <span className="shape-arrow">&rarr;</span>
                <ValueDisplay value={step.to ?? 'n/a'} />
              </strong>
            </div>
          ) : null}
          {step.reason ? <p>{step.reason}</p> : null}
          {step.substitution ? <code className="formula-code">{step.substitution}</code> : null}
        </section>
      ))}
      {explanation.formula ? (
        <section className="formula-block">
          <h3>Formula</h3>
          <code className="formula-code">{explanation.formula.display}</code>
          {explanation.formula.substitution ? <code className="formula-code formula-code--substitution">{explanation.formula.substitution}</code> : null}
        </section>
      ) : null}
    </section>
  )
}

