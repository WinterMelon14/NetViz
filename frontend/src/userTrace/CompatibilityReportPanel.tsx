import type { CompatibilityCategory, CompatibilityReport, CompatibilityStatus } from '../desktop/sourceInspectionApi.ts'
import { compatibilityFindingKey } from './compatibilityState.ts'

const STATUS_LABELS: Record<CompatibilityStatus, string> = {
  supported: 'Supported',
  configuration_required: 'Configuration required',
  unsupported: 'Unsupported',
  unknown: 'Unknown',
}

const SECTION_LABELS: Array<[CompatibilityCategory, string]> = [
  ['class', 'Model class'],
  ['constructor', 'Constructor'],
  ['forward', 'Forward signature'],
  ['input', 'Representative inputs'],
  ['import', 'Imports'],
  ['resource', 'Resources'],
  ['fx', 'Symbolic tracing'],
]

export function CompatibilityReportPanel({
  report,
  resolvedFindingKeys,
}: {
  report: CompatibilityReport
  resolvedFindingKeys: Set<string>
}) {
  const visibleFindings = report.findings.filter((finding) => finding.category !== 'runtime')
  const counts = visibleFindings.reduce<Record<CompatibilityStatus, number>>((current, finding) => {
    current[finding.status] += 1
    return current
  }, { supported: 0, configuration_required: 0, unsupported: 0, unknown: 0 })

  return (
    <section className="user-trace-section compatibility-report" aria-labelledby="compatibility-report-title">
      <div className="user-trace-section-heading">
        <div><span>5</span><strong id="compatibility-report-title">Compatibility report</strong></div>
        <small>Schema v{report.schemaVersion}</small>
      </div>
      <div className="compatibility-summary" aria-label="Compatibility finding counts">
        {(Object.keys(STATUS_LABELS) as CompatibilityStatus[]).map((status) => (
          <div className={`compatibility-count compatibility-status--${status}`} key={status}>
            <strong>{counts[status]}</strong><span>{STATUS_LABELS[status]}</span>
          </div>
        ))}
      </div>
      <p className="compatibility-outcome">{report.tracingOutcome.statement}</p>
      <div className="compatibility-sections">
        {SECTION_LABELS.map(([category, label]) => {
          const findings = visibleFindings.filter((finding) => finding.category === category)
          if (!findings.length) return null
          const criticalCount = findings.filter((finding) => finding.status === 'unsupported' || finding.status === 'configuration_required').length
          return (
            <details key={category} open={criticalCount > 0}>
              <summary><span>{label}</span><small>{findings.length} finding{findings.length === 1 ? '' : 's'}</small></summary>
              <div className="compatibility-findings">
                {findings.map((finding, index) => {
                  const resolved = finding.status === 'configuration_required' && resolvedFindingKeys.has(compatibilityFindingKey(finding))
                  return (
                    <article className={`compatibility-finding compatibility-status--${finding.status}`} key={`${finding.code}-${index}`}>
                      <header>
                        <strong>{finding.title}</strong>
                        <span>{resolved ? 'Configured' : STATUS_LABELS[finding.status]}</span>
                      </header>
                      <p>{finding.explanation}</p>
                      <small>{finding.origin === 'heuristic' ? 'Conservative source heuristic' : finding.origin === 'runtime' ? 'Current runtime constraint' : 'Observed in source'}</small>
                      {finding.evidence.map((evidence, evidenceIndex) => (
                        <code key={evidenceIndex}>{evidence.lineNumber ? `Line ${evidence.lineNumber}: ` : ''}{evidence.text}</code>
                      ))}
                      {finding.remediation ? <p className="compatibility-remediation">{finding.remediation}</p> : null}
                    </article>
                  )
                })}
              </div>
            </details>
          )
        })}
      </div>
    </section>
  )
}
