import type { CompatibilityFinding } from '../desktop/sourceInspectionApi.ts'
import type { ConstructorFieldState } from './constructorConfig.ts'
import type { ParameterInputDraft } from './structuredInputDrafts.ts'

export type CompatibilityConfiguration = {
  constructorFields: Record<string, ConstructorFieldState>
  inputDrafts: ParameterInputDraft[]
  constructorValid: boolean
  inputsValid: boolean
  useProviderInputs: boolean
}

export function compatibilityFindingKey(finding: CompatibilityFinding) {
  return `${finding.code}:${finding.target?.kind ?? ''}:${finding.target?.name ?? ''}`
}

export function isCompatibilityFindingResolved(finding: CompatibilityFinding, configuration: CompatibilityConfiguration) {
  if (finding.status !== 'configuration_required') return false
  const target = finding.target
  if (target?.kind === 'constructor_parameter' && target.name) {
    const field = configuration.constructorFields[target.name]
    return configuration.constructorValid && Boolean(field?.enabled && field.text.trim())
  }
  if (target?.kind === 'forward_parameter' && target.name) {
    return configuration.useProviderInputs || (configuration.inputsValid && configuration.inputDrafts.some((draft) => draft.parameterName === target.name && draft.included))
  }
  return false
}

export function blockingCompatibilityFindings(findings: CompatibilityFinding[], configuration: CompatibilityConfiguration) {
  return findings.filter((finding) => finding.status === 'unsupported'
    || (finding.status === 'configuration_required' && !isCompatibilityFindingResolved(finding, configuration)))
}
