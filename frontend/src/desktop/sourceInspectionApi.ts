export type SerializableLiteral = null | boolean | number | string | SerializableLiteral[] | { [key: string]: SerializableLiteral }

export type FunctionParameter = {
  name: string
  position: 'positional_only' | 'positional_or_keyword' | 'keyword_only'
  required: boolean
  annotationText?: string
  typeText?: string
  defaultValue?: SerializableLiteral
  defaultDisplay?: string
}

export type ConstructorInspection = {
  kind: 'implicit' | 'explicit'
  supportsNoArgumentConstruction: boolean | 'unknown'
  parameters: FunctionParameter[]
}

export type ForwardSignature = {
  parameters: FunctionParameter[]
  hasVarArgs: boolean
  hasVarKwargs: boolean
  varArgName?: string | null
  varKwargName?: string | null
  lineNumber: number
  isAsync?: boolean
  inputSuggestions?: InputSuggestion[]
}

export type InputSuggestion = {
  parameterName: string
  shapeTemplate: Array<number | null>
  dimensionSources: Array<'inferred' | 'default' | 'unknown'>
  dtypeCategory?: 'floating' | 'integer' | 'boolean'
  confidence: 'high' | 'medium' | 'low'
  evidence: string[]
  consumerPath?: string
  presetKind?: 'image' | 'sequence'
  integerRange?: { min: number; maxExclusive: number }
}

export type ModelCandidate = {
  className: string
  lineNumber: number
  bases: string[]
  confidence: 'confirmed' | 'likely' | 'possible'
  constructor: ConstructorInspection
  forward: ForwardSignature | null
}

export type SourceInspectionWarning = {
  code: string
  message: string
  lineNumber?: number
}

export type SourceInspectionError = {
  code: string
  title: string
  message: string
  stage: 'source_inspection' | string
  details?: Record<string, unknown>
  traceback?: string | null
}

export type InspectModelSourceSuccess = {
  ok: true
  candidates: ModelCandidate[]
  warnings: SourceInspectionWarning[]
  sourceIdentity?: { contentSha256: string; sizeBytes: number }
  exampleInputProvider?: 'netviz_example_inputs' | null
}

export type InspectModelSourceFailure = {
  ok: false
  error: SourceInspectionError
}

export type InspectModelSourceResult = InspectModelSourceSuccess | InspectModelSourceFailure

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeInspectionError(value: unknown): SourceInspectionError {
  if (!isRecord(value)) {
    return {
      code: 'source_inspection_failed',
      title: 'Source inspection failed',
      message: 'The desktop bridge returned an unknown source inspection error.',
      stage: 'source_inspection',
    }
  }

  return {
    code: typeof value.code === 'string' ? value.code : 'source_inspection_failed',
    title: typeof value.title === 'string' ? value.title : 'Source inspection failed',
    message: typeof value.message === 'string' ? value.message : 'The source could not be inspected.',
    stage: typeof value.stage === 'string' ? value.stage : 'source_inspection',
    details: isRecord(value.details) ? value.details : undefined,
    traceback: typeof value.traceback === 'string' || value.traceback === null ? value.traceback : undefined,
  }
}

function isParameter(value: unknown): value is FunctionParameter {
  if (!isRecord(value)) return false
  return typeof value.name === 'string'
    && (value.position === 'positional_only' || value.position === 'positional_or_keyword' || value.position === 'keyword_only')
    && typeof value.required === 'boolean'
}

function parseParameters(value: unknown): FunctionParameter[] {
  return Array.isArray(value) ? value.filter(isParameter) : []
}

function isCandidate(value: unknown): value is ModelCandidate {
  if (!isRecord(value) || !isRecord(value.constructor)) return false
  const confidence = value.confidence
  return typeof value.className === 'string'
    && typeof value.lineNumber === 'number'
    && Array.isArray(value.bases)
    && value.bases.every((base) => typeof base === 'string')
    && (confidence === 'confirmed' || confidence === 'likely' || confidence === 'possible')
}

function normalizeCandidate(value: ModelCandidate): ModelCandidate {
  const forward = isRecord(value.forward)
    ? {
        parameters: parseParameters(value.forward.parameters),
        hasVarArgs: value.forward.hasVarArgs === true,
        hasVarKwargs: value.forward.hasVarKwargs === true,
        varArgName: typeof value.forward.varArgName === 'string' || value.forward.varArgName === null ? value.forward.varArgName : undefined,
        varKwargName: typeof value.forward.varKwargName === 'string' || value.forward.varKwargName === null ? value.forward.varKwargName : undefined,
        lineNumber: typeof value.forward.lineNumber === 'number' ? value.forward.lineNumber : value.lineNumber,
        isAsync: value.forward.isAsync === true,
        inputSuggestions: parseInputSuggestions(value.forward.inputSuggestions),
      }
    : null

  return {
    className: value.className,
    lineNumber: value.lineNumber,
    bases: value.bases,
    confidence: value.confidence,
    constructor: {
      kind: value.constructor.kind === 'explicit' ? 'explicit' : 'implicit',
      supportsNoArgumentConstruction:
        value.constructor.supportsNoArgumentConstruction === true || value.constructor.supportsNoArgumentConstruction === false
          ? value.constructor.supportsNoArgumentConstruction
          : 'unknown',
      parameters: parseParameters(value.constructor.parameters),
    },
    forward,
  }
}

function parseInputSuggestions(value: unknown): InputSuggestion[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!isRecord(item)
      || typeof item.parameterName !== 'string'
      || !Array.isArray(item.shapeTemplate)
      || !item.shapeTemplate.every((dimension) => dimension === null || (typeof dimension === 'number' && Number.isInteger(dimension) && dimension > 0))
      || !Array.isArray(item.evidence)
      || !item.evidence.every((evidence) => typeof evidence === 'string')
      || (item.confidence !== 'high' && item.confidence !== 'medium' && item.confidence !== 'low')) return []
    const dimensionSources = Array.isArray(item.dimensionSources)
      && item.dimensionSources.length === item.shapeTemplate.length
      && item.dimensionSources.every((source) => source === 'inferred' || source === 'default' || source === 'unknown')
      ? item.dimensionSources as InputSuggestion['dimensionSources']
      : item.shapeTemplate.map((dimension) => dimension === null ? 'unknown' as const : 'inferred' as const)
    const integerRange = isRecord(item.integerRange)
      && typeof item.integerRange.min === 'number'
      && typeof item.integerRange.maxExclusive === 'number'
      ? { min: item.integerRange.min, maxExclusive: item.integerRange.maxExclusive }
      : undefined
    return [{
      parameterName: item.parameterName,
      shapeTemplate: item.shapeTemplate as Array<number | null>,
      dimensionSources,
      dtypeCategory: item.dtypeCategory === 'floating' || item.dtypeCategory === 'integer' || item.dtypeCategory === 'boolean' ? item.dtypeCategory : undefined,
      confidence: item.confidence,
      evidence: item.evidence,
      consumerPath: typeof item.consumerPath === 'string' ? item.consumerPath : undefined,
      presetKind: item.presetKind === 'image' || item.presetKind === 'sequence' ? item.presetKind : undefined,
      integerRange,
    }]
  })
}

export function parseInspectModelSourceResult(value: unknown): InspectModelSourceResult {
  if (!isRecord(value)) {
    return {
      ok: false,
      error: {
        code: 'source_protocol_error',
        title: 'Unexpected source inspection response',
        message: 'The desktop bridge returned a response that does not match the source inspection protocol.',
        stage: 'source_inspection',
      },
    }
  }

  if (value.ok === true) {
    const warnings = Array.isArray(value.warnings)
      ? value.warnings.filter((warning): warning is SourceInspectionWarning => isRecord(warning) && typeof warning.code === 'string' && typeof warning.message === 'string')
      : []

    return {
      ok: true,
      candidates: Array.isArray(value.candidates) ? value.candidates.filter(isCandidate).map(normalizeCandidate) : [],
      warnings,
      sourceIdentity: isRecord(value.sourceIdentity)
        && typeof value.sourceIdentity.contentSha256 === 'string'
        && typeof value.sourceIdentity.sizeBytes === 'number'
        ? { contentSha256: value.sourceIdentity.contentSha256, sizeBytes: value.sourceIdentity.sizeBytes }
        : undefined,
      exampleInputProvider: value.exampleInputProvider === 'netviz_example_inputs' ? value.exampleInputProvider : null,
    }
  }

  return {
    ok: false,
    error: normalizeInspectionError(value.error),
  }
}

