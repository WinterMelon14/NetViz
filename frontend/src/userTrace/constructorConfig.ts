import type { FunctionParameter, SerializableLiteral } from '../desktop/sourceInspectionApi.ts'
import {
  MAX_CONSTRUCTOR_LITERAL_DEPTH,
  MAX_CONSTRUCTOR_LITERAL_VALUES,
  MAX_CONSTRUCTOR_SERIALIZED_BYTES,
  MAX_CONSTRUCTOR_STRING_CHARS,
} from './constants.ts'

export type ConstructorFieldState = {
  enabled: boolean
  text: string
}

export type ConstructorConfigResult =
  | { ok: true; args: SerializableLiteral[]; kwargs: Record<string, SerializableLiteral> }
  | { ok: false; message: string }

function validateLiteral(value: unknown, path: string, depth: number, count: { value: number }): SerializableLiteral {
  if (depth > MAX_CONSTRUCTOR_LITERAL_DEPTH) throw new Error(`${path} exceeds maximum depth ${MAX_CONSTRUCTOR_LITERAL_DEPTH}.`)
  count.value += 1
  if (count.value > MAX_CONSTRUCTOR_LITERAL_VALUES) throw new Error(`Constructor values exceed the limit of ${MAX_CONSTRUCTOR_LITERAL_VALUES}.`)
  if (value === null || typeof value === 'boolean') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${path} must be a finite number.`)
    return value
  }
  if (typeof value === 'string') {
    if (value.length > MAX_CONSTRUCTOR_STRING_CHARS) throw new Error(`${path} exceeds the string length limit.`)
    return value
  }
  if (Array.isArray(value)) return value.map((item, index) => validateLiteral(item, `${path}[${index}]`, depth + 1, count))
  if (typeof value === 'object') {
    const normalized: Record<string, SerializableLiteral> = {}
    for (const [key, item] of Object.entries(value)) normalized[key] = validateLiteral(item, `${path}.${key}`, depth + 1, count)
    return normalized
  }
  throw new Error(`${path} must be a JSON literal.`)
}

export function initialConstructorFields(parameters: FunctionParameter[]): Record<string, ConstructorFieldState> {
  return Object.fromEntries(parameters.map((parameter) => [
    parameter.name,
    Object.prototype.hasOwnProperty.call(parameter, 'defaultValue')
      ? { enabled: true, text: JSON.stringify(parameter.defaultValue) ?? 'null' }
      : parameter.defaultDisplay
        ? { enabled: false, text: parameter.defaultDisplay }
        : { enabled: true, text: '' },
  ]))
}

export function buildConstructorConfig(
  parameters: FunctionParameter[],
  fields: Record<string, ConstructorFieldState>,
): ConstructorConfigResult {
  const args: SerializableLiteral[] = []
  const kwargs: Record<string, SerializableLiteral> = {}
  const count = { value: 0 }
  let hasOmittedPositionalParameter = false
  try {
    for (const parameter of parameters) {
      const field = fields[parameter.name]
      if (!field?.enabled) {
        if (parameter.required) return { ok: false, message: `${parameter.name} is required.` }
        if (parameter.position !== 'keyword_only') hasOmittedPositionalParameter = true
        continue
      }
      if (!field.text.trim()) return { ok: false, message: `${parameter.name} needs a JSON literal value.` }
      const literal = validateLiteral(JSON.parse(field.text), parameter.name, 1, count)
      if (parameter.position === 'positional_only' || (parameter.position === 'positional_or_keyword' && !hasOmittedPositionalParameter)) {
        args.push(literal)
      } else {
        kwargs[parameter.name] = literal
      }
    }
    const serializedBytes = new TextEncoder().encode(JSON.stringify({ args, kwargs })).byteLength
    if (serializedBytes > MAX_CONSTRUCTOR_SERIALIZED_BYTES) {
      return { ok: false, message: `Constructor values exceed ${MAX_CONSTRUCTOR_SERIALIZED_BYTES.toLocaleString()} bytes.` }
    }
    return { ok: true, args, kwargs }
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : 'Constructor values are invalid.' }
  }
}

