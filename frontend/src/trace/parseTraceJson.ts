import { validateTracePayload } from './validateTracePayload'

export function parseTraceJson(text: string) {
  return validateTracePayload(JSON.parse(text) as unknown)
}
