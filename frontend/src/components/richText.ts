import type { RichText } from '../explanations'

export function richTextToString(value: RichText): string {
  return value.map((part) => String(part.text)).join('')
}
