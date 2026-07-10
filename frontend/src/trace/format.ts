export function formatShape(shape?: number[]) {
  return shape ? `[${shape.join(', ')}]` : 'scalar'
}

export function shapeParts(shape?: number[]) {
  return shape?.map((dim) => String(dim)) ?? ['scalar']
}

export function formatDtype(dtype?: string) {
  return dtype?.replace('torch.', '') ?? 'n/a'
}

export function formatNumber(value?: number, digits = 3, suffix = '') {
  if (typeof value !== 'number') return 'n/a'
  if (value === 0) return `0${suffix}`
  return `${value.toFixed(Math.abs(value) >= 100 ? 1 : digits)}${suffix}`
}

export function formatPreview(values?: number[]) {
  if (!values?.length) return 'n/a'
  const preview = values.slice(0, 4).map((value) => formatNumber(value, 4))
  return `[${preview.join(', ')}${values.length > 4 ? ', ...' : ''}]`
}

export function formatUnknown(value: unknown) {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (value === null || value === undefined) return 'n/a'
  return JSON.stringify(value)
}

export function isShapeString(value: string) {
  return value === 'scalar' || /^\[[\d\s,-]+\]$/.test(value)
}
