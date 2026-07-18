import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const faviconPath = fileURLToPath(new URL('../public/favicon.svg', import.meta.url))
const iconPath = fileURLToPath(new URL('../../packaging/assets/netviz.ico', import.meta.url))

async function requireFavicon() {
  let source
  try {
    source = await readFile(faviconPath, 'utf8')
  } catch {
    throw new Error(`Missing required NetViz favicon: ${faviconPath}`)
  }
  if (!source.includes('<svg') || source.includes('#863bff') || source.includes('color(display-p3 .5252 .23 1)')) {
    throw new Error('favicon.svg must be a user-supplied NetViz SVG, not the Vite placeholder.')
  }
}

async function requireWindowsIcon() {
  let icon
  try {
    icon = await readFile(iconPath)
  } catch {
    throw new Error(`Missing required NetViz Windows icon: ${iconPath}`)
  }
  if (icon.length < 6 || icon.readUInt16LE(0) !== 0 || icon.readUInt16LE(2) !== 1) {
    throw new Error('netviz.ico is not a valid Windows icon container.')
  }
  const count = icon.readUInt16LE(4)
  if (icon.length < 6 + count * 16) throw new Error('netviz.ico has an incomplete directory.')
  const sizes = new Set()
  for (let index = 0; index < count; index += 1) {
    const offset = 6 + index * 16
    const width = icon[offset] || 256
    const height = icon[offset + 1] || 256
    if (width === height) sizes.add(width)
  }
  const missing = [16, 32, 48, 256].filter((size) => !sizes.has(size))
  if (missing.length) throw new Error(`netviz.ico is missing square icon sizes: ${missing.join(', ')}`)
}

await Promise.all([requireFavicon(), requireWindowsIcon()])
console.log('NetViz branding assets are present and valid.')
