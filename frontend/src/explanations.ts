const noChange = "This transformation does not change the tensor shape";

export type TensorValue = {
  index: number
  role: string
  shape?: number[]
  dtype?: string
  preview?: number[]
  summary?: {
    mean?: number
    std?: number
    min?: number
    max?: number
    zeros_pct?: number
  }
  memory?: {
    human?: string
  }
  from?: string
  source_output?: number
  value?: unknown
}

export type TraceNodeForExplanation = {
  id: string
  label: string
  kind: string
  inputs: TensorValue[]
  outputs: TensorValue[]
  attrs?: Record<string, unknown>
  formula?: string
}

export type ShapeStep = {
  label: string
  from?: unknown
  to?: unknown
  reason?: string
  substitution?: string
}

export type TextPart =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string | number }

export type RichText = TextPart[]

export type Explanation = {
  title: string
  short: RichText
  description: RichText
  formula?: {
    display: string
    substitution?: string
  }
  shapeSteps: ShapeStep[]
}

function text(text: string): TextPart {
  return { kind: 'text', text }
}

function code(value: string | number): TextPart {
  return { kind: 'code', text: value }
}

function rich(...parts: Array<string | TextPart>): RichText {
  return parts.map((part) => (typeof part === 'string' ? text(part) : part))
}

function tensorValues(values: TensorValue[]) {
  return values.filter((value) => value.shape)
}

function shapeText(shape?: number[]) {
  return shape ? `[${shape.join(', ')}]` : 'scalar'
}

function product(values: number[]) {
  return values.reduce((acc, value) => acc * value, 1)
}

function normalizeDim(dim: number, rank: number) {
  return dim < 0 ? rank + dim : dim
}

function normalizeInsertionDim(dim: number, outputRank: number) {
  return dim < 0 ? outputRank + dim : dim
}

function rightAlignedShapeValue(shape: number[], targetRank: number, targetDim: number) {
  const offset = targetRank - shape.length
  return shape[targetDim - offset] ?? 1
}

function numberPair(value: unknown, fallback: [number, number]) {
  if (Array.isArray(value)) {
    const first = Number(value[0] ?? fallback[0])
    const second = Number(value[1] ?? first)
    return [first, second] as const
  }

  if (typeof value === 'number') {
    return [value, value] as const
  }

  return fallback
}

function numberTriple(value: unknown, defaults: [number, number, number]): [number, number, number] {
  if (Array.isArray(value)) return [Number(value[0]), Number(value[1]), Number(value[2])]
  if (typeof value === 'number') return [value, value, value]
  return defaults
}

function stringAttr(value: unknown) {
  return typeof value === 'string' ? value.toLowerCase() : null
}

function numericPaddingPair(value: unknown, fallback: [number, number]) {
  const paddingMode = stringAttr(value)
  if (paddingMode === 'valid') return [0, 0] as const
  if (paddingMode) return null
  return numberPair(value, fallback)
}

function numericPaddingTriple(value: unknown, fallback: [number, number, number]) {
  const paddingMode = stringAttr(value)
  if (paddingMode === 'valid') return [0, 0, 0] as const
  if (paddingMode) return null
  return numberTriple(value, fallback)
}

function ceilModePoolFormula(baseNumerator: string) {
  return `out = ceil((${baseNumerator}) / s) + 1, then drop windows starting in right padding`
}

function floorModePoolFormula(baseNumerator: string) {
  return `out = floor((${baseNumerator}) / s) + 1`
}

function affineSuffix(node: TraceNodeForExplanation, granularity: string, defaultAffine = true): Array<string | TextPart> {
  const affine = node.attrs?.affine
  const elementwiseAffine = node.attrs?.elementwise_affine
  const hasAffine = affine !== undefined ? affine !== false : elementwiseAffine !== undefined ? elementwiseAffine !== false : defaultAffine
  if (!hasAffine) return ['. No learned affine scale or bias is applied.']

  const hasWeight = node.attrs?.weight !== false
  const hasBias = node.attrs?.bias !== false
  if (hasWeight && hasBias) return [', then applies learned ', granularity, ' ', code('weight'), ' and ', code('bias'), '.']
  if (hasWeight) return [', then applies learned ', granularity, ' ', code('weight'), '.']
  if (hasBias) return [', then applies learned ', granularity, ' ', code('bias'), '.']
  return ['. No learned affine scale or bias is applied.']
}

function affineFormula(base: string, node: TraceNodeForExplanation, defaultAffine = true) {
  const affine = node.attrs?.affine
  const elementwiseAffine = node.attrs?.elementwise_affine
  const hasAffine = affine !== undefined ? affine !== false : elementwiseAffine !== undefined ? elementwiseAffine !== false : defaultAffine
  if (!hasAffine || (node.attrs?.weight === false && node.attrs?.bias === false)) return base
  if (node.attrs?.bias === false) return `${base} * weight`
  if (node.attrs?.weight === false) return `${base} + bias`
  return `${base} * weight + bias`
}

// ============================================================================
// Learned Dense And Lookup Layers
// ============================================================================

function explainLinear(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const inFeatures = inputShape.at(-1)
  const outFeatures = outputShape.at(-1)
  const batchIn = inputShape.slice(0, -1)
  const batchOut = outputShape.slice(0, -1)
  const hasBias = node.attrs?.bias !== false

  return {
    title: 'Linear',
    short: rich(
      'Maps the last dimension from ',
      code(`${inFeatures}`),
      ' to ',
      code(`${outFeatures}`),
    ),
    description: rich(
      code('Linear'),
      ' applies a learned matrix multiplication (',
      code(hasBias ? 'y = xWᵀ + b' : 'y = xWᵀ'),
      ') to the last dimension of ',
      code(shapeText(inputShape)),
      '. Leading batch dimensions are preserved.',
    ),
    formula: {
      display: node.formula ?? (hasBias ? 'y = xW^T + b' : 'y = xW^T'),
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Batch dimension',
        from: shapeText(batchIn),
        to: shapeText(batchOut)
      },
      {
        label: 'Feature dimension',
        from: inFeatures,
        to: outFeatures,
        reason: `Linear layers map the last dimension of the tensor to the number of neurons in the layer.`,
      },
    ],
  }
}

function explainEmbedding(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const embeddingDim = Number(node.attrs?.embedding_dim ?? outputShape.at(-1))
  const numEmbeddings = node.attrs?.num_embeddings

  const numEmbeddingsSuffix: Array<string | TextPart> = numEmbeddings
    ? [' from a table of ', code(`${numEmbeddings}`), ' embeddings']
    : []

  return {
    title: 'Embedding',
    short: rich(
      `Looks up a length `,
      code(embeddingDim),
      ` vector for each index in `,
      code(shapeText(inputShape)),
    ),
    description: rich(
      code('Embedding'),
      ' replaces each integer index with a learned vector of length ',
      code(embeddingDim),
      ...numEmbeddingsSuffix,
      '. A new trailing dimension is appended for the embedding vector.',
    ),
    formula: {
      display: 'out = weight[input]',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Index dimensions',
        from: shapeText(inputShape),
        to: shapeText(outputShape.slice(0, -1)),
        reason: 'Index dimensions are preserved.',
      },
      {
        label: 'Embedding dimension',
        from: '-',
        to: embeddingDim,
        reason: `A new trailing dimension of size ${embeddingDim} is appended.`,
      },
    ],
  }
}

// ============================================================================
// Convolution Layers
// ============================================================================
function explainConv1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 3 || outputShape.length < 3) return null

  const [n, c, l] = inputShape
  const [n2, c2, lOut] = outputShape
  const [k] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [s] = numberPair(node.attrs?.stride, [1, 1])
  const paddingMode = stringAttr(node.attrs?.padding)
  const paddingPair = numericPaddingPair(node.attrs?.padding, [0, 0])
  const [p] = paddingPair ?? [0, 0]
  const [d] = numberPair(node.attrs?.dilation, [1, 1])

  return {
    title: 'Conv1d',
    short: rich(
      'Maps channels ',
      code(`${c} ⟶ ${c2}`),
      ' and sequence length ',
      code(`${l} ⟶ ${lOut}`)
    ),
    description: rich(
      code('Conv1d'),
      ' applies learned filters along the sequence length dimension. Batch is preserved, channels become ',
      code('out_channels'),
      ', and length shrinks or grows based on ',
      code('kernel_size'),
      ', ',
      code('stride'),
      ', ',
      code('padding'),
      ', and ',
      code('dilation'),
      '.',
    ),
    formula: {
      display: 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: `Filters map input channels to out_channels=${c2}.` },
      paddingPair
        ? { label: 'Length', from: l, to: lOut, substitution: `floor((${l} + 2*${p} - ${d}*(${k} - 1) - 1) / ${s}) + 1 = ${lOut}` }
        : { label: 'Length', from: l, to: lOut, reason: `String padding "${paddingMode}" determines the output length; numeric padding substitution is omitted.` },
    ],
  }
}


function explainConv2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4 || outputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const [n2, c2, hOut, wOut] = outputShape
  const [kh, kw] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [sh, sw] = numberPair(node.attrs?.stride, [1, 1])
  const paddingMode = stringAttr(node.attrs?.padding)
  const paddingPair = numericPaddingPair(node.attrs?.padding, [0, 0])
  const [ph, pw] = paddingPair ?? [0, 0]
  const [dh, dw] = numberPair(node.attrs?.dilation, [1, 1])

  return {
    title: 'Conv2d',
    short: rich(
      'Maps channels ',
      code(`${c} ⟶ ${c2}`),
      ' and spatial size ',
      code(`${h}x${w} ⟶ ${hOut}x${wOut}`),
    ),
    description: rich(
      code('Conv2d'),
      ' applies learned filters across height and width. Batch is preserved, channels become ',
      code('out_channels'),
      ', and spatial dimensions shrink or grow based on ',
      code('kernel_size'),
      ', ',
      code('stride'),
      ', ',
      code('padding'),
      ', and ',
      code('dilation.'),
    ),
    formula: {
      display: 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: `Filters map input channels to out_channels=${c2}.` },
      paddingPair
        ? { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` }
        : { label: 'Height', from: h, to: hOut, reason: `String padding "${paddingMode}" determines the output height; numeric padding substitution is omitted.` },
      paddingPair
        ? { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` }
        : { label: 'Width', from: w, to: wOut, reason: `String padding "${paddingMode}" determines the output width; numeric padding substitution is omitted.` },
    ],
  }
}

function explainConv3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5 || outputShape.length < 5) return null

  const [n, c, d, h, w] = inputShape
  const [n2, c2, dOut, hOut, wOut] = outputShape
  const [kd, kh, kw] = numberTriple(node.attrs?.kernel_size, [0, 0, 0])
  const [sd, sh, sw] = numberTriple(node.attrs?.stride, [1, 1, 1])
  const paddingMode = stringAttr(node.attrs?.padding)
  const paddingTriple = numericPaddingTriple(node.attrs?.padding, [0, 0, 0])
  const [pd, ph, pw] = paddingTriple ?? [0, 0, 0]
  const [dd, dh, dw] = numberTriple(node.attrs?.dilation, [1, 1, 1])

  return {
    title: 'Conv3d',
    short: rich(
      'Maps channels ',
      code(`${c} ⟶ ${c2}`),
      ' and spatial size ',
      code(`${d}x${h}x${w} ⟶ ${dOut}x${hOut}x${wOut}`)
    ),
    description: rich(
      code('Conv3d'),
      ' applies learned filters across depth, height, and width. Batch is preserved, channels become ',
      code('out_channels'),
      ', and spatial dimensions shrink or grow based on ',
      code('kernel_size'),
      ', ',
      code('stride'),
      ', ',
      code('padding'),
      ', and ',
      code('dilation'),
      '.',
    ),
    formula: {
      display: 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: `Filters map input channels to out_channels=${c2}.` },
      paddingTriple
        ? { label: 'Depth', from: d, to: dOut, substitution: `floor((${d} + 2*${pd} - ${dd}*(${kd} - 1) - 1) / ${sd}) + 1 = ${dOut}` }
        : { label: 'Depth', from: d, to: dOut, reason: `String padding "${paddingMode}" determines the output depth; numeric padding substitution is omitted.` },
      paddingTriple
        ? { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` }
        : { label: 'Height', from: h, to: hOut, reason: `String padding "${paddingMode}" determines the output height; numeric padding substitution is omitted.` },
      paddingTriple
        ? { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` }
        : { label: 'Width', from: w, to: wOut, reason: `String padding "${paddingMode}" determines the output width; numeric padding substitution is omitted.` },
    ],
  }
}

// ============================================================================
// Pooling Layers
// ============================================================================

function explainMaxPool1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 3 || outputShape.length < 3) return null

  const [n, c, l] = inputShape
  const [n2, c2, lOut] = outputShape
  const [k] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [s] = numberPair(node.attrs?.stride, [k, k])
  const [p] = numberPair(node.attrs?.padding, [0, 0])
  const [d] = numberPair(node.attrs?.dilation, [1, 1])
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'MaxPool1d',
    short: rich(
      'Pools L from ',
      code(`${l}`),
      ' to ',
      code(`${lOut}`),
      ' using ',
      code(`kernel=${k}`),
      ', ',
      code(`stride=${s}`)
    ),
    description: rich(
      code('MaxPool1d'),
      ' slides a 1D window along the sequence length and keeps the maximum value in each window. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: ceilMode
        ? ceilModePoolFormula('in + 2p - d(k - 1) - 1')
        : 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: ceilMode ? undefined : `floor((${l} + 2*${p} - ${d}*(${k} - 1) - 1) / ${s}) + 1 = ${lOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Length', from: l, to: lOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Length', from: l, to: lOut, substitution: `floor((${l} + 2*${p} - ${d}*(${k} - 1) - 1) / ${s}) + 1 = ${lOut}` },
    ],
  }
}

function explainMaxPool2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4 || outputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const [n2, c2, hOut, wOut] = outputShape
  const [kh, kw] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [sh, sw] = numberPair(node.attrs?.stride, [kh, kw])
  const [ph, pw] = numberPair(node.attrs?.padding, [0, 0])
  const [dh, dw] = numberPair(node.attrs?.dilation, [1, 1])
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'MaxPool2d',
    short: rich(
      'Pools H and W from ',
      code(`${h}x${w}`),
      ' to ',
      code(`${hOut}x${wOut}`),
      ' using ',
      code(`kernel=${kh}x${kw}`),
      ', ',
      code(`stride=${sh}x${sw}`)
    ),
    description: rich(
      code('MaxPool2d'),
      ' slides a 2D window over height and width and keeps the maximum value in each window. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: ceilMode
        ? ceilModePoolFormula('in + 2p - d(k - 1) - 1')
        : 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: ceilMode ? undefined : `H: floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Height', from: h, to: hOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` },
      ceilMode
        ? { label: 'Width', from: w, to: wOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` },
    ],
  }
}

function explainMaxPool3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5 || outputShape.length < 5) return null

  const [n, c, d, h, w] = inputShape
  const [n2, c2, dOut, hOut, wOut] = outputShape
  const [kd, kh, kw] = numberTriple(node.attrs?.kernel_size, [0, 0, 0])
  const [sd, sh, sw] = numberTriple(node.attrs?.stride, [kd, kh, kw])
  const [pd, ph, pw] = numberTriple(node.attrs?.padding, [0, 0, 0])
  const [dd, dh, dw] = numberTriple(node.attrs?.dilation, [1, 1, 1])
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'MaxPool3d',
    short: rich(
      'Pools D, H, W from ',
      code(`${d}x${h}x${w}`),
      ' to ',
      code(`${dOut}x${hOut}x${wOut}`),
      ' using ',
      code(`kernel=${kd}x${kh}x${kw}`)
    ),
    description: rich(
      code('MaxPool3d'),
      ' slides a 3D window over depth, height, and width and keeps the maximum value in each window. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: ceilMode
        ? ceilModePoolFormula('in + 2p - d(k - 1) - 1')
        : 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: ceilMode ? undefined : `D: floor((${d} + 2*${pd} - ${dd}*(${kd} - 1) - 1) / ${sd}) + 1 = ${dOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Depth', from: d, to: dOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Depth', from: d, to: dOut, substitution: `floor((${d} + 2*${pd} - ${dd}*(${kd} - 1) - 1) / ${sd}) + 1 = ${dOut}` },
      ceilMode
        ? { label: 'Height', from: h, to: hOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` },
      ceilMode
        ? { label: 'Width', from: w, to: wOut, reason: 'Ceil mode may include one extra pooling position, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` },
    ],
  }
}

function explainAvgPool1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 3 || outputShape.length < 3) return null

  const [n, c, l] = inputShape
  const [n2, c2, lOut] = outputShape
  const [k] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [s] = numberPair(node.attrs?.stride, [k, k])
  const [p] = numberPair(node.attrs?.padding, [0, 0])
  const countIncludePad = node.attrs?.count_include_pad ?? true
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'AvgPool1d',
    short: rich(
      'Pools L from ',
      code(`${l}`),
      ' to ',
      code(`${lOut}`),
      ' using ',
      code(`kernel=${k}`),
      ', ',
      code(`stride=${s}`)
    ),
    description: rich(
      code('AvgPool1d'),
      ' slides a 1D window along the sequence length and averages the values in each window. Batch and channel dimensions are preserved.',
      ...(!countIncludePad ? [' Padded positions are ', code('excluded'), ' from the average.'] : []),
    ),
    formula: {
      display: ceilMode ? ceilModePoolFormula('in + 2p - k') : floorModePoolFormula('in + 2p - k'),
      substitution: ceilMode ? undefined : `floor((${l} + 2*${p} - ${k}) / ${s}) + 1 = ${lOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Length', from: l, to: lOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Length', from: l, to: lOut, substitution: `floor((${l} + 2*${p} - ${k}) / ${s}) + 1 = ${lOut}` },
    ],
  }
}

function explainAvgPool2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4 || outputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const [n2, c2, hOut, wOut] = outputShape
  const [kh, kw] = numberPair(node.attrs?.kernel_size, [0, 0])
  const [sh, sw] = numberPair(node.attrs?.stride, [kh, kw])
  const [ph, pw] = numberPair(node.attrs?.padding, [0, 0])
  const countIncludePad = node.attrs?.count_include_pad ?? true
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'AvgPool2d',
    short: rich(
      'Pools H and W from ',
      code(`${h}x${w}`),
      ' to ',
      code(`${hOut}x${wOut}`),
      ' using ',
      code(`kernel=${kh}x${kw}`),
      ', ',
      code(`stride=${sh}x${sw}`)
    ),
    description: rich(
      code('AvgPool2d'),
      ' slides a 2D window over height and width and averages the values in each window. Batch and channel dimensions are preserved.',
      ...(!countIncludePad ? [' Padded positions are ', code('excluded'), ' from the average.'] : []),
    ),
    formula: {
      display: ceilMode ? ceilModePoolFormula('in + 2p - k') : floorModePoolFormula('in + 2p - k'),
      substitution: ceilMode ? undefined : `H: floor((${h} + 2*${ph} - ${kh}) / ${sh}) + 1 = ${hOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Height', from: h, to: hOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${kh}) / ${sh}) + 1 = ${hOut}` },
      ceilMode
        ? { label: 'Width', from: w, to: wOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${kw}) / ${sw}) + 1 = ${wOut}` },
    ],
  }
}

function explainAvgPool3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5 || outputShape.length < 5) return null

  const [n, c, d, h, w] = inputShape
  const [n2, c2, dOut, hOut, wOut] = outputShape
  const [kd, kh, kw] = numberTriple(node.attrs?.kernel_size, [0, 0, 0])
  const [sd, sh, sw] = numberTriple(node.attrs?.stride, [kd, kh, kw])
  const [pd, ph, pw] = numberTriple(node.attrs?.padding, [0, 0, 0])
  const countIncludePad = node.attrs?.count_include_pad ?? true
  const ceilMode = node.attrs?.ceil_mode === true

  return {
    title: 'AvgPool3d',
    short: rich(
      'Pools D, H, W from ',
      code(`${d}x${h}x${w}`),
      ' to ',
      code(`${dOut}x${hOut}x${wOut}`),
      ' using ',
      code(`kernel=${kd}x${kh}x${kw}`)
    ),
    description: rich(
      code('AvgPool3d'),
      ' slides a 3D window over depth, height, and width and averages the values in each window. Batch and channel dimensions are preserved.',
      ...(!countIncludePad ? [' Padded positions are ', code('excluded'), ' from the average.'] : []),
    ),
    formula: {
      display: ceilMode ? ceilModePoolFormula('in + 2p - k') : floorModePoolFormula('in + 2p - k'),
      substitution: ceilMode ? undefined : `D: floor((${d} + 2*${pd} - ${kd}) / ${sd}) + 1 = ${dOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      ceilMode
        ? { label: 'Depth', from: d, to: dOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Depth', from: d, to: dOut, substitution: `floor((${d} + 2*${pd} - ${kd}) / ${sd}) + 1 = ${dOut}` },
      ceilMode
        ? { label: 'Height', from: h, to: hOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${kh}) / ${sh}) + 1 = ${hOut}` },
      ceilMode
        ? { label: 'Width', from: w, to: wOut, reason: 'Ceil mode may include one extra averaging window, then PyTorch drops windows that would start in right-side padding.' }
        : { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${kw}) / ${sw}) + 1 = ${wOut}` },
    ],
  }
}

function explainAdaptiveAvgPool1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 3 || outputShape.length < 3) return null

  const [n, c, l] = inputShape
  const [n2, c2, lOut] = outputShape

  return {
    title: 'AdaptiveAvgPool1d',
    short: rich(
      'Adaptively pools L from ',
      code(`${l}`),
      ' to ',
      code(`${lOut}`)
    ),
    description: rich(
      code('AdaptiveAvgPool1d'),
      ' averages a selected input region for each output position to hit a target output length of ',
      code(`${lOut}`),
      '. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: 'each output position averages a selected input region; region sizes and boundaries may vary and can overlap',
      substitution: `L: ${l} ⟶ ${lOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      { label: 'Length', from: l, to: lOut, reason: `Output length is fixed at ${lOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
    ],
  }
}

function explainAdaptiveAvgPool2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4 || outputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const [n2, c2, hOut, wOut] = outputShape

  return {
    title: 'AdaptiveAvgPool2d',
    short: rich(
      'Adaptively pools H and W from ',
      code(`${h}x${w}`),
      ' to ',
      code(`${hOut}x${wOut}`)
    ),
    description: rich(
      code('AdaptiveAvgPool2d'),
      ' averages a selected input region for each output position to hit a target output size of ',
      code(`${hOut}x${wOut}`),
      '. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: 'each output position averages a selected input region; region sizes and boundaries may vary and can overlap',
      substitution: `H: ${h} ⟶ ${hOut}, W: ${w} ⟶ ${wOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      { label: 'Height', from: h, to: hOut, reason: `Output height is fixed at ${hOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
      { label: 'Width', from: w, to: wOut, reason: `Output width is fixed at ${wOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
    ],
  }
}

function explainAdaptiveAvgPool3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5 || outputShape.length < 5) return null

  const [n, c, d, h, w] = inputShape
  const [n2, c2, dOut, hOut, wOut] = outputShape

  return {
    title: 'AdaptiveAvgPool3d',
    short: rich(
      'Adaptively pools D, H, W from ',
      code(`${d}x${h}x${w}`),
      ' to ',
      code(`${dOut}x${hOut}x${wOut}`)
    ),
    description: rich(
      code('AdaptiveAvgPool3d'),
      ' averages a selected input region for each output position to hit a target output size of ',
      code(`${dOut}x${hOut}x${wOut}`),
      '. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: 'each output position averages a selected input region; region sizes and boundaries may vary and can overlap',
      substitution: `D: ${d} ⟶ ${dOut}, H: ${h} ⟶ ${hOut}, W: ${w} ⟶ ${wOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      { label: 'Depth', from: d, to: dOut, reason: `Output depth is fixed at ${dOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
      { label: 'Height', from: h, to: hOut, reason: `Output height is fixed at ${hOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
      { label: 'Width', from: w, to: wOut, reason: `Output width is fixed at ${wOut}; region sizes and boundaries may vary and can overlap when sizes do not divide evenly.` },
    ],
  }
}
// ============================================================================
// Normalization Layers
// ============================================================================

function batchNormStatSource(node: TraceNodeForExplanation): Array<string | TextPart> {
  if (batchNormUsesCurrentBatchStats(node)) return ['statistics computed from the current batch']
  if (batchNormHasRunningStats(node)) return ['stored running statistics from training']
  return ['stored running statistics when available; otherwise current-batch statistics']
}

function batchNormUsesCurrentBatchStats(node: TraceNodeForExplanation) {
  const training = node.attrs?.training === true
  const trackRunningStats = node.attrs?.track_running_stats !== false
  return training || !trackRunningStats
}

function batchNormHasRunningStats(node: TraceNodeForExplanation) {
  const hasRunningStats =
    node.inputs.some((input) => input.role === 'running_mean' || input.role === 'running_var')
    || node.attrs?.running_mean !== undefined
    || node.attrs?.running_var !== undefined
    || node.attrs?.track_running_stats === true
  return hasRunningStats
}

function batchNormStatSubstitution(node: TraceNodeForExplanation, currentBatchText: string) {
  if (batchNormUsesCurrentBatchStats(node)) return currentBatchText
  if (batchNormHasRunningStats(node)) return 'stored statistics applied independently to each channel'
  return 'stored statistics applied when available; otherwise current-batch statistics are used'
}

function batchNormAffineDescription(node: TraceNodeForExplanation): Array<string | TextPart> {
  if (node.attrs?.affine === false) return ['. No learned affine scale or bias is applied.']

  const hasWeight = node.attrs?.weight !== false
  const hasBias = node.attrs?.bias !== false
  if (hasWeight && hasBias) {
    return [', then applies learned per-channel ', code('weight'), ' and ', code('bias'), '.']
  }
  if (hasWeight) return [', then applies learned per-channel ', code('weight'), '.']
  if (hasBias) return [', then applies learned per-channel ', code('bias'), '.']
  return ['. No learned affine scale or bias is applied.']
}

function batchNormFormula(node: TraceNodeForExplanation) {
  if (node.attrs?.affine === false || (node.attrs?.weight === false && node.attrs?.bias === false)) {
    return 'out = (x - mean) / sqrt(var + eps)'
  }
  if (node.attrs?.bias === false) return 'out = (x - mean) / sqrt(var + eps) * weight'
  if (node.attrs?.weight === false) return 'out = (x - mean) / sqrt(var + eps) + bias'
  return 'out = (x - mean) / sqrt(var + eps) * weight + bias'
}

function explainBatchNorm1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 2) return null

  const [n, c, l] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const statSource = batchNormStatSource(node)
  const affineDescription = batchNormAffineDescription(node)

  const spatialNote: Array<string | TextPart> = l !== undefined
    ? [` across N=${n}, L=${l} for each channel`]
    : [` across N=${n} for each feature`]

  return {
    title: 'BatchNorm1d',
    short: rich(`Normalizes each of the ${c} channels/features across the batch`),
    description: rich(
      code('BatchNorm1d'),
      ' normalizes each channel using ',
      ...statSource,
      '. It subtracts the mean and divides by the standard deviation (',
      code(`eps=${eps}`),
      ')',
      ...spatialNote,
      ...affineDescription,
    ),
    formula: {
      display: batchNormFormula(node),
      substitution: batchNormStatSubstitution(
        node,
        l !== undefined
          ? `statistics computed per-channel over current N=${n}, L=${l}`
          : `statistics computed per-feature over current N=${n}`,
      ),
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainBatchNorm2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const statSource = batchNormStatSource(node)
  const affineDescription = batchNormAffineDescription(node)

  return {
    title: 'BatchNorm2d',
    short: rich(`Normalizes each of the ${c} channels independently across batch and spatial dims`),
    description: rich(
      code('BatchNorm2d'),
      ' normalizes each channel using ',
      ...statSource,
      '. It subtracts the mean and divides by the standard deviation (',
      code(`eps=${eps}`),
      `) across N=${n}, H=${h}, W=${w} for each channel`,
      ...affineDescription,
    ),
    formula: {
      display: batchNormFormula(node),
      substitution: batchNormStatSubstitution(node, `statistics computed per-channel over current N=${n}, H=${h}, W=${w}`),
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainBatchNorm3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5) return null

  const [n, c, d, h, w] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const statSource = batchNormStatSource(node)
  const affineDescription = batchNormAffineDescription(node)

  return {
    title: 'BatchNorm3d',
    short: rich(`Normalizes each of the ${c} channels independently across batch and volumetric dims`),
    description: rich(
      code('BatchNorm3d'),
      ' normalizes each channel using ',
      ...statSource,
      '. It subtracts the mean and divides by the standard deviation (',
      code(`eps=${eps}`),
      `) across N=${n}, D=${d}, H=${h}, W=${w} for each channel`,
      ...affineDescription,
    ),
    formula: {
      display: batchNormFormula(node),
      substitution: batchNormStatSubstitution(node, `statistics computed per-channel over current N=${n}, D=${d}, H=${h}, W=${w}`),
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainLayerNorm(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const rawNormalizedShape = node.attrs?.normalized_shape
  const normDims = Array.isArray(rawNormalizedShape) ? rawNormalizedShape.map(Number) : inputShape.slice(-1)
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const affineDescription = affineSuffix(node, 'per-element')

  return {
    title: 'LayerNorm',
    short: rich('Normalizes over the trailing dimension'),
    description: rich(
      code('LayerNorm'),
      ' normalizes each sample independently across the trailing dimensions ',
      code(shapeText(normDims)),
      ' by subtracting the mean and dividing by the standard deviation (',
      code(`eps=${eps}`),
      ')',
      ...affineDescription,
      ' Other dimensions are normalized independently.',
    ),
    formula: {
      display: affineFormula('out = (x - mean) / sqrt(var + eps)', node),
      substitution: `mean/var computed over the last ${normDims.length} dim(s): ${shapeText(normDims)}`,
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainRMSNorm(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const eps = node.attrs?.eps
  const epsText = eps === undefined || eps === null ? 'dtype-dependent default eps' : `eps=${Number(eps)}`
  const affineDescription = affineSuffix(node, 'per-element')

  return {
    title: 'RMSNorm',
    short: rich('Normalizes by root-mean-square'),
    description: rich(
      code('RMSNorm'),
      ' divides each value by the root mean square of the elements along the normalized dimension (',
      code(epsText),
      ')',
      ...affineDescription,
      ' It skips the mean subtraction step of ',
      code('LayerNorm'),
      ', so it can be cheaper and has been used successfully in LLaMA, Mistral, and other modern LLMs.',
    ),
    formula: {
      display: affineFormula('out = x / sqrt(mean(x^2) + eps)', node),
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainGroupNorm(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const [, c] = inputShape
  const numGroups = Number(node.attrs?.num_groups ?? 1)
  const chansPerGroup = c / numGroups
  const affineDescription = affineSuffix(node, 'per-channel')

  return {
    title: 'GroupNorm',
    short: rich(
      `Divides the ${c} channels into `,
      code(`${numGroups} groups`),
      ' and normalizes each group independently',
    ),
    description: rich(
      code('GroupNorm'),
      ` splits the ${c} channels into `,
      code(`${numGroups}`),
      ` groups of `,
      code(`${chansPerGroup}`),
      ' channels each, then normalizes within each group across channels and spatial dims. Unlike ',
      code('BatchNorm'),
      ', it does not depend on batch size, making it suitable for small batches and recurrent models.',
      ...affineDescription,
    ),
    formula: {
      display: affineFormula('out = (x - mean) / sqrt(var + eps)', node),
      substitution: `mean/var computed per group (${chansPerGroup} channels/group) per sample`,
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainInstanceNorm1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 3) return null

  const [, c, l] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const affineDescription = affineSuffix(node, 'per-channel', false)
  const usesCurrentStats = node.attrs?.training === true || node.attrs?.track_running_stats !== true
  const statDescription = usesCurrentStats
    ? 'statistics are computed per sample rather than across the batch, so batch size has no effect.'
    : 'stored running statistics from training are applied independently to each channel.'
  const statSubstitution = usesCurrentStats
    ? `mean/var computed per channel per sample over L=${l}`
    : 'stored statistics applied independently to each channel'

  return {
    title: 'InstanceNorm1d',
    short: rich(`Normalizes each of the ${c} channels independently per sample`),
    description: rich(
      code('InstanceNorm1d'),
      ' normalizes each channel of each sample independently across the length dimension (',
      code(`L=${l}`),
      ', ',
      code(`eps=${eps}`),
      '). Unlike ',
      code('BatchNorm'),
      ', ',
      statDescription,
      ...affineDescription,
    ),
    formula: {
      display: affineFormula('out = (x - mean) / sqrt(var + eps)', node, false),
      substitution: statSubstitution,
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainInstanceNorm2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4) return null

  const [, c, h, w] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const affineDescription = affineSuffix(node, 'per-channel', false)
  const usesCurrentStats = node.attrs?.training === true || node.attrs?.track_running_stats !== true
  const statDescription = usesCurrentStats
    ? 'statistics are computed per sample rather than across the batch.'
    : 'stored running statistics from training are applied independently to each channel.'
  const statSubstitution = usesCurrentStats
    ? `mean/var computed per channel per sample over H=${h}, W=${w}`
    : 'stored statistics applied independently to each channel'

  return {
    title: 'InstanceNorm2d',
    short: rich(`Normalizes each of the ${c} channels independently per sample across H and W`),
    description: rich(
      code('InstanceNorm2d'),
      ' normalizes each channel of each sample independently across the spatial dimensions (',
      code(`H=${h}, W=${w}`),
      ', ',
      code(`eps=${eps}`),
      '). ',
      statDescription,
      ' Widely used in style transfer where per-sample, per-channel statistics carry style information.',
      ...affineDescription,
    ),
    formula: {
      display: affineFormula('out = (x - mean) / sqrt(var + eps)', node, false),
      substitution: statSubstitution,
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

function explainInstanceNorm3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 5) return null

  const [, c, d, h, w] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const affineDescription = affineSuffix(node, 'per-channel', false)
  const usesCurrentStats = node.attrs?.training === true || node.attrs?.track_running_stats !== true
  const statDescription = usesCurrentStats
    ? 'statistics are computed per sample rather than across the batch.'
    : 'stored running statistics from training are applied independently to each channel.'
  const statSubstitution = usesCurrentStats
    ? `mean/var computed per channel per sample over D=${d}, H=${h}, W=${w}`
    : 'stored statistics applied independently to each channel'

  return {
    title: 'InstanceNorm3d',
    short: rich(`Normalizes each of the ${c} channels independently per sample across D, H, and W`),
    description: rich(
      code('InstanceNorm3d'),
      ' normalizes each channel of each sample independently across the volumetric dimensions (',
      code(`D=${d}, H=${h}, W=${w}`),
      ', ',
      code(`eps=${eps}`),
      '). ',
      statDescription,
      ' The 3D counterpart to ',
      code('InstanceNorm2d'),
      ', used in volumetric medical imaging and video models.',
      ...affineDescription,
    ),
    formula: {
      display: affineFormula('out = (x - mean) / sqrt(var + eps)', node, false),
      substitution: statSubstitution,
    },
    shapeSteps: [
      { label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange },
    ],
  }
}

// ============================================================================
// Activation Layers And Functions
// ============================================================================

// ─── ReLU family ─────────────────────────────────────────────────────────────

function explainReLU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const inplace = node.attrs?.inplace ?? false

  return {
    title: 'ReLU',
    short: rich('Clamps all negative values to ', code('0'), ' element-wise'),
    description: rich(
      code('ReLU'),
      ' applies ',
      code('max(0, x)'),
      ' to every element. Negative values become ',
      code('0'),
      ', positive values pass through unchanged.',
      ...(inplace ? [' Operates ', code('in-place'), ' on the input tensor.'] : []),
    ),
    formula: { display: 'out = max(0, x)' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainLeakyReLU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const slope = Number(node.attrs?.negative_slope ?? 0.01)
  const inplace = node.attrs?.inplace ?? false

  return {
    title: 'LeakyReLU',
    short: rich('Like ReLU, but negative values leak through with slope ', code(`${slope}`)),
    description: rich(
      code('LeakyReLU'),
      ' passes positive values through unchanged and scales negative values by ',
      code(`negative_slope=${slope}`),
      ' instead of clamping them to zero. This can reduce dead-neuron behavior during training.',
      ...(inplace ? [' Operates ', code('in-place'), ' on the input tensor.'] : []),
    ),
    formula: { display: `out = x if x >= 0 else ${slope} * x` },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainELU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const alpha = Number(node.attrs?.alpha ?? 1.0)
  const inplace = node.attrs?.inplace ?? false

  return {
    title: 'ELU',
    short: rich('Exponential Linear Unit: negative values curve smoothly toward ', code(`-${alpha}`)),
    description: rich(
      code('ELU'),
      ' passes positive values through unchanged and replaces negative values with ',
      code(`alpha * (exp(x) - 1)`),
      ` (alpha=${alpha}). Unlike Leaky ReLU, the negative branch is smooth and saturates, which can help gradient flow in some networks.`,
      ...(inplace ? [' Operates ', code('in-place'), ' on the input tensor.'] : []),
    ),
    formula: { display: `out = x if x >= 0 else ${alpha} * (exp(x) - 1)` },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainSELU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'SELU',
    short: rich('Self-normalizing activation: output mean and variance are automatically stabilized'),
    description: rich(
      code('SELU'),
      ' is a scaled ELU with fixed constants ',
      code('alpha ≈ 1.6733'),
      ' and ',
      code('scale ≈ 1.0507'),
      ' designed to help activations self-normalize toward zero mean and unit variance in compatible network architectures.',
    ),
    formula: { display: 'out = scale * (x if x >= 0 else alpha * (exp(x) - 1))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

// ─── Sigmoid family ───────────────────────────────────────────────────────────

function explainSigmoid(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'Sigmoid',
    short: rich('Squashes all values to the range ', code('(0, 1)'), ' element-wise'),
    description: rich(
      code('Sigmoid'),
      ' maps every value to ',
      code('(0, 1)'),
      ' via ',
      code('1 / (1 + exp(-x))'),
      '. Large positive values approach ',
      code('1'),
      ', large negative values approach ',
      code('0'),
      '. Commonly used for binary classification outputs.',
    ),
    formula: { display: 'out = 1 / (1 + exp(-x))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainHardsigmoid(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'Hardsigmoid',
    short: rich('Piecewise-linear approximation of ', code('Sigmoid'), ', squashing to ', code('(0, 1)')),
    description: rich(
      code('Hardsigmoid'),
      ' approximates sigmoid with a clipped linear function: ',
      code('0'),
      ' below ',
      code('-3'),
      ', ',
      code('1'),
      ' above ',
      code('3'),
      ', and linear between. Much cheaper to compute than the real sigmoid.',
    ),
    formula: { display: 'out = clamp(x / 6 + 0.5, 0, 1)' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainTanh(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'Tanh',
    short: rich('Squashes all values to the range ', code('(-1, 1)'), ' element-wise'),
    description: rich(
      code('Tanh'),
      ' maps every value to ',
      code('(-1, 1)'),
      ' via ',
      code('(exp(x) - exp(-x)) / (exp(x) + exp(-x))'),
      '. Zero-centered, unlike Sigmoid, which makes it preferable in many hidden layer contexts.',
    ),
    formula: { display: 'out = (exp(x) - exp(-x)) / (exp(x) + exp(-x))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

// ─── Swish / SiLU family ─────────────────────────────────────────────────────

function explainSiLU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'SiLU',
    short: rich('Swish/SiLU: ', code('x * sigmoid(x)'), ' element-wise'),
    description: rich(
      code('SiLU'),
      ' (also called Swish) multiplies each value by its own sigmoid, producing a smooth non-monotonic activation. It can outperform ',
      code('ReLU'),
      ' on some deep models and is the default activation in many modern architectures.',
    ),
    formula: { display: 'out = x * sigmoid(x) = x / (1 + exp(-x))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainMish(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'Mish',
    short: rich('Smooth, non-monotonic activation similar to Swish'),
    description: rich(
      code('Mish'),
      ' applies ',
      code('x * tanh(softplus(x))'),
      ' element-wise. Like SiLU, it is smooth and non-monotonic, allowing small negative values to pass through. Often used in object detection models.',
    ),
    formula: { display: 'out = x * tanh(ln(1 + exp(x)))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainHardswish(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'Hardswish',
    short: rich('Piecewise-linear approximation of Swish, optimized for mobile'),
    description: rich(
      code('Hardswish'),
      ' approximates SiLU using a clipped linear function, making it cheaper to compute while retaining most of the accuracy benefit. Introduced in MobileNetV3.',
    ),
    formula: { display: 'out = x * clamp(x + 3, 0, 6) / 6' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

// ─── Softmax family ───────────────────────────────────────────────────────────

function explainSoftmax(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const dim = node.attrs?.dim !== undefined ? Number(node.attrs.dim) : null

  return {
    title: 'Softmax',
    short: rich(
      'Converts values along ',
      ...(dim !== null ? [code(`dim=${dim}`)] : ['a dimension']),
      ' to a probability distribution summing to ',
      code('1')
    ),
    description: rich(
      code('Softmax'),
      ' exponentiates each value and divides by the sum of all exponentiated values along the target dimension. The output sums to ',
      code('1'),
      ' and is in the range ',
      code('[0, 1]'),
      ', making it suitable for multi-class classification outputs.',
    ),
    formula: { display: 'out_i = exp(x_i) / sum(exp(x_j))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainLogSoftmax(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const dim = node.attrs?.dim !== undefined ? Number(node.attrs.dim) : null

  return {
    title: 'LogSoftmax',
    short: rich(
      'Log of Softmax along ',
      ...(dim !== null ? [code(`dim=${dim}`)] : ['a dimension']),
    ),
    description: rich(
      code('LogSoftmax'),
      ' computes ',
      code('log(softmax(x))'),
      ' in a numerically stable way. Preferred over applying ',
      code('Softmax'),
      ' then ',
      code('log'),
      ' separately, and pairs directly with ',
      code('NLLLoss'),
      ' (equivalent to using ',
      code('CrossEntropyLoss'),
      ' with raw logits).',
    ),
    formula: { display: 'out_i = x_i - log(sum(exp(x_j)))' },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

// ─── Other ────────────────────────────────────────────────────────────────────

function explainGELU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const approximate = String(node.attrs?.approximate ?? 'none')
  const isTanh = approximate === 'tanh'

  const approximateSuffix: Array<string | TextPart> = isTanh
    ? [' Using the ', code(approximate), ' approximation for speed.']
    : []

  return {
    title: 'GELU',
    short: rich('Applies the Gaussian Error Linear Unit activation element-wise'),
    description: rich(
      code('GELU'),
      ' smoothly scales each input value by the standard normal CDF at that value, behaving like a smoother version of ',
      code('ReLU'),
      '.',
      ...approximateSuffix,
    ),
    formula: {
      display: isTanh
        ? 'out ≈ 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))'
        : 'out = x * Phi(x)',
    },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

function explainSoftplus(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const beta = Number(node.attrs?.beta ?? 1)
  const threshold = Number(node.attrs?.threshold ?? 20)

  return {
    title: 'Softplus',
    short: rich('Smooth approximation of ', code('ReLU'), ': mathematically positive; may underflow to zero numerically'),
    description: rich(
      code('Softplus'),
      ' computes ',
      code(`(1 / beta) * log(1 + exp(beta * x))`),
      ` (beta=${beta}). It approximates ReLU but is smooth and mathematically positive for finite real inputs. Reverts to a linear function when `,
      code('beta * input'),
      ' is above the threshold ',
      code(`${threshold}`),
      ' for numerical stability.',
    ),
    formula: { display: `out = (1 / ${beta}) * log(1 + exp(${beta} * x))` },
    shapeSteps: [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape ?? inputShape), reason: noChange }],
  }
}

// ============================================================================
// Regularization Layers
// ============================================================================

function explainDropout(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const p = Number(node.attrs?.p ?? 0.5)
  const training = node.attrs?.training ?? true
  const pct = Math.round(p * 100)

  return {
    title: 'Dropout',
    short: training
      ? rich(`Randomly zeroes ~${pct}% of values during training`)
      : rich('Inactive in eval mode'),
    description: training
      ? rich(
          code('Dropout'),
          ' independently zeroes each element with probability ',
          code(`p=${p}`),
          ', then scales the remaining values by ',
          code(`1 / (1 - p)`),
          ' so the expected sum is preserved. Active only during training.',
        )
      : rich(
          code('Dropout'),
          ' is a no-op outside of training mode (',
          code('model.eval()'),
          '). The input passes through unchanged. With ',
          code(`p=${p}`),
          ', it would zero elements with that probability during training.',
        ),
    formula: {
      display: training ? `out = mask * x / (1 - p),  p=${p}` : 'out = x  (eval mode)',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: noChange,
      },
    ],
  }
}

function explainDropout1d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const p = Number(node.attrs?.p ?? 0.5)
  const training = node.attrs?.training ?? true
  const pct = Math.round(p * 100)

  return {
    title: 'Dropout1d',
    short: training
      ? rich(`Randomly zeroes entire channels (~${pct}% of them) along the length dimension`)
      : rich('Inactive in eval mode'),
    description: training
      ? rich(
          code('Dropout1d'),
          ' drops entire channels rather than individual elements. Every value in a channel is zeroed together with probability ',
          code(`p=${p}`),
          '. This can be more effective than element-wise dropout for 1D inputs like sequences when adjacent elements in a channel are correlated.',
        )
      : rich(
          code('Dropout1d'),
          ' is a no-op outside of training mode (',
          code('model.eval()'),
          '). With ',
          code(`p=${p}`),
          ', it would zero entire channels with that probability during training.',
        ),
    formula: {
      display: training ? `out = channel_mask * x / (1 - p),  p=${p}` : 'out = x  (eval mode)',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: noChange,
      },
    ],
  }
}

function explainDropout2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const p = Number(node.attrs?.p ?? 0.5)
  const training = node.attrs?.training ?? true
  const pct = Math.round(p * 100)

  return {
    title: 'Dropout2d',
    short: training
      ? rich(`Randomly zeroes entire 2D feature maps (~${pct}% of channels)`)
      : rich('Inactive in eval mode'),
    description: training
      ? rich(
          code('Dropout2d'),
          ' drops entire H×W feature maps rather than individual elements. Every spatial position in a channel is zeroed together with probability ',
          code(`p=${p}`),
          '. It is often used instead of element-wise ',
          code('Dropout'),
          ' for convolutional feature maps, where spatially adjacent values can be strongly correlated.',
        )
      : rich(
          code('Dropout2d'),
          ' is a no-op outside of training mode (',
          code('model.eval()'),
          '). With ',
          code(`p=${p}`),
          ', it would zero entire feature maps with that probability during training.',
        ),
    formula: {
      display: training ? `out = channel_mask * x / (1 - p),  p=${p}` : 'out = x  (eval mode)',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: noChange,
      },
    ],
  }
}

function explainDropout3d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const p = Number(node.attrs?.p ?? 0.5)
  const training = node.attrs?.training ?? true
  const pct = Math.round(p * 100)

  return {
    title: 'Dropout3d',
    short: training
      ? rich(`Randomly zeroes entire 3D feature volumes (~${pct}% of channels)`)
      : rich('Inactive in eval mode'),
    description: training
      ? rich(
          code('Dropout3d'),
          ' drops entire D×H×W feature volumes with probability ',
          code(`p=${p}`),
          '. The 3D counterpart to ',
          code('Dropout2d'),
          ', used in volumetric models such as 3D CNNs for medical imaging or video.',
        )
      : rich(
          code('Dropout3d'),
          ' is a no-op outside of training mode (',
          code('model.eval()'),
          '). With ',
          code(`p=${p}`),
          ', it would zero entire feature volumes with that probability during training.',
        ),
    formula: {
      display: training ? `out = channel_mask * x / (1 - p),  p=${p}` : 'out = x  (eval mode)',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: noChange,
      },
    ],
  }
}

function explainAlphaDropout(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const p = Number(node.attrs?.p ?? 0.5)
  const training = node.attrs?.training ?? true

  return {
    title: 'AlphaDropout',
    short: training
      ? rich(`Dropout variant that preserves self-normalizing properties of `, code('SELU'))
      : rich('Inactive in eval mode'),
    description: training
      ? rich(
          code('AlphaDropout'),
          ' is designed to pair with ',
          code('SELU'),
          ' activations. Rather than zeroing dropped elements, it replaces them with a value that keeps the output mean and variance stable, preserving the self-normalizing property that ',
          code('SELU'),
          ' relies on. Drops elements with probability ',
          code(`p=${p}`),
          '.',
        )
      : rich(
          code('AlphaDropout'),
          ' is a no-op outside of training mode (',
          code('model.eval()'),
          '). With ',
          code(`p=${p}`),
          ', it would drop elements with that probability during training.',
        ),
    formula: {
      display: training ? `out = (mask * x + alpha * (1 - mask)) * scale + bias,  p=${p}` : 'out = x  (eval mode)',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: noChange,
      },
    ],
  }
}

// ============================================================================
// Shape And Layout Transformations
// ============================================================================

function explainFlatten(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const startDim = Number(node.attrs?.start_dim ?? node.attrs?.startDim ?? 1)
  const rawEndDim = Number(node.attrs?.end_dim ?? node.attrs?.endDim ?? -1)
  const endDim = rawEndDim < 0 ? inputShape.length + rawEndDim : rawEndDim
  const flattenedDims = inputShape.slice(startDim, endDim + 1)
  const flattenedProduct = product(flattenedDims)

  return {
    title: 'Flatten',
    short: rich(
      'Flattens dims ',
      code(`${startDim}..${endDim}`),
      ' : ',
      code(`${flattenedDims.join(' x ')} = ${flattenedProduct}`),
    ),
    description: rich(
      code('Flatten'),
      ' combines a range of dimensions into one by multiplying their sizes.',
    ),
    formula: {
      display: 'out = reshape(input)',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Flattened dimensions',
        from: shapeText(flattenedDims),
        to: flattenedProduct,
        substitution: `${flattenedDims.join(' x ')} = ${flattenedProduct}`,
      },
    ],
  }

}

function explainReshape(node: TraceNodeForExplanation): Explanation | null {
const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const totalIn = product(inputShape)
  const inferredIndex = outputShape.findIndex((dim) => dim === -1)
  const resolvedOutputShape =
    inferredIndex >= 0
      ? outputShape.map((dim, i) =>
          i === inferredIndex ? totalIn / product(outputShape.filter((_, j) => j !== inferredIndex)) : dim,
        )
      : outputShape

  return {
    title: "reshape",
    short: rich("Returns the tensor with a new shape, using a view when possible and a copy when required"),
    description: rich(
      code("reshape"),
      ' changes the shape of the tensor without changing its data or total number of elements (',
      code(`${totalIn}`),
      ' elements). PyTorch returns a view when the existing storage layout permits it, otherwise it makes a copy.',
    ),
    formula: {
      display: `out = ${"reshape"}(input, shape)`,
      substitution: `${shapeText(inputShape)} (${totalIn} elements) ⟶ ${shapeText(resolvedOutputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Total elements',
        from: totalIn,
        to: totalIn,
        reason: 'Total number of elements is preserved; only the dimension layout changes.',
      },
      ...(inferredIndex >= 0
        ? [
            {
              label: `Inferred dimension ${inferredIndex}`,
              from: code(-1),
              to: resolvedOutputShape[inferredIndex],
              reason: `PyTorch infers this size from the remaining elements: ${totalIn} / ${product(
                outputShape.filter((_, i) => i !== inferredIndex),
              )} = ${resolvedOutputShape[inferredIndex]}.`,
            },
          ]
        : []),
    ],
  }
}

function explainView(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const totalIn = product(inputShape)
  const inferredIndex = outputShape.findIndex((dim) => dim === -1)
  const resolvedOutputShape =
    inferredIndex >= 0
      ? outputShape.map((dim, i) =>
          i === inferredIndex ? totalIn / product(outputShape.filter((_, j) => j !== inferredIndex)) : dim,
        )
      : outputShape

  return {
    title: "view",
    short: rich("Returns a view sharing the input data, with a different shape"),
    description: rich(
      code("view"),
      ' returns a new tensor object sharing the same underlying data with a different shape. It preserves the total number of elements (',
      code(`${totalIn}`),
      ' elements) and requires the requested shape to be compatible with the input size and strides.',
    ),
    formula: {
      display: `out = ${"view"}(input, shape)`,
      substitution: `${shapeText(inputShape)} (${totalIn} elements) ⟶ ${shapeText(resolvedOutputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Total elements',
        from: totalIn,
        to: totalIn,
        reason: 'Total number of elements is preserved; only the dimension layout changes.',
      },
      ...(inferredIndex >= 0
        ? [
            {
              label: `Inferred dimension ${inferredIndex}`,
              from: code(-1),
              to: resolvedOutputShape[inferredIndex],
              reason: `PyTorch infers this size from the remaining elements: ${totalIn} / ${product(
                outputShape.filter((_, i) => i !== inferredIndex),
              )} = ${resolvedOutputShape[inferredIndex]}.`,
            },
          ]
        : []),
    ],
  }
}

function explainPermute(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const rawDims = node.attrs?.dims ?? node.attrs?.dim
  const dims = Array.isArray(rawDims) ? rawDims.map((dim) => normalizeDim(Number(dim), inputShape.length)) : undefined

  return {
    title: 'permute',
    short: rich(
      'Reorders dimensions: ',
      code(shapeText(inputShape)),
      ' ⟶ ',
      code(shapeText(outputShape)),
      ...(dims ? [' using order ', code(`(${dims.join(', ')})`)] : [])
    ),
    description: rich(
      code('permute'),
      ' reorders the dimensions of a tensor according to the given order. The underlying data is unchanged.'
    ),
    formula: {
      display: 'out = input.permute(dims)',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: dims
      ? dims.map((sourceDim, targetDim) => ({
          label: `Dimension ${targetDim}`,
          from: `dim ${sourceDim} (size ${inputShape[sourceDim]})`,
          to: outputShape[targetDim],
          reason: `New dimension ${targetDim} comes from original dimension ${sourceDim}.`,
        }))
      : [
          {
            label: 'Shape',
            from: shapeText(inputShape),
            to: shapeText(outputShape),
          },
        ],
  }
}

function explainTranspose(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const rawDims = Array.isArray(node.attrs?.dims) ? node.attrs.dims.map(Number) : null
  const dim0 = normalizeDim(Number(node.attrs?.dim0 ?? rawDims?.[0] ?? 0), inputShape.length)
  const dim1 = normalizeDim(Number(node.attrs?.dim1 ?? rawDims?.[1] ?? 1), inputShape.length)

  return {
    title: 'transpose',
    short: rich(
      'Swaps dimensions ',
      code(`${dim0}`),
      ' and ',
      code(`${dim1}`),
      ': ',
      code(shapeText(inputShape)),
      ' ⟶ ',
      code(shapeText(outputShape))
    ),
    description: rich(
      code('transpose'),
      ' swaps two dimensions of a tensor. The underlying data is unchanged; only the strides/layout differ. Commonly used in attention as ',
      code('.transpose(-2, -1)'),
      ' to swap the sequence and head dimensions.',
    ),
    formula: {
      display: 'out = input.transpose(dim0, dim1)',
      substitution: `dim ${dim0} (size ${inputShape[dim0]}) ↔ dim ${dim1} (size ${inputShape[dim1]})`,
    },
    shapeSteps: [
      {
        label: `Dimension ${dim0}`,
        from: inputShape[dim0],
        to: outputShape[dim1],
        reason: `Moved to position ${dim1}.`,
      },
      {
        label: `Dimension ${dim1}`,
        from: inputShape[dim1],
        to: outputShape[dim0],
        reason: `Moved to position ${dim0}.`,
      },
    ],
  }
}


function explainUnsqueeze(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const dim = normalizeInsertionDim(Number(node.attrs?.dim ?? 0), inputShape.length + 1)

  return {
    title: 'unsqueeze',
    short: rich(
      'Inserts a size-',
      code('1'),
      ' dimension at position ',
      code(`${dim}`),
      ': ',
      code(shapeText(inputShape)),
      ' ⟶ ',
      code(shapeText(outputShape))
    ),
    description: rich(
      code('unsqueeze'),
      ' inserts a new dimension of size ',
      code('1'),
      ' at position ',
      code(`${dim}`),
      '. The data is unchanged; this is commonly used to add a batch or channel dimension before an operation that requires it.',
    ),
    formula: {
      display: `out = unsqueeze(input, dim=${dim})`,
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: `Inserted dimension ${dim}`,
        from: '-',
        to: 1,
        reason: `A new size-1 dimension is inserted at position ${dim}.`,
      },
      {
        label: 'Other dimensions',
        from: shapeText(inputShape),
        to: shapeText(outputShape.filter((_, i) => i !== dim)),
        reason: 'All existing dimensions are preserved and shifted accordingly.',
      },
    ],
  }
}

function explainExpand(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const expandedDims = outputShape
    .map((size, i) => ({ i, from: rightAlignedShapeValue(inputShape, outputShape.length, i), to: size }))
    .filter(({ from, to }) => from !== to)

  return {
    title: 'expand',
    short: rich(
      'Broadcasts ',
      code(shapeText(inputShape)),
      ' to ',
      code(shapeText(outputShape)),
      ' without copying data',
    ),
    description: rich(
      code('expand'),
      ' broadcasts size-',
      code('1'),
      ' dimensions to a larger size by repeating values via stride tricks, so no data is copied. Any dimension already larger than ',
      code('1'),
      ' must match the target size exactly.',
    ),
    formula: {
      display: 'out = input.expand(size)  # stride trick, no copy',
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: expandedDims.length
      ? expandedDims.map(({ i, from, to }) => ({
          label: `Dimension ${i}`,
          from,
          to,
          reason: `Size-${from} dimension broadcast to ${to}.`,
        }))
      : [{ label: 'Shape', from: shapeText(inputShape), to: shapeText(outputShape), reason: noChange }],
  }
}

function explainContiguous(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  return {
    title: 'contiguous',
    short: rich('Returns a tensor with contiguous memory layout'),
    description: rich(
      code('contiguous'),
      ' returns a tensor with the same data laid out contiguously in memory. Operations like ',
      code('permute'),
      ' and ',
      code('transpose'),
      ' only change strides without moving data; ',
      code('contiguous'),
      ' returns the original tensor if it already has the requested contiguous layout, or copies into that layout when needed. Often required before calling ',
      code('view')
    ),
    formula: {
      display: 'out = input.contiguous()',
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: 'Shape is unchanged; only the memory layout is affected.',
      },
    ],
  }
}

function explainChunk(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputs = tensorValues(node.outputs)
  if (!inputShape || !outputs.length) return null

  const chunks = Number(node.attrs?.chunks ?? outputs.length)
  const dim = normalizeDim(Number(node.attrs?.dim ?? 0), inputShape.length)
  const dimSize = inputShape[dim]

  return {
    title: 'chunk',
    short: rich(
      'Splits ',
      code(shapeText(inputShape)),
      ' into ',
      code(`${outputs.length}`),
      ' pieces along ',
      code(`dim=${dim}`)
    ),
    description: rich(
      code('chunk'),
      ' attempts to split a tensor into ',
      code(`${chunks}`),
      ' pieces along dimension ',
      code(`${dim}`),
      ` (size ${dimSize}). PyTorch may return fewer pieces than requested, and returned chunks are views. If the dimension is not evenly divisible, chunk sizes can differ. Commonly used in multi-head attention to split Q, K, and V from a single projection.`,
    ),
    formula: {
      display: `out = chunk(input, ${chunks}, dim=${dim})`,
      substitution: `dim ${dim} (size ${dimSize}) ⟶ ${outputs.map((o) => o.shape?.[dim] ?? '?').join(', ')}`,
    },
    shapeSteps: outputs.map((output, i) => ({
      label: `Chunk ${i}`,
      from: shapeText(inputShape),
      to: shapeText(output.shape),
      reason: `Slice of dim ${dim} from the input.`,
    })),
  }
}

function explainSplit(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputs = tensorValues(node.outputs)
  if (!inputShape || !outputs.length) return null

  const splitSize = node.attrs?.split_size_or_sections
  const dim = normalizeDim(Number(node.attrs?.dim ?? 0), inputShape.length)
  const dimSize = inputShape[dim]

  const splitSizeText = Array.isArray(splitSize)
    ? `[${splitSize.join(', ')}]`
    : String(splitSize ?? '?')

  return {
    title: 'split',
    short: rich(
      'Splits ',
      code(shapeText(inputShape)),
      ' into ',
      code(`${outputs.length}`),
      ' pieces of size ',
      code(splitSizeText),
      ' along ',
      code(`dim=${dim}`)
    ),
    description: rich(
      code('split'),
      ' divides a tensor along dimension ',
      code(`${dim}`),
      ` (size ${dimSize}) into chunks of size `,
      code(splitSizeText),
      '. Unlike ',
      code('chunk'),
      ', you specify the size of each piece rather than the number of pieces, and pieces can be unequal when passing a list.',
    ),
    formula: {
      display: `out = split(input, ${splitSizeText}, dim=${dim})`,
      substitution: `dim ${dim} (size ${dimSize}) ⟶ ${outputs.map((o) => o.shape?.[dim] ?? '?').join(', ')}`,
    },
    shapeSteps: outputs.map((output, i) => ({
      label: `Split ${i}`,
      from: shapeText(inputShape),
      to: shapeText(output.shape),
      reason: `Slice of dim ${dim} from the input.`,
    })),
  }
}

function explainStack(node: TraceNodeForExplanation): Explanation | null {
  const inputs = tensorValues(node.inputs)
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputs.length || !outputShape) return null

  const inputShapes = inputs.map((input) => input.shape).filter(Boolean) as number[][]
  const dim = normalizeInsertionDim(Number(node.attrs?.dim ?? 0), outputShape.length)

  return {
    title: 'stack',
    short: rich(
      'Stacks ',
      code(`${inputShapes.length}`),
      ' tensors into a new dimension at ',
      code(`dim=${dim}`),
      ': ',
      code(shapeText(outputShape))
    ),
    description: rich(
      code('stack'),
      ' joins tensors along a new dimension, unlike ',
      code('cat'),
      ' which joins along an existing one. All input tensors must have identical shapes. The output has one more dimension than the inputs.',
    ),
    formula: {
      display: 'out = stack(inputs, dim)',
      substitution: `${inputShapes.length} × ${shapeText(inputShapes[0])} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'New dimension',
        from: '-',
        to: inputShapes.length,
        reason: `A new dimension of size ${inputShapes.length} (one per input tensor) is inserted at position ${dim}.`,
      },
      {
        label: 'Input dimensions',
        from: shapeText(inputShapes[0]),
        to: shapeText(outputShape.filter((_, i) => i !== dim)),
        reason: 'All input dimensions are preserved.',
      },
    ],
  }
}

function explainUnbind(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputs = tensorValues(node.outputs)
  if (!inputShape || !outputs.length) return null

  const dim = normalizeDim(Number(node.attrs?.dim ?? 0), inputShape.length)
  const dimSize = inputShape[dim]

  return {
    title: 'unbind',
    short: rich(
      'Splits ',
      code(shapeText(inputShape)),
      ' into ',
      code(`${dimSize}`),
      ' tensors by removing ',
      code(`dim=${dim}`)
    ),
    description: rich(
      code('unbind'),
      ' is the inverse of ',
      code('stack'),
      '. It removes a dimension and returns a tuple of tensors, one per slice along that dimension. Each output has one fewer dimension than the input.',
    ),
    formula: {
      display: `out = unbind(input, dim=${dim})`,
      substitution: `${shapeText(inputShape)} ⟶ ${dimSize} × ${shapeText(outputs[0]?.shape)}`,
    },
    shapeSteps: [
      {
        label: `Removed dimension ${dim}`,
        from: dimSize,
        to: '-',
        reason: `Dimension ${dim} (size ${dimSize}) is removed; one tensor is returned per slice.`,
      },
      {
        label: 'Remaining dimensions',
        from: shapeText(inputShape.filter((_, i) => i !== dim)),
        to: shapeText(outputs[0]?.shape),
        reason: 'All other dimensions are preserved in each output tensor.',
      },
    ],
  }
}

function explainRepeat(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const repeats = node.attrs?.repeats
  const repeatsText = Array.isArray(repeats) ? `(${repeats.join(', ')})` : String(repeats ?? '?')

  return {
    title: 'repeat',
    short: rich(
      'Tiles ',
      code(shapeText(inputShape)),
      ' to ',
      code(shapeText(outputShape)),
      ' by repeating data',
    ),
    description: rich(
      code('repeat'),
      ' copies the tensor data along each dimension by the given number of times ',
      code(repeatsText),
      '. Unlike ',
      code('expand'),
      ', this always allocates new memory, so the data is physically repeated.',
    ),
    formula: {
      display: `out = input.repeat(${repeatsText})`,
      substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: outputShape.map((size, i) => ({
      label: `Dimension ${i}`,
      from: rightAlignedShapeValue(inputShape, outputShape.length, i),
      to: size,
      reason: Array.isArray(repeats)
        ? `Repeated ${repeats[i]} time${repeats[i] === 1 ? '' : 's'}.`
        : 'Repeated along this dimension.',
    })),
  }
}

function explainNarrow(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const dim = normalizeDim(Number(node.attrs?.dim ?? 0), inputShape.length)
  const start = Number(node.attrs?.start ?? 0)
  const length = Number(node.attrs?.length ?? outputShape[dim])

  return {
    title: 'narrow',
    short: rich(
      'Slices dim ',
      code(`${dim}`),
      ' from ',
      code(`${start}`),
      ' to ',
      code(`${start + length}`),
      ': ',
      code(shapeText(inputShape)),
      ' ⟶ ',
      code(shapeText(outputShape))
    ),
    description: rich(
      code('narrow'),
      ' returns a view over a consecutive range along one dimension. It is equivalent to slicing one dimension with a bounded range and preserves rank, so the sliced dimension shrinks rather than disappearing.',
    ),
    formula: {
      display: `out = narrow(input, dim=${dim}, start=${start}, length=${length})`,
      substitution: `dim ${dim}: ${inputShape[dim]} ⟶ ${length}  (indices ${start}..${start + length - 1})`,
    },
    shapeSteps: [
      {
        label: `Dimension ${dim}`,
        from: inputShape[dim],
        to: length,
        reason: `Sliced from index ${start} to ${start + length - 1}.`,
      },
      {
        label: 'Other dimensions',
        from: shapeText(inputShape.filter((_, i) => i !== dim)),
        to: shapeText(outputShape.filter((_, i) => i !== dim)),
        reason: 'All other dimensions are preserved.',
      },
    ],
  }
}

function explainRoll(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const shifts = node.attrs?.shifts
  const dims = node.attrs?.dims
  const shiftsText = Array.isArray(shifts) ? `(${shifts.join(', ')})` : String(shifts ?? '?')
  const dimsText = Array.isArray(dims) ? `(${dims.join(', ')})` : String(dims ?? '?')

  return {
    title: 'roll',
    short: rich(
      'Shifts elements by ',
      code(shiftsText),
      ' along ',
      code(`dims=${dimsText}`),
      ' with wraparound',
    ),
    description: rich(
      code('roll'),
      ' cyclically shifts elements along the given dimensions, and elements that fall off one end wrap around to the other. Shape never changes. Used in ',
      code('Swin Transformer'),
      ' to implement cyclic window shifting.',
    ),
    formula: {
      display: `out = roll(input, shifts=${shiftsText}, dims=${dimsText})`,
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: 'roll only moves elements; it never changes the tensor shape.',
      },
    ],
  }
}

function explainFlip(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const dims = node.attrs?.dims
  const dimsText = Array.isArray(dims) ? `(${dims.join(', ')})` : String(dims ?? '?')

  return {
    title: 'flip',
    short: rich(
      'Reverses elements along ',
      code(`dims=${dimsText}`)
    ),
    description: rich(
      code('flip'),
      ' reverses the order of elements along one or more dimensions. Unlike NumPy\'s ',
      code('flip'),
      ', PyTorch\'s version always returns a copy rather than a view.',
    ),
    formula: {
      display: `out = flip(input, dims=${dimsText})`,
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape ?? inputShape),
        reason: 'flip only reverses element order; it never changes the tensor shape.',
      },
    ],
  }
}

// ============================================================================
// Tensor Combination And Arithmetic
// ============================================================================

function explainCat(node: TraceNodeForExplanation): Explanation | null {
  const inputs = tensorValues(node.inputs)
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputs.length || !outputShape) return null

  const inputShapes = inputs.map((input) => input.shape).filter(Boolean) as number[][]
  const dim = normalizeDim(Number(node.attrs?.dim ?? inputs.find((input) => input.role === 'dim')?.value ?? 0), inputShapes[0]?.length ?? outputShape.length)

  return {
    title: 'Concatenate',
    short: rich(
      `Concatenates ${inputShapes.length} tensors along `,
      code(`dim=${dim}`)
    ),
    description: rich(
      code('cat'),
      ' joins tensors along one dimension. All other dimensions must match exactly and are preserved.',
    ),
    formula: {
      display: 'out = concat(inputs, dim)',
      substitution: `${inputShapes.map(shapeText).join(' + ')} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: `Concatenated dimension ${dim}`,
        from: inputShapes.map((shape) => shape[dim]).join(' + '),
        to: outputShape[dim],
        reason: 'The selected dimension is summed across inputs.',
      },
      {
        label: 'Other dimensions',
        from: shapeText(inputShapes[0]?.map((value, index) => (index === dim ? -1 : value))),
        to: shapeText(outputShape.map((value, index) => (index === dim ? -1 : value))),
        reason: 'Non-concatenated dimensions are preserved.',
      },
    ],
  }
}

function explainAdd(node: TraceNodeForExplanation): Explanation | null {
  const inputs = tensorValues(node.inputs)
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (inputs.length < 2 || !outputShape) return null

  const shapeA = inputs[0]?.shape
  const shapeB = inputs[1]?.shape
  if (!shapeA || !shapeB) return null

  const broadcasts = shapeA.length !== shapeB.length || shapeA.some((val, idx) => val !== shapeB[idx]);
  return {
    title: 'add',
    short: rich(
      'Adds ',
      code(shapeText(shapeA)),
      ' and ',
      code(shapeText(shapeB)),
      ' element-wise',
      ...(broadcasts ? [' with broadcasting'] : [])
    ),
    description: rich(
      code('add'),
      ' performs element-wise addition. When shapes differ, dimensions of size ',
      code('1'),
      ' are broadcast to match the other tensor.',
    ),
    formula: {
      display: 'out = a + b',
      substitution: `${shapeText(shapeA)} + ${shapeText(shapeB)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Result shape',
        from: `${shapeText(shapeA)}, ${shapeText(shapeB)}`,
        to: shapeText(outputShape),
        reason: broadcasts
          ? 'Dimensions of size 1 (or missing leading dimensions) are broadcast to match the larger shape.'
          : 'Shapes already match; no broadcasting needed.',
      },
    ],
  }
}

// ============================================================================
// Padding And Boundary Operations
// ============================================================================

function explainPad(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const kwargs = node.attrs?.kwargs as Record<string, unknown> | undefined
  const rawPad = node.attrs?.pad ?? node.attrs?.padding ?? kwargs?.pad
  const parsedPad = Array.isArray(rawPad) ? rawPad.map(Number) : null
  const pad = parsedPad?.every(Number.isFinite) ? parsedPad : null
  const mode = typeof node.attrs?.mode === 'string' ? node.attrs.mode : typeof kwargs?.mode === 'string' ? kwargs.mode : null
  const fillValue = node.attrs?.value ?? kwargs?.value

  // pad list applies from the last dimension backwards: [left_-1, right_-1, left_-2, right_-2, ...]
  const numPaddedDims = pad ? Math.floor(pad.length / 2) : 0
  const dimSteps = []
  for (let i = 0; i < numPaddedDims; i++) {
    const dimIndex = inputShape.length - 1 - i
    const before = pad?.[i * 2] ?? 0
    const after = pad?.[i * 2 + 1] ?? 0
    dimSteps.push({
      label: `Dimension ${dimIndex}`,
      from: inputShape[dimIndex],
      to: outputShape[dimIndex],
      substitution: `${inputShape[dimIndex]} + ${before} (before) + ${after} (after) = ${outputShape[dimIndex]}`,
    })
  }
  dimSteps.reverse()

  const modePart = code(`mode=${mode ?? 'unknown'}`)
  const valuePart = code(`value=${String(fillValue ?? 'unknown')}`)

  const noPadDescription =
    mode === 'constant'
      ? rich(code('pad'), ' was called with an empty pad list, so the shape is unchanged. ', modePart, ' and ', valuePart, ' are present, but no dimensions are padded.')
      : mode
        ? rich(code('pad'), ' was called with an empty pad list, so the shape is unchanged. ', modePart, ' is present, but no dimensions are padded.')
        : rich(code('pad'), ' was called with an empty pad list, so the shape is unchanged.')

  const ambiguousDescription = rich(
    code('pad'),
    ' changes tensor boundaries, but this trace does not identify normalized pad, mode, and value metadata clearly enough to describe the exact padding arguments.',
  )

  return {
    title: 'pad',
    short: mode
      ? rich(`Pads ${shapeText(inputShape)} to ${shapeText(outputShape)} using `, modePart)
      : rich(`Pads ${shapeText(inputShape)} to ${shapeText(outputShape)}`),
    description:
      !pad
        ? ambiguousDescription
        : numPaddedDims > 0
        ? rich(
            code('pad'),
            ` adds values around the edges of the last ${numPaddedDims} dimension${numPaddedDims === 1 ? '' : 's'}. `,
            ...(mode ? [modePart, mode === 'constant' ? ' fills with ' : ` fills using ${mode} values from the existing tensor.`] : ['The fill mode is not recorded.']),
            ...(mode === 'constant' && fillValue !== undefined ? [valuePart, '.'] : []),
            ' Dimensions not covered by the pad list are left unchanged.',
          )
        : noPadDescription,
    formula: {
      display: 'out = F.pad(input, pad, mode, value)',
      substitution: pad ? `pad=(${pad.join(', ')}): ${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}` : undefined,
    },
    shapeSteps: dimSteps.length
      ? dimSteps
      : [
          {
            label: 'Shape',
            from: shapeText(inputShape),
            to: shapeText(outputShape),
            reason: pad ? undefined : 'Exact pad amounts are not available from normalized metadata.',
          },
        ],
  }
}

// ============================================================================
// Fallback Explanation
// ============================================================================

function explainGeneric(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape && !node.formula) return null

  return {
    title: node.label,
    short: rich("Unknown operation"),
    description: rich(
      code(node.label),
      ' transforms the tensor according to the traced pytorch metadata',
    ),
    formula: node.formula
      ? {
          display: node.formula,
          substitution: `${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
        }
      : undefined,
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape),
      },
    ],
  }
}

// ============================================================================
// Explanation Registry
// ============================================================================

type ExplanationMatcher = {
  labels: string[]
  explain: (node: TraceNodeForExplanation) => Explanation | null
}

const learnedLayerExplanations: ExplanationMatcher[] = [
  { labels: ['Linear'], explain: explainLinear },
  { labels: ['Embedding'], explain: explainEmbedding },
]

const convolutionExplanations: ExplanationMatcher[] = [
  { labels: ['Conv1d'], explain: explainConv1d },
  { labels: ['Conv2d'], explain: explainConv2d },
  { labels: ['Conv3d'], explain: explainConv3d },
]

const poolingExplanations: ExplanationMatcher[] = [
  { labels: ['MaxPool1d'], explain: explainMaxPool1d },
  { labels: ['MaxPool2d'], explain: explainMaxPool2d },
  { labels: ['MaxPool3d'], explain: explainMaxPool3d },
  { labels: ['AvgPool1d'], explain: explainAvgPool1d },
  { labels: ['AvgPool2d'], explain: explainAvgPool2d },
  { labels: ['AvgPool3d'], explain: explainAvgPool3d },
  { labels: ['AdaptiveAvgPool1d'], explain: explainAdaptiveAvgPool1d },
  { labels: ['AdaptiveAvgPool2d'], explain: explainAdaptiveAvgPool2d },
  { labels: ['AdaptiveAvgPool3d'], explain: explainAdaptiveAvgPool3d },
]

const normalizationExplanations: ExplanationMatcher[] = [
  { labels: ['BatchNorm1d'], explain: explainBatchNorm1d },
  { labels: ['BatchNorm2d'], explain: explainBatchNorm2d },
  { labels: ['BatchNorm3d'], explain: explainBatchNorm3d },
  { labels: ['LayerNorm'], explain: explainLayerNorm },
  { labels: ['RMSNorm'], explain: explainRMSNorm },
  { labels: ['GroupNorm'], explain: explainGroupNorm },
  { labels: ['InstanceNorm1d'], explain: explainInstanceNorm1d },
  { labels: ['InstanceNorm2d'], explain: explainInstanceNorm2d },
  { labels: ['InstanceNorm3d'], explain: explainInstanceNorm3d },
]

const activationExplanations: ExplanationMatcher[] = [
  { labels: ['ReLU', 'relu'], explain: explainReLU },
  { labels: ['LeakyReLU', 'leaky_relu'], explain: explainLeakyReLU },
  { labels: ['ELU', 'elu'], explain: explainELU },
  { labels: ['SELU', 'selu'], explain: explainSELU },
  { labels: ['Sigmoid', 'sigmoid'], explain: explainSigmoid },
  { labels: ['Hardsigmoid', 'hardsigmoid'], explain: explainHardsigmoid },
  { labels: ['Tanh', 'tanh'], explain: explainTanh },
  { labels: ['SiLU', 'silu'], explain: explainSiLU },
  { labels: ['Mish', 'mish'], explain: explainMish },
  { labels: ['Hardswish', 'hardswish'], explain: explainHardswish },
  { labels: ['Softmax', 'softmax'], explain: explainSoftmax },
  { labels: ['LogSoftmax', 'log_softmax'], explain: explainLogSoftmax },
  { labels: ['GELU', 'gelu'], explain: explainGELU },
  { labels: ['Softplus', 'softplus'], explain: explainSoftplus },
]

const regularizationExplanations: ExplanationMatcher[] = [
  { labels: ['Dropout', 'dropout'], explain: explainDropout },
  { labels: ['Dropout1d'], explain: explainDropout1d },
  { labels: ['Dropout2d'], explain: explainDropout2d },
  { labels: ['Dropout3d'], explain: explainDropout3d },
  { labels: ['AlphaDropout'], explain: explainAlphaDropout },
]

const shapeTransformExplanations: ExplanationMatcher[] = [
  { labels: ['Flatten', 'flatten'], explain: explainFlatten },
  { labels: ['reshape'], explain: explainReshape },
  { labels: ['view'], explain: explainView },
  { labels: ['permute'], explain: explainPermute },
  { labels: ['transpose'], explain: explainTranspose },
  { labels: ['unsqueeze'], explain: explainUnsqueeze },
  { labels: ['expand'], explain: explainExpand },
  { labels: ['contiguous'], explain: explainContiguous },
  { labels: ['chunk'], explain: explainChunk },
  { labels: ['split'], explain: explainSplit },
  { labels: ['stack'], explain: explainStack },
  { labels: ['unbind'], explain: explainUnbind },
  { labels: ['repeat'], explain: explainRepeat },
  { labels: ['narrow'], explain: explainNarrow },
  { labels: ['roll'], explain: explainRoll },
  { labels: ['flip'], explain: explainFlip },
]

const tensorOperationExplanations: ExplanationMatcher[] = [
  { labels: ['cat', 'concat'], explain: explainCat },
  { labels: ['add', 'Add'], explain: explainAdd },
]

const paddingExplanations: ExplanationMatcher[] = [
  { labels: ['pad'], explain: explainPad },
]

const explanationGroups: ExplanationMatcher[][] = [
  learnedLayerExplanations,
  convolutionExplanations,
  poolingExplanations,
  normalizationExplanations,
  activationExplanations,
  regularizationExplanations,
  shapeTransformExplanations,
  tensorOperationExplanations,
  paddingExplanations,
]

export function explainNode(node: TraceNodeForExplanation): Explanation | null {
  for (const group of explanationGroups) {
    const match = group.find((item) => item.labels.includes(node.label))
    if (match) return match.explain(node)
  }
  return explainGeneric(node)
}
