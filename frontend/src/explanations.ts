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

function explainLinear(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const inFeatures = inputShape.at(-1)
  const outFeatures = outputShape.at(-1)
  const batchIn = inputShape.slice(0, -1)
  const batchOut = outputShape.slice(0, -1)

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
      code('y = xWᵀ + b'),
      ') to the last dimension of ',
      code(shapeText(inputShape)),
      '. Leading batch dimensions are preserved.',
    ),
    formula: {
      display: node.formula ?? 'y = xW^T + b',
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
      code(`stride=${sh}x${sw}`),
    ),
    description: rich(
      code('MaxPool2d'),
      ' slides a 2D window over height and width and keeps the maximum value in each window. Batch and channel dimensions are preserved.',
    ),
    formula: {
      display: 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: `H: floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}`,
    },
    shapeSteps: [
      { label: 'Batch', from: n, to: n2, reason: 'Batch dimension is preserved.' },
      { label: 'Channels', from: c, to: c2, reason: 'Pooling operates independently per channel.' },
      { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` },
      { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` },
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
  const [ph, pw] = numberPair(node.attrs?.padding, [0, 0])
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
      { label: 'Height', from: h, to: hOut, substitution: `floor((${h} + 2*${ph} - ${dh}*(${kh} - 1) - 1) / ${sh}) + 1 = ${hOut}` },
      { label: 'Width', from: w, to: wOut, substitution: `floor((${w} + 2*${pw} - ${dw}*(${kw} - 1) - 1) / ${sw}) + 1 = ${wOut}` },
    ],
  }
}

function explainCat(node: TraceNodeForExplanation): Explanation | null {
  const inputs = tensorValues(node.inputs)
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputs.length || !outputShape) return null

  const dim = Number(node.attrs?.dim ?? inputs.find((input) => input.role === 'dim')?.value ?? 0)
  const inputShapes = inputs.map((input) => input.shape).filter(Boolean) as number[][]

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
      ...(broadcasts ? [' with broadcasting'] : []),
      '.',
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
    short: rich("Returns a copy of the original tensor with a new shape"),
    description: rich(
      code("reshape"),
      ' changes the shape of the tensor without changing its data or total number of elements (',
      code(`${totalIn}`),
      ' elements).',
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
    short: rich("Returns the original tensor with a new shape"),
    description: rich(
      code("view"),
      ' changes the shape of the tensor without changing its data or total number of elements (',
      code(`${totalIn}`),
      ' elements).',
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
  const dims = Array.isArray(rawDims) ? rawDims.map(Number) : undefined

  return {
    title: 'permute',
    short: rich(
      'Reorders dimensions: ',
      code(shapeText(inputShape)),
      ' ⟶ ',
      code(shapeText(outputShape)),
      ...(dims ? [' using order ', code(`(${dims.join(', ')})`)] : []),
      '.',
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


function explainLayerNorm(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const rawNormalizedShape = node.attrs?.normalized_shape
  const normDims = Array.isArray(rawNormalizedShape) ? rawNormalizedShape.map(Number) : inputShape.slice(-1)
  const eps = Number(node.attrs?.eps ?? 1e-5)

  return {
    title: 'LayerNorm',
    short: rich(
      `Normalizes over the trailing dimension`
    ),
    description: rich(
      code('LayerNorm'),
      ` normalizes each sample independently across the trailing dimension `,
      code(shapeText(normDims)),
      ' by subtracting the mean and dividing by the standard deviation (',
      code(`eps=${eps}`),
      '). It then applies a learned per-element ',
      code('weight'),
      ' and ',
      code('bias'),
      '. Other dimensions are normalized independently.',
    ),
    formula: {
      display: 'out = (x - mean) / sqrt(var + eps) * weight + bias',
      substitution: `mean/var computed over the last ${normDims.length} dim(s): ${shapeText(normDims)}`,
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape),
        reason: noChange,
      },
    ],
  }
}

function explainReLU(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape) return null

  const inplace = node.attrs?.inplace ?? false

  return {
    title: 'ReLU',
    short: rich(
      'Clamps all negative values to ',
      code('0'),
      ' element-wise',
    ),
    description: rich(
      code('ReLU'),
      ' applies ',
      code('max(0, x)'),
      ' to every element. Negative values become ',
      code('0'),
      ', positive values pass through unchanged.',
      ...(inplace ? [' Operates ', code('in-place'), ' on the input tensor.'] : []),
    ),
    formula: {
      display: 'out = max(0, x)',
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
      " smoothly scales each input value by the standard normal CDF at that value, behaving like a smoother version of ",
      code('ReLU'),
      '.',
      ...approximateSuffix,
    ),
    formula: {
      display: isTanh
        ? 'out ≈ 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))'
        : 'out = x * Phi(x)',
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

function explainBatchNorm2d(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape || inputShape.length < 4) return null

  const [n, c, h, w] = inputShape
  const eps = Number(node.attrs?.eps ?? 1e-5)
  const training = node.attrs?.training ?? false

  const statSource: Array<string | TextPart> = training
    ? ['statistics computed from the current batch']
    : ['stored ', code('running_mean'), ' and ', code('running_var'), ' from training']

  return {
    title: 'BatchNorm2d',
    short: rich(
      `Normalizes each of the ${c} channels independently across batch and spatial dims; shape is unchanged.`,
    ),
    description: rich(
      code('BatchNorm2d'),
      ' normalizes each channel using ',
      ...statSource,
      '. It subtracts the mean and divides by the standard deviation (',
      code(`eps=${eps}`),
      `) across N=${n}, H=${h}, W=${w} for each channel, then applies a learned per-channel `,
      code('weight'),
      ' and ',
      code('bias'),
      '. The shape never changes.',
    ),
    formula: {
      display: 'out = (x - running_mean) / sqrt(running_var + eps) * weight + bias',
      substitution: `stats computed per-channel over N=${n}, H=${h}, W=${w}`,
    },
    shapeSteps: [
      {
        label: 'Shape',
        from: shapeText(inputShape),
        to: shapeText(outputShape),
        reason: noChange,
      },
    ],
  }
}

function explainPad(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const allInputs = node.inputs as Array<{ role: string; value?: unknown; shape?: number[] }>

  // Pad amounts are positional scalar inputs after the tensor
  const pad = allInputs
    .filter((input) => input.role === 'scalar')
    .map((input) => Number(input.value))

  // Mode and fill value come from constant inputs, falling back to attrs.kwargs
  const constants = allInputs.filter((input) => input.role === 'constant')
  const mode = String(constants[0]?.value ?? (node.attrs?.kwargs as Record<string, unknown>)?.mode ?? 'constant')
  const fillValue = constants[1]?.value ?? (node.attrs?.kwargs as Record<string, unknown>)?.value ?? 0

  // pad list applies from the last dimension backwards: [left_-1, right_-1, left_-2, right_-2, ...]
  const numPaddedDims = Math.floor(pad.length / 2)
  const dimSteps = []
  for (let i = 0; i < numPaddedDims; i++) {
    const dimIndex = inputShape.length - 1 - i
    const before = pad[i * 2]
    const after = pad[i * 2 + 1]
    dimSteps.push({
      label: `Dimension ${dimIndex}`,
      from: inputShape[dimIndex],
      to: outputShape[dimIndex],
      substitution: `${inputShape[dimIndex]} + ${before} (before) + ${after} (after) = ${outputShape[dimIndex]}`,
    })
  }
  dimSteps.reverse()

  const modePart = code(`mode=${mode}`)
  const valuePart = code(`value=${String(fillValue)}`)

  const noPadDescription =
    mode === 'constant'
      ? rich(code('pad'), ' was called with an empty pad list, so the shape is unchanged. ', modePart, ' and ', valuePart, ' are present, but no dimensions are padded.')
      : rich(code('pad'), ' was called with an empty pad list, so the shape is unchanged. ', modePart, ' is present, but no dimensions are padded.')

  return {
    title: 'pad',
    short: rich(`Pads ${shapeText(inputShape)} to ${shapeText(outputShape)} using `, modePart, '.'),
    description:
      numPaddedDims > 0
        ? rich(
            code('pad'),
            ` adds values around the edges of the last ${numPaddedDims} dimension${numPaddedDims === 1 ? '' : 's'}. `,
            modePart,
            mode === 'constant' ? ' fills with ' : ` fills using ${mode} values from the existing tensor.`,
            ...(mode === 'constant' ? [valuePart, '.'] : []),
            ' Dimensions not covered by the pad list are left unchanged.',
          )
        : noPadDescription,
    formula: {
      display: 'out = F.pad(input, pad, mode, value)',
      substitution: `pad=(${pad.join(', ')}): ${shapeText(inputShape)} ⟶ ${shapeText(outputShape)}`,
    },
    shapeSteps: dimSteps.length
      ? dimSteps
      : [
          {
            label: 'Shape',
            from: shapeText(inputShape),
            to: shapeText(outputShape),
          },
        ],
  }
}

function explainGeneric(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape && !outputShape && !node.formula) return null

  return {
    title: node.label,
    short: rich("Unknown operation"),
    description: rich(
      code(node.label),
      ' transforms the tensor according to the traced pytorch metadata.',
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

export function explainNode(node: TraceNodeForExplanation): Explanation | null {
  if (node.label === 'Linear') return explainLinear(node)
  if (node.label === 'Flatten' || node.label === 'flatten') return explainFlatten(node)
  if (node.label === 'MaxPool2d') return explainMaxPool2d(node)
  if (node.label === 'Conv2d') return explainConv2d(node)
  if (node.label === 'cat') return explainCat(node)
  if (node.label === 'add' || node.label === 'Add') return explainAdd(node)
  if (node.label === 'reshape') return explainReshape(node)
  if (node.label === 'view') return explainView(node)
  if (node.label === 'permute') return explainPermute(node)
  if (node.label === 'Embedding') return explainEmbedding(node)
  if (node.label === 'LayerNorm') return explainLayerNorm(node)
  if (node.label === 'GELU' || node.label === 'gelu') return explainGELU(node)
  if (node.label === 'ReLU' || node.label === 'relu') return explainReLU(node)
  if (node.label === 'Dropout' || node.label === 'dropout') return explainDropout(node)
  if (node.label === 'BatchNorm2d') return explainBatchNorm2d(node)
  if (node.label === 'pad') return explainPad(node)
  return explainGeneric(node)
}
