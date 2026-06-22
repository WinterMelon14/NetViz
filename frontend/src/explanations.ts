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
  from_node?: string
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

export type Explanation = {
  title: string
  short: string
  description: string
  formula?: {
    display: string
    substitution?: string
  }
  shapeSteps: ShapeStep[]
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
    short: `Maps the last dimension from ${inFeatures} to ${outFeatures}.`,
    description: `Linear applies a learned matrix multiplication to the last dimension of ${shapeText(inputShape)}. Leading batch dimensions are preserved.`,
    formula: {
      display: node.formula ?? 'y = xW^T + b',
      substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
    },
    shapeSteps: [
      {
        label: 'Batch dimensions',
        from: shapeText(batchIn),
        to: shapeText(batchOut),
        reason: 'Leading dimensions are preserved.',
      },
      {
        label: 'Feature dimension',
        from: inFeatures,
        to: outFeatures,
        reason: `The layer maps in_features=${inFeatures} to out_features=${outFeatures}.`,
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
    short: `Flattens dims ${startDim}..${endDim}: ${flattenedDims.join(' x ')} = ${flattenedProduct}.`,
    description: 'Flatten combines a range of dimensions into one dimension by multiplying their sizes.',
    formula: {
      display: 'out = reshape(input)',
      substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
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
    short: `Pools H,W from ${h}x${w} to ${hOut}x${wOut} using kernel=${kh}x${kw}, stride=${sh}x${sw}.`,
    description: 'MaxPool2d slides a 2D window over height and width and keeps the maximum value in each window. Batch and channel dimensions are preserved.',
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
    short: `Maps channels ${c} -> ${c2} and spatial size ${h}x${w} -> ${hOut}x${wOut}.`,
    description: 'Conv2d applies learned filters across height and width. Batch is preserved, channels become out_channels, and spatial dimensions depend on kernel, stride, padding, and dilation.',
    formula: {
      display: 'out = floor((in + 2p - d(k - 1) - 1) / s) + 1',
      substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
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
    short: `Concatenates ${inputShapes.length} tensors along dim=${dim}.`,
    description: 'Concatenate joins tensors along one dimension. Other dimensions must match and are preserved.',
    formula: {
      display: 'out = concat(inputs, dim)',
      substitution: `${inputShapes.map(shapeText).join(' + ')} -> ${shapeText(outputShape)}`,
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

  const broadcasts =
    shapeA.length !== shapeB.length ||
    shapeA.some((value, index) => value !== shapeB[index] && value !== 1 && shapeB[index] !== 1)

  return {
    title: 'add',
    short: `Adds ${shapeText(shapeA)} and ${shapeText(shapeB)} element-wise${broadcasts ? ' with broadcasting' : ''}.`,
    description: 'add performs element-wise addition. When shapes differ, dimensions of size 1 are broadcast to match the other tensor.',
    formula: {
      display: 'out = a + b',
      substitution: `${shapeText(shapeA)} + ${shapeText(shapeB)} -> ${shapeText(outputShape)}`,
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

function explainReshapeLike(node: TraceNodeForExplanation, title: string): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const totalIn = product(inputShape)
  const inferredIndex = outputShape.findIndex((dim) => dim === -1)
  const resolvedOutputShape =
    inferredIndex >= 0
      ? outputShape.map((dim, i) => (i === inferredIndex ? totalIn / product(outputShape.filter((_, j) => j !== inferredIndex))  : dim))
      : outputShape

  return {
    title,
    short: `Reshapes ${shapeText(inputShape)} to ${shapeText(resolvedOutputShape)}.`,
    description: `${title} changes the shape of the tensor without changing its data or total number of elements (${totalIn} elements).`,
    formula: {
      display: `out = ${title}(input, shape)`,
      substitution: `${shapeText(inputShape)} (${totalIn} elements) -> ${shapeText(resolvedOutputShape)}`,
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
              from: '-1',
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

function explainReshape(node: TraceNodeForExplanation): Explanation | null {
  return explainReshapeLike(node, 'reshape')
}

function explainView(node: TraceNodeForExplanation): Explanation | null {
  return explainReshapeLike(node, 'view')
}

function explainPermute(node: TraceNodeForExplanation): Explanation | null {
  const inputShape = tensorValues(node.inputs)[0]?.shape
  const outputShape = tensorValues(node.outputs)[0]?.shape
  if (!inputShape || !outputShape) return null

  const rawDims = node.attrs?.dims ?? node.attrs?.dim
  const dims = Array.isArray(rawDims) ? rawDims.map(Number) : undefined

  return {
    title: 'permute',
    short: `Reorders dimensions: ${shapeText(inputShape)} -> ${shapeText(outputShape)}${
      dims ? ` using order (${dims.join(', ')})` : ''
    }.`,
    description: 'permute reorders the dimensions of a tensor according to the given order. The underlying data is unchanged; only the strides/layout differ.',
    formula: {
      display: 'out = input.permute(dims)',
      substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
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

  return {
    title: 'Embedding',
    short: `Looks up a length-${embeddingDim} vector for each index in ${shapeText(inputShape)}.`,
    description: `Embedding replaces each integer index with a learned vector of length ${embeddingDim}${
      numEmbeddings ? ` from a table of ${numEmbeddings} embeddings` : ''
    }. A new trailing dimension is appended for the embedding vector; index dimensions are preserved.`,
    formula: {
      display: 'out = weight[input]',
      substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
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
        reason: `A new trailing dimension of size embedding_dim=${embeddingDim} is appended.`,
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
    short: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
    description: `${node.label} transforms tensor values according to the traced PyTorch operation metadata.`,
    formula: node.formula
      ? {
          display: node.formula,
          substitution: `${shapeText(inputShape)} -> ${shapeText(outputShape)}`,
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
  return explainGeneric(node)
}