import type { TensorValue, TraceEdge, TraceNode, TracePayload } from './types.ts'

function fail(path: string, message: string): never {
  throw new Error(`${path}: ${message}`)
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(path, 'must be an object.')
  return value as Record<string, unknown>
}

function string(value: unknown, path: string) {
  if (typeof value !== 'string' || value.length === 0) fail(path, 'must be a non-empty string.')
}

function finiteNumber(value: unknown, path: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(path, 'must be a finite number.')
}

function integer(value: unknown, path: string) {
  finiteNumber(value, path)
  if (!Number.isInteger(value)) fail(path, 'must be an integer.')
}

function optionalFiniteNumber(value: unknown, path: string) {
  if (value !== undefined && value !== null) finiteNumber(value, path)
}

function numberArray(value: unknown, path: string, integersOnly = false) {
  if (!Array.isArray(value)) fail(path, 'must be an array.')
  value.forEach((item, index) => (integersOnly ? integer(item, `${path}[${index}]`) : finiteNumber(item, `${path}[${index}]`)))
}

function validateMemory(value: unknown, path: string) {
  const memory = record(value, path)
  if (memory.num_bytes !== undefined) integer(memory.num_bytes, `${path}.num_bytes`)
  if (memory.human !== undefined && typeof memory.human !== 'string') fail(`${path}.human`, 'must be a string.')
}

function validateTensorValue(value: unknown, path: string): asserts value is TensorValue {
  const tensor = record(value, path)
  integer(tensor.index, `${path}.index`)
  string(tensor.role, `${path}.role`)
  if (tensor.shape !== undefined) numberArray(tensor.shape, `${path}.shape`, true)
  if (tensor.dtype !== undefined && typeof tensor.dtype !== 'string') fail(`${path}.dtype`, 'must be a string.')
  if (tensor.preview !== undefined) numberArray(tensor.preview, `${path}.preview`)
  if (tensor.source_output !== undefined) integer(tensor.source_output, `${path}.source_output`)
  if (tensor.from !== undefined && typeof tensor.from !== 'string') fail(`${path}.from`, 'must be a string.')
  if (tensor.memory !== undefined) validateMemory(tensor.memory, `${path}.memory`)

  if (tensor.summary !== undefined) {
    const summary = record(tensor.summary, `${path}.summary`)
    optionalFiniteNumber(summary.numel, `${path}.summary.numel`)
    optionalFiniteNumber(summary.min, `${path}.summary.min`)
    optionalFiniteNumber(summary.max, `${path}.summary.max`)
    optionalFiniteNumber(summary.mean, `${path}.summary.mean`)
    optionalFiniteNumber(summary.std, `${path}.summary.std`)
    optionalFiniteNumber(summary.zeros_pct, `${path}.summary.zeros_pct`)
    if (summary.has_nan !== undefined && typeof summary.has_nan !== 'boolean') fail(`${path}.summary.has_nan`, 'must be a boolean.')
    if (summary.has_inf !== undefined && typeof summary.has_inf !== 'boolean') fail(`${path}.summary.has_inf`, 'must be a boolean.')
  }
}

function validateNode(value: unknown, path: string): asserts value is TraceNode {
  const node = record(value, path)
  for (const field of ['id', 'kind', 'label', 'fx_op', 'target'] as const) string(node[field], `${path}.${field}`)
  for (const field of ['inputs', 'outputs'] as const) {
    const values = node[field]
    if (!Array.isArray(values)) fail(`${path}.${field}`, 'must be an array.')
    values.forEach((item, index) => validateTensorValue(item, `${path}.${field}[${index}]`))
  }
}

function validateEdge(value: unknown, path: string): asserts value is TraceEdge {
  const edge = record(value, path)
  for (const field of ['id', 'source', 'target'] as const) string(edge[field], `${path}.${field}`)
  integer(edge.source_output, `${path}.source_output`)
  integer(edge.target_input, `${path}.target_input`)
}

function validatePercentileMap(value: unknown, path: string) {
  const percentiles = record(value, path)
  for (const [key, item] of Object.entries(percentiles)) {
    if (!/^[1-9][0-9]?$/.test(key)) fail(`${path}.${key}`, 'must be keyed by percentile.')
    finiteNumber(item, `${path}.${key}`)
  }
}

function validateNodeProfile(value: unknown, path: string) {
  const profile = record(value, path)
  integer(profile.sample_count, `${path}.sample_count`)
  finiteNumber(profile.median_ms, `${path}.median_ms`)
  validatePercentileMap(profile.percentiles_ms, `${path}.percentiles_ms`)
}

function validateNodeTiming(value: unknown, path: string) {
  const timing = record(value, path)
  for (const field of ['node_id', 'label', 'kind', 'target'] as const) string(timing[field], `${path}.${field}`)
  if (timing.module_path !== undefined && timing.module_path !== null && typeof timing.module_path !== 'string') fail(`${path}.module_path`, 'must be a string or null.')
  integer(timing.sample_count, `${path}.sample_count`)
  if (timing.median_ms !== null) finiteNumber(timing.median_ms, `${path}.median_ms`)
  validatePercentileMap(timing.percentiles_ms, `${path}.percentiles_ms`)
}

function validateProfiling(value: unknown, path: string, nodeIds: Set<string>) {
  const profiling = record(value, path)
  if (profiling.schemaVersion !== 1) fail(`${path}.schemaVersion`, 'must equal 1.')
  if (profiling.mode !== 'cpu') fail(`${path}.mode`, 'must equal cpu.')
  const config = record(profiling.config, `${path}.config`)
  integer(config.warmup_runs, `${path}.config.warmup_runs`)
  integer(config.measurement_runs, `${path}.config.measurement_runs`)
  numberArray(config.percentiles, `${path}.config.percentiles`, true)
  const environment = record(profiling.environment, `${path}.environment`)
  for (const field of ['timer', 'python', 'torch', 'device'] as const) string(environment[field], `${path}.environment.${field}`)
  const semantics = record(profiling.semantics, `${path}.semantics`)
  for (const field of ['duration', 'aggregation'] as const) string(semantics[field], `${path}.semantics.${field}`)
  integer(semantics.repeated_execution, `${path}.semantics.repeated_execution`)
  finiteNumber(profiling.total_profiled_ms, `${path}.total_profiled_ms`)

  for (const field of ['nodes', 'expensive_operations'] as const) {
    const timings = profiling[field]
    if (!Array.isArray(timings)) fail(`${path}.${field}`, 'must be an array.')
    timings.forEach((item, index) => {
      validateNodeTiming(item, `${path}.${field}[${index}]`)
      const timing = item as { node_id: string }
      if (!nodeIds.has(timing.node_id)) fail(`${path}.${field}[${index}].node_id`, `references missing node "${timing.node_id}".`)
    })
  }

  const criticalPath = record(profiling.critical_path, `${path}.critical_path`)
  if (!Array.isArray(criticalPath.node_ids)) fail(`${path}.critical_path.node_ids`, 'must be an array.')
  criticalPath.node_ids.forEach((nodeId, index) => {
    string(nodeId, `${path}.critical_path.node_ids[${index}]`)
    if (!nodeIds.has(nodeId)) fail(`${path}.critical_path.node_ids[${index}]`, `references missing node "${nodeId}".`)
  })
  finiteNumber(criticalPath.total_ms, `${path}.critical_path.total_ms`)
  if (criticalPath.weight !== 'median_ms') fail(`${path}.critical_path.weight`, 'must equal median_ms.')
  if (!Array.isArray(criticalPath.missing_timing_nodes)) fail(`${path}.critical_path.missing_timing_nodes`, 'must be an array.')
  criticalPath.missing_timing_nodes.forEach((nodeId, index) => string(nodeId, `${path}.critical_path.missing_timing_nodes[${index}]`))
}

export function validateTracePayload(value: unknown): TracePayload {
  const payload = record(value, 'trace')
  string(payload.model_name, 'model_name')
  const graph = record(payload.graph, 'graph')
  if (!Array.isArray(graph.nodes)) fail('graph.nodes', 'must be an array.')
  if (!Array.isArray(graph.edges)) fail('graph.edges', 'must be an array.')

  const nodeIds = new Set<string>()
  graph.nodes.forEach((node, index) => {
    const path = `graph.nodes[${index}]`
    validateNode(node, path)
    const item = node as { profile?: unknown }
    if (item.profile !== undefined) validateNodeProfile(item.profile, `${path}.profile`)
    if (nodeIds.has(node.id)) fail(`${path}.id`, `duplicates node ID "${node.id}".`)
    nodeIds.add(node.id)
  })

  const edgeIds = new Set<string>()
  graph.edges.forEach((edge, index) => {
    const path = `graph.edges[${index}]`
    validateEdge(edge, path)
    if (edgeIds.has(edge.id)) fail(`${path}.id`, `duplicates edge ID "${edge.id}".`)
    if (!nodeIds.has(edge.source)) fail(`${path}.source`, `references missing node "${edge.source}".`)
    if (!nodeIds.has(edge.target)) fail(`${path}.target`, `references missing node "${edge.target}".`)
    edgeIds.add(edge.id)
  })

  if (payload.profiling !== undefined) validateProfiling(payload.profiling, 'profiling', nodeIds)

  return value as TracePayload
}
