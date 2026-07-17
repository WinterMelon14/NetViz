export type TensorSummary = {
  numel?: number
  min?: number
  max?: number
  mean?: number
  std?: number
  zeros_pct?: number
  has_nan?: boolean
  has_inf?: boolean
}

export type TensorValue = {
  index: number
  role: string
  shape?: number[]
  dtype?: string
  preview?: number[]
  summary?: TensorSummary
  memory?: {
    num_bytes?: number
    human?: string
  }
  from?: string
  source_output?: number
  value?: unknown
}

export type ParamsInfo = {
  count?: number
  shapes?: Record<string, number[]>
  dtypes?: Record<string, string>
  memory?: {
    num_bytes?: number
    human?: string
  }
}

export type TraceNode = {
  id: string
  kind: string
  label: string
  fx_op: string
  target: string
  inputs: TensorValue[]
  outputs: TensorValue[]
  module?: {
    path?: string
    type?: string
    is_reused?: boolean
    reuse_count?: number
  }
  params?: ParamsInfo
  attrs?: Record<string, unknown>
  formula?: string
  profile?: {
    sample_count: number
    median_ms: number
    percentiles_ms: Record<string, number>
  }
}

export type TraceEdge = {
  id: string
  source: string
  target: string
  source_output: number
  target_input: number
}

export type TraceStats = {
  total_nodes?: number
  total_edges?: number
  total_params?: number
  trainable_params?: number
  non_trainable_params?: number
  total_param_memory?: { human?: string }
  total_activation_memory?: { human?: string }
  input_specs?: {
    index: number
    name?: string
    shape?: number[]
    dtype?: string
    memory?: { human?: string }
  }[]
}

export type TracePayload = {
  model_name: string
  stats?: TraceStats
  graph: {
    nodes: TraceNode[]
    edges: TraceEdge[]
  }
  profiling?: CPUProfilingResult
}

export type CPUNodeTiming = {
  node_id: string
  label: string
  kind: string
  target: string
  module_path?: string | null
  sample_count: number
  median_ms: number | null
  percentiles_ms: Record<string, number>
}

export type CPUProfilingResult = {
  schemaVersion: 1
  mode: 'cpu'
  config: {
    warmup_runs: number
    measurement_runs: number
    percentiles: number[]
  }
  environment: {
    timer: string
    python: string
    torch: string
    device: 'cpu'
  }
  semantics: {
    duration: string
    aggregation: string
    repeated_execution: number
  }
  total_profiled_ms: number
  nodes: CPUNodeTiming[]
  expensive_operations: CPUNodeTiming[]
  critical_path: {
    node_ids: string[]
    total_ms: number
    weight: 'median_ms'
    missing_timing_nodes: string[]
  }
}
