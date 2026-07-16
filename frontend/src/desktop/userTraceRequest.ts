import type { SerializableLiteral } from './sourceInspectionApi.ts'
import {
  FLOAT32_BYTES,
  INT64_BYTES,
  MAX_TENSOR_DIMENSIONS,
  MAX_TENSOR_ELEMENTS,
  MAX_TOTAL_INPUT_BYTES,
} from '../userTrace/constants.ts'

export type UserTensorInputSpec = {
  kind: 'tensor'
  shape: number[]
  dtype: 'float32' | 'int64'
  generator: 'random_normal' | 'random_integer'
  integer_max_exclusive?: number
}

export type StructuredInputSpec =
  | UserTensorInputSpec
  | { kind: 'none' }
  | { kind: 'boolean'; value: boolean }
  | { kind: 'integer'; value: number }
  | { kind: 'float'; value: number }
  | { kind: 'string'; value: string }
  | { kind: 'list' | 'tuple'; items: StructuredInputSpec[] }
  | { kind: 'dict'; entries: Array<{ key: string; value: StructuredInputSpec }> }

type TraceRequestBase = {
  run_id: string
  source: {
    source_id: string
    class_name: string
    content_sha256: string
  }
  constructor: {
    args: SerializableLiteral[]
    kwargs: Record<string, SerializableLiteral>
  }
  input_provider: { function_name: 'netviz_example_inputs'; parameter_names: string[] } | null
}

export type UserTraceBridgeRequest = TraceRequestBase & {
  input_schema_version: 2
  args: StructuredInputSpec[]
  kwargs: Record<string, StructuredInputSpec>
}

export type LegacyUserTraceBridgeRequest = TraceRequestBase & {
  input_schema_version?: 1
  inputs: Array<UserTensorInputSpec & { parameter_name: string }>
}

export type UserTraceWorkerRequest = {
  protocol_version: 1
  input_schema_version?: 1 | 2
  run_id: string
  output_path: string
  source: {
    file_path: string
    class_name: string
    content_sha256: string
  }
  constructor: TraceRequestBase['constructor']
  input_provider: TraceRequestBase['input_provider']
  inputs?: LegacyUserTraceBridgeRequest['inputs']
  args?: StructuredInputSpec[]
  kwargs?: Record<string, StructuredInputSpec>
}

export { FLOAT32_BYTES, INT64_BYTES, MAX_TENSOR_DIMENSIONS, MAX_TENSOR_ELEMENTS, MAX_TOTAL_INPUT_BYTES }
