export type UserTensorInputSpec = {
  kind: 'tensor'
  parameter_name: string
  shape: number[]
  dtype: 'float32' | 'int64'
  generator: 'random_normal' | 'random_integer'
  integer_max_exclusive?: number
}

export { FLOAT32_BYTES, INT64_BYTES, MAX_TENSOR_DIMENSIONS, MAX_TENSOR_ELEMENTS, MAX_TOTAL_INPUT_BYTES }

export type UserTraceBridgeRequest = {
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
  inputs: UserTensorInputSpec[]
  input_provider: { function_name: 'netviz_example_inputs'; parameter_names: string[] } | null
}

export type UserTraceWorkerRequest = Omit<UserTraceBridgeRequest, 'source'> & {
  protocol_version: 1
  output_path: string
  source: {
    file_path: string
    class_name: string
    content_sha256: string
  }
}
import {
  FLOAT32_BYTES,
  INT64_BYTES,
  MAX_TENSOR_DIMENSIONS,
  MAX_TENSOR_ELEMENTS,
  MAX_TOTAL_INPUT_BYTES,
} from '../userTrace/constants.ts'
import type { SerializableLiteral } from './sourceInspectionApi.ts'
