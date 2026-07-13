export type UserTensorInputSpec = {
  kind: 'tensor'
  parameter_name: string
  shape: number[]
  dtype: 'float32'
  generator: 'random_normal'
}

export { FLOAT32_BYTES, MAX_TENSOR_DIMENSIONS, MAX_TENSOR_ELEMENTS, MAX_TOTAL_INPUT_BYTES }

export type UserTraceBridgeRequest = {
  run_id: string
  source: {
    selection_id: string
    class_name: string
    content_sha256: string
  }
  constructor: {
    args: SerializableLiteral[]
    kwargs: Record<string, SerializableLiteral>
  }
  inputs: UserTensorInputSpec[]
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
  MAX_TENSOR_DIMENSIONS,
  MAX_TENSOR_ELEMENTS,
  MAX_TOTAL_INPUT_BYTES,
} from '../userTrace/constants.ts'
import type { SerializableLiteral } from './sourceInspectionApi.ts'
