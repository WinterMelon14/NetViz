export type UserTensorInputSpec = {
  kind: 'tensor'
  shape: number[]
  dtype: 'float32'
  generator: 'random_normal'
}

export const MAX_TENSOR_DIMENSIONS = 8
export const MAX_TENSOR_ELEMENTS = 16_777_216
export const MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
export const FLOAT32_BYTES = 4

export type UserTraceBridgeRequest = {
  run_id: string
  source: {
    selection_id: string
    class_name: string
  }
  constructor: {
    args: []
    kwargs: Record<string, never>
  }
  inputs: [UserTensorInputSpec]
}

export type UserTraceWorkerRequest = Omit<UserTraceBridgeRequest, 'source'> & {
  protocol_version: 1
  output_path: string
  source: {
    file_path: string
    class_name: string
  }
}
