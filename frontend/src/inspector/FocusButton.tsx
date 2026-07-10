export function FocusButton({ nodeId, onFocusNode }: { nodeId: string; onFocusNode: (nodeId: string) => void }) {
  return (
    <button
      type="button"
      className="focus-chip"
      onClick={(event) => {
        event.stopPropagation()
        onFocusNode(nodeId)
      }}
    >
      {nodeId}
    </button>
  )
}
