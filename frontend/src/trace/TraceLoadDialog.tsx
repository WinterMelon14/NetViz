import { useRef } from 'react'
import type { ChangeEvent } from 'react'

export function TraceLoadDialog({
  jsonText,
  loadError,
  onJsonTextChange,
  onFileSelected,
  onLoadPastedJson,
  onClose,
}: {
  jsonText: string
  loadError: string | null
  onJsonTextChange: (text: string) => void
  onFileSelected: (file: File) => void
  onLoadPastedJson: () => void
  onClose: () => void
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  function onFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) {
      onFileSelected(file)
    }
    event.target.value = ''
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="load-modal" role="dialog" aria-modal="true" aria-labelledby="load-json-title">
        <header>
          <div>
            <p className="eyebrow">Developer Tool</p>
            <h2 id="load-json-title">Load Trace Fixture</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close loader" onClick={onClose}>x</button>
        </header>
        <textarea
          value={jsonText}
          onChange={(event) => onJsonTextChange(event.target.value)}
          spellCheck={false}
          placeholder="Paste trace JSON here..."
        />
        {loadError ? <p className="load-error">{loadError}</p> : null}
        <footer>
          <input ref={fileInputRef} type="file" accept="application/json,.json" onChange={onFileInputChange} />
          <button type="button" onClick={() => fileInputRef.current?.click()}>Select File</button>
          <button type="button" className="primary-button" onClick={onLoadPastedJson}>Load Pasted JSON</button>
        </footer>
      </section>
    </div>
  )
}
