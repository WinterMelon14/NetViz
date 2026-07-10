import type { RichText } from '../explanations'

export function RichTextView({ value }: { value: RichText }) {
  return (
    <>
      {value.map((part, index) =>
        part.kind === 'code' ? (
          <code key={index} className="code-text">
            {part.text}
          </code>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  )
}
