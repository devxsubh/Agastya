import { useState, useRef } from 'react'

export default function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file) onFile(file)
  }

  return (
    <div style={{ maxWidth: 580, margin: '72px auto' }}>

      {/* Hero text */}
      <div style={{ marginBottom: 36 }}>
        <div style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1.2px', color: '#999', marginBottom: 12 }}>
          Contract Risk Classifier
        </div>
        <h1 style={{ fontSize: 34, fontWeight: 800, letterSpacing: '-0.8px', lineHeight: 1.15, marginBottom: 14, color: '#0a0a0a' }}>
          Analyze any contract<br />in seconds.
        </h1>
        <p style={{ color: '#666', fontSize: 15, lineHeight: 1.7 }}>
          Upload a PDF or text file. Legal-BERT detects clause types,
          the Random Forest classifies contract-level risk as{' '}
          <span style={{ fontWeight: 600, color: '#c0392b' }}>High</span>,{' '}
          <span style={{ fontWeight: 600, color: '#b45309' }}>Medium</span>, or{' '}
          <span style={{ fontWeight: 600, color: '#166534' }}>Low</span>.
        </p>
      </div>

      {/* Upload zone */}
      <div
        className={`upload-zone${dragging ? ' dragover' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt"
          style={{ display: 'none' }}
          onChange={handleChange}
        />
        <div className="upload-icon">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="5" y="5" width="30" height="30" rx="5" />
            <path d="M20 13v14M13 20l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="upload-title">Drop your contract here</div>
        <div className="upload-subtitle">or click to browse files from your computer</div>
        <div className="upload-types">
          <span className="type-badge">PDF</span>
          <span className="type-badge">TXT</span>
        </div>
      </div>

      {/* Feature list */}
      <div style={{ marginTop: 28, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {[
          { label: '41 CUAD clause categories', detail: 'Full CUAD legal clause taxonomy' },
          { label: 'Calibrated RF probabilities', detail: 'High / Medium / Low with confidence scores' },
          { label: 'Feature importance breakdown', detail: 'Which clauses drove the risk decision' },
        ].map(({ label, detail }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13, color: '#555' }}>
            <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#0a0a0a', flexShrink: 0 }} />
            <span><strong style={{ color: '#0a0a0a' }}>{label}</strong> — {detail}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
