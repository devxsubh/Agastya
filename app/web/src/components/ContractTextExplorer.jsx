import { useCallback, useRef, useState } from 'react'

const MAX_CHARS = 8000

function selectionInside(container, winSel) {
  if (!container || !winSel || winSel.rangeCount === 0) return ''
  const range = winSel.getRangeAt(0)
  if (!container.contains(range.commonAncestorContainer)) return ''
  return winSel.toString().trim()
}

export default function ContractTextExplorer({ preview, totalChars, truncated, apiBase }) {
  const excerptRef = useRef(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [preds, setPreds] = useState(null)
  const [lastQuery, setLastQuery] = useState('')

  const classifySelection = useCallback(async () => {
    const sel = selectionInside(excerptRef.current, window.getSelection())
    setErr(null)
    setPreds(null)
    if (sel.length < 4) {
      setErr('Select at least a few words inside the contract text.')
      return
    }
    const slice = sel.length > MAX_CHARS ? sel.slice(0, MAX_CHARS) : sel
    setBusy(true)
    setLastQuery(slice)
    try {
      const res = await fetch(`${apiBase}/classify-clause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: slice }),
      })
      if (!res.ok) {
        const j = await res.json().catch(() => ({}))
        throw new Error(j.detail || res.statusText)
      }
      const data = await res.json()
      setPreds(data.predictions || [])
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }, [apiBase])

  if (!preview) {
    return (
      <div className="card" style={{ padding: 20 }}>
        <div className="card-title">Contract text</div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>No excerpt available for this file.</p>
      </div>
    )
  }

  return (
    <div className="contract-explorer">
      <div className="card-title" style={{ marginBottom: 8 }}>Contract text</div>
      <p className="contract-explorer-hint">
        Read the excerpt below. <strong>Select any passage</strong> with your cursor, then run clause classification on exactly that span.
      </p>
      <div className="contract-meta">
        {truncated ? (
          <>
            Showing <strong>{preview.length.toLocaleString()}</strong> of{' '}
            <strong>{totalChars.toLocaleString()}</strong> characters (preview trimmed for the browser).
          </>
        ) : (
          <>
            <strong>{totalChars.toLocaleString()}</strong> characters in document.
          </>
        )}
      </div>

      <div ref={excerptRef} className="contract-excerpt">
        {preview}
      </div>

      <div className="contract-actions">
        <button type="button" className="primary-btn" disabled={busy} onClick={classifySelection}>
          {busy ? 'Classifying…' : 'Classify selected text'}
        </button>
        <span className="contract-actions-note">Legal-BERT top labels (41 CUAD). Does not change overall contract risk until you re-run full analysis.</span>
      </div>

      {err && <div className="inline-error">{err}</div>}

      {preds && preds.length > 0 && (
        <div className="selection-predictions">
          <div className="card-title" style={{ marginBottom: 10 }}>Predictions for selection</div>
          {lastQuery && (
            <div className="selection-query">
              “{lastQuery.length > 220 ? `${lastQuery.slice(0, 220)}…` : lastQuery}”
            </div>
          )}
          <ul className="prediction-list">
            {preds.map((p, i) => (
              <li key={`${p.phase2_label}-${i}`} className="prediction-row">
                <span className="prediction-rank">{i + 1}</span>
                <div className="prediction-body">
                  <div className="prediction-label">{p.phase2_label}</div>
                  <div className="prediction-sub">
                    RF bucket: <code>{p.mapped_bucket}</code> · {(p.confidence * 100).toFixed(1)}% score
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
