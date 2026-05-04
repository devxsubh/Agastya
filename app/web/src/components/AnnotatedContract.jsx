/**
 * Minimal “marked-up contract”: each segmented passage is a rounded block with a
 * colored rail (risk tier), CUAD classification label, and heuristic clause concern level.
 */

function riskRailClass(tier) {
  const t = (tier || '').toLowerCase()
  if (t === 'high') return 'anno-rail-high'
  if (t === 'medium') return 'anno-rail-medium'
  return 'anno-rail-low'
}

function riskChipClass(tier) {
  const t = (tier || '').toLowerCase()
  if (t === 'high') return 'anno-chip-high'
  if (t === 'medium') return 'anno-chip-medium'
  return 'anno-chip-low'
}

export default function AnnotatedContract({
  segments,
  overallRisk,
}) {
  const rows = Array.isArray(segments) ? segments : []

  if (!rows.length) {
    return (
      <div className="anno-empty">
        <p>No segmented markup returned for this document.</p>
      </div>
    )
  }

  return (
    <div className="anno-root">
      <div className="anno-legend">
        <span className="anno-legend-title">Legend</span>
        <span className="anno-legend-item"><span className="anno-dot anno-dot-high" aria-hidden /> Clause concern — High</span>
        <span className="anno-legend-item"><span className="anno-dot anno-dot-medium" aria-hidden /> Clause concern — Medium</span>
        <span className="anno-legend-item"><span className="anno-dot anno-dot-low" aria-hidden /> Clause concern — Low</span>
        <span className="anno-legend-note">
          Colors reflect a keyword heuristic on the CUAD label (not the overall contract RF score).
          Overall contract risk: <strong>{overallRisk || '—'}</strong>
        </span>
      </div>

      <div className="anno-scroll">
        {rows.map((seg) => (
          <article key={seg.segment_index} className={`anno-block ${riskRailClass(seg.clause_risk)}`}>
            <header className="anno-head">
              <span className="anno-num" title="Segment order">{seg.segment_index + 1}</span>
              <div className="anno-labels">
                <span className="anno-class-title" title="Legal-BERT CUAD classification">
                  {seg.phase2_label}
                </span>
                <span className={`anno-chip ${riskChipClass(seg.clause_risk)}`}>
                  Clause concern: {seg.clause_risk}
                </span>
                <span className="anno-meta">
                  {(seg.confidence * 100).toFixed(0)}% · RF bucket {seg.mapped_bucket}
                </span>
              </div>
            </header>
            <div className="anno-body">{seg.text}</div>
          </article>
        ))}
      </div>
    </div>
  )
}
