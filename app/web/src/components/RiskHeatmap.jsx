function getColor(clauseType, confidence) {
  const t = (clauseType || '').toLowerCase()
  if (t.includes('liability') || t.includes('assignment') || t.includes('compete') || t.includes('ownership')) {
    return confidence > 0.2 ? '#c0392b' : '#f0c4c0'
  }
  if (t.includes('termination') || t.includes('payment') || t.includes('insurance') || t.includes('minimum')) {
    return confidence > 0.2 ? '#b45309' : '#f0dfa0'
  }
  if (t.includes('governing') || t.includes('audit') || t.includes('covenant')) {
    return '#a7d9b8'
  }
  return '#e4e4e4'
}

export default function RiskHeatmap({ clauses }) {
  if (!clauses?.length) return <p style={{ color: 'var(--text-muted)' }}>No clause data</p>

  return (
    <>
      <div className="heatmap">
        {clauses.slice(0, 200).map((c, i) => (
          <div
            key={i}
            className="heatmap-cell"
            style={{ background: getColor(c.clause_type, c.confidence || 0) }}
            title={`${c.clause_type} (${((c.confidence || 0) * 100).toFixed(0)}%)`}
          />
        ))}
      </div>
      <div className="heatmap-legend">
        {[
          { color: '#c0392b', label: 'High risk' },
          { color: '#f0c4c0', label: 'High risk (low confidence)' },
          { color: '#b45309', label: 'Medium risk' },
          { color: '#a7d9b8', label: 'Mitigating clause' },
          { color: '#e4e4e4', label: 'Neutral / other' },
        ].map(({ color, label }) => (
          <div key={label} className="legend-item">
            <div className="legend-dot" style={{ background: color, border: '1px solid #ddd' }} />
            {label}
          </div>
        ))}
      </div>
    </>
  )
}
