const HIGH_RISK_TYPES = ['Cap On Liability', 'Uncapped Liability', 'Anti-Assignment', 'Non-Compete', 'Ip Ownership Assignment', 'Change Of Control', 'Liquidated Damages', 'Irrevocable Or Perpetual License']
const MEDIUM_RISK_TYPES = ['Termination For Convenience', 'Notice Period', 'Revenue', 'Minimum Commitment', 'Insurance', 'Non-Disparagement', 'Non-Solicit']

function classify(clauseType) {
  if (HIGH_RISK_TYPES.some(t => clauseType.includes(t.split(' ')[0]))) return 'high'
  if (MEDIUM_RISK_TYPES.some(t => clauseType.includes(t.split(' ')[0]))) return 'medium'
  if (['Governing Law', 'Audit Rights', 'Covenant'].some(t => clauseType.includes(t.split(' ')[0]))) return 'mitigating'
  return 'neutral'
}

export default function ClauseViewer({ clause }) {
  if (!clause) {
    return (
      <div className="card" style={{ minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="empty-hero">
          <div className="empty-hero-icon">👈</div>
          <div className="empty-hero-title">Select a clause</div>
          <div className="empty-hero-sub">Click any clause in the list to view its text</div>
        </div>
      </div>
    )
  }

  const cls = classify(clause.clause_type)
  const headerCls = cls === 'high' ? 'high' : cls === 'medium' ? 'medium' : cls === 'mitigating' ? 'low' : 'default'

  return (
    <div className="card">
      <div className={`clause-viewer-header ${headerCls}`}>
        <span title="CUAD-style clause label">{clause.clause_type}</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, opacity: 0.8, textAlign: 'right' }}>
          Model confidence {(clause.confidence * 100).toFixed(1)}%
        </span>
      </div>
      <div className="clause-viewer clause-viewer-body">
        {clause.text || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>No text available for this segment.</span>}
      </div>
    </div>
  )
}
