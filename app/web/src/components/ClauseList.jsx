const RISK_CLAUSE_TYPES = {
  High: ['Cap On Liability', 'Uncapped Liability', 'Anti-Assignment', 'Change Of Control', 'Non-Compete', 'Ip Ownership Assignment'],
  Medium: ['Termination For Convenience', 'Notice Period To Terminate Renewal', 'Revenue/Profit Sharing', 'Minimum Commitment'],
  Low: ['Governing Law', 'Audit Rights', 'Covenant Not To Sue'],
}

function clauseRiskColor(clauseType, confidence) {
  if (RISK_CLAUSE_TYPES.High.some(t => clauseType.includes(t.split(' ')[0]))) {
    return confidence > 0.3 ? '#ef4444' : '#fca5a5'
  }
  if (RISK_CLAUSE_TYPES.Medium.some(t => clauseType.includes(t.split(' ')[0]))) {
    return confidence > 0.3 ? '#f59e0b' : '#fde68a'
  }
  if (RISK_CLAUSE_TYPES.Low.some(t => clauseType.includes(t.split(' ')[0]))) {
    return '#10b981'
  }
  return '#334155'
}

export default function ClauseList({ clauses, activeIdx, onSelect }) {
  if (!clauses.length) {
    return (
      <div className="empty-hero">
        <div className="empty-hero-icon">📭</div>
        <div className="empty-hero-title">No named clauses detected</div>
        <div className="empty-hero-sub">Legal-BERT did not find any of the 41 CUAD clause types</div>
      </div>
    )
  }

  return (
    <div className="clause-list">
      {clauses.map((c, i) => {
        const color = clauseRiskColor(c.clause_type, c.confidence)
        return (
          <div
            key={i}
            className={`clause-item ${i === activeIdx ? 'active' : ''}`}
            onClick={() => onSelect(i)}
          >
            <div className="clause-dot" style={{ background: color }} />
            <div className="clause-name">{c.clause_type}</div>
            <div className="clause-conf">{(c.confidence * 100).toFixed(0)}%</div>
          </div>
        )
      })}
    </div>
  )
}
