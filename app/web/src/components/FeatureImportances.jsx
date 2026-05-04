export default function FeatureImportances({ data }) {
  if (!data || !data.length) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Feature importances not available.</p>
  }

  const max = data[0].importance

  return (
    <div className="fi-list">
      {data.slice(0, 12).map((item) => (
        <div key={item.label} className="fi-item">
          <div className="fi-label" title={item.label}>{item.label}</div>
          <div className="fi-track">
            <div
              className="fi-fill"
              style={{ width: `${(item.importance / max) * 100}%` }}
            />
          </div>
          <div className="fi-value">{(item.importance * 100).toFixed(1)}%</div>
        </div>
      ))}
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
        Mean Decrease in Impurity — from the trained RF model
      </p>
    </div>
  )
}
