import { useState } from 'react'
import ClauseList from './ClauseList'
import ClauseViewer from './ClauseViewer'
import AnnotatedContract from './AnnotatedContract'
import ContractTextExplorer from './ContractTextExplorer'
import RiskHeatmap from './RiskHeatmap'
import FeatureImportances from './FeatureImportances'

const RISK_CLS = { High: 'high', Medium: 'medium', Low: 'low' }

function ProbBar({ label, value, cls }) {
  return (
    <div className="prob-item">
      <div className="prob-header">
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="prob-track">
        <div className={`prob-fill ${cls}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

export default function Dashboard({ result, fileName, onReset, apiBase }) {
  const [activeClause, setActiveClause] = useState(0)
  const [activeTab, setActiveTab] = useState('annotate')
  const [showAllSegments, setShowAllSegments] = useState(false)

  const riskCls = RISK_CLS[result.risk_level] || 'low'
  const probs = result.risk_probabilities || {}
  const allSegments = result.bert_details || []
  const namedClauses = allSegments.filter(d => d.clause_type !== 'Other')
  const clauseListSource = showAllSegments ? allSegments : namedClauses
  const clauseIdx = Math.min(activeClause, Math.max(0, clauseListSource.length - 1))

  return (
    <div className="fade-in">

      {/* Page header */}
      <div className="section-header">
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 5, fontFamily: 'var(--font-mono)' }}>
            {fileName} · {result.n_clauses} segments
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <h2 className="section-title">Risk Analysis</h2>
            <span className={`risk-badge ${riskCls}`}>{result.risk_level} Risk</span>
          </div>
        </div>
        <button className="reset-btn" onClick={onReset}>← New analysis</button>
      </div>

      <div className="divider" style={{ marginBottom: 20 }} />

      {/* Metric cards */}
      <div className="metrics-row">
        <div className={`metric-card risk-${riskCls}`}>
          <div className="metric-label">Overall Risk</div>
          <div className={`metric-value ${riskCls}`}>{result.risk_level}</div>
          <div className="metric-sub">Contract classification</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Named Clauses</div>
          <div className="metric-value">{namedClauses.length}</div>
          <div className="metric-sub">of 41 CUAD categories detected</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">High-Risk Probability</div>
          <div className="metric-value high">{((probs.High || 0) * 100).toFixed(1)}%</div>
          <div className="metric-sub">Calibrated RF confidence</div>
        </div>
      </div>

      {/* Dashboard grid */}
      <div className="dashboard">

        {/* Left sidebar */}
        <div className="dashboard-left">
          <div className="card">
            <div className="card-title">Risk Distribution</div>
            <div className="prob-bar">
              <ProbBar label="High" value={probs.High || 0} cls="high" />
              <ProbBar label="Medium" value={probs.Medium || 0} cls="medium" />
              <ProbBar label="Low" value={probs.Low || 0} cls="low" />
            </div>
          </div>

          <div className="card">
            <div className="card-title">Feature Importance</div>
            <FeatureImportances data={result.feature_importances} />
          </div>

          <div style={{
            padding: '12px 16px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            fontSize: 12,
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
            lineHeight: 1.7,
          }}>
            Backend: {(result.reasoner || 'rf').toUpperCase()}<br />
            Model: Legal-BERT + RF<br />
            Metrics: see <code style={{ fontSize: 10 }}>hybrid_eval.json</code>
          </div>
        </div>

        {/* Right content */}
        <div className="dashboard-right">
          <div className="tabs tabs-4">
            <button type="button" className={`tab${activeTab === 'annotate' ? ' active' : ''}`} onClick={() => setActiveTab('annotate')}>
              Markup
            </button>
            <button type="button" className={`tab${activeTab === 'text' ? ' active' : ''}`} onClick={() => setActiveTab('text')}>
              Plain text
            </button>
            <button type="button" className={`tab${activeTab === 'clauses' ? ' active' : ''}`} onClick={() => setActiveTab('clauses')}>
              List ({clauseListSource.length})
            </button>
            <button type="button" className={`tab${activeTab === 'heatmap' ? ' active' : ''}`} onClick={() => setActiveTab('heatmap')}>
              Heatmap
            </button>
          </div>

          {activeTab === 'annotate' && (
            <div className="card" style={{ padding: 18 }}>
              <div className="card-title" style={{ marginBottom: 12 }}>Annotated contract</div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
                Each block follows segmentation order. The colored rail matches heuristic <strong>clause concern</strong>;
                the title line is the CUAD-style classification from Legal-BERT (top prediction per segment).
              </p>
              <AnnotatedContract
                segments={result.segment_annotations || []}
                overallRisk={result.risk_level}
              />
            </div>
          )}

          {activeTab === 'text' && (
            <div className="card" style={{ padding: 18 }}>
              <ContractTextExplorer
                preview={result.contract_text_preview || ''}
                totalChars={result.contract_char_total || 0}
                truncated={!!result.contract_truncated}
                apiBase={apiBase}
              />
            </div>
          )}

          {activeTab === 'clauses' && (
            <div className="clause-split">
              <div className="card" style={{ padding: 14 }}>
                <div className="card-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                  <span>Detected segments</span>
                  <label className="toggle-all">
                    <input
                      type="checkbox"
                      checked={showAllSegments}
                      onChange={e => {
                        setShowAllSegments(e.target.checked)
                        setActiveClause(0)
                      }}
                    />
                    All segments
                  </label>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
                  Each row is text BERT scored during contract risk analysis. Select one to read the exact wording.
                </p>
                <ClauseList clauses={clauseListSource} activeIdx={clauseIdx} onSelect={setActiveClause} />
              </div>
              <ClauseViewer clause={clauseListSource[clauseIdx]} />
            </div>
          )}

          {activeTab === 'heatmap' && (
            <div className="card">
              <div className="card-title">Clause Risk Density</div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                Each cell represents one detected clause segment. Hover for details.
              </p>
              <RiskHeatmap clauses={result.bert_details} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
