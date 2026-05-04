import { useState, useEffect } from 'react'

const STEPS = [
  'Extracting text from document',
  'Running Legal-BERT clause detection',
  'Building 41-dim feature vector',
  'Random Forest risk classification',
  'Computing feature importances',
]

export default function LoadingView() {
  const [stepIdx, setStepIdx] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setStepIdx(i => (i < STEPS.length - 1 ? i + 1 : i)), 1800)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="loading-overlay fade-in">
      <div className="spinner" />
      <div>
        <div className="loading-text">Analyzing contract…</div>
        <div className="loading-sub" style={{ marginTop: 4 }}>
          Legal-BERT processes your document — this takes 30–60 s
        </div>
      </div>
      <div className="loading-steps">
        {STEPS.map((step, i) => (
          <div key={step} className={`step${i === stepIdx ? ' active' : i < stepIdx ? ' done' : ''}`}>
            <span>{i < stepIdx ? '✓' : i === stepIdx ? '·' : '○'}</span>
            {step}
          </div>
        ))}
      </div>
    </div>
  )
}
