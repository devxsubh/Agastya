export default function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
          <polygon points="11,2 20,7 20,17 11,22 2,17 2,7" stroke="#0a0a0a" strokeWidth="1.5" fill="none" />
          <circle cx="11" cy="12" r="2.5" fill="#0a0a0a" />
          <line x1="11" y1="2" x2="11" y2="9.5" stroke="#0a0a0a" strokeWidth="1" opacity="0.35" />
        </svg>
        <span className="header-title">Agastya</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className="header-badge">Legal-BERT + RF · Hybrid v2</span>
        <span className="header-model">Macro-F1 0.819</span>
      </div>
    </header>
  )
}
