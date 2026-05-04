import { useState, useCallback } from 'react'
import './index.css'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import LoadingView from './components/LoadingView'
import Dashboard from './components/Dashboard'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fileName, setFileName] = useState(null)

  const handleFile = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setFileName(file.name)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Analysis failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleReset = () => {
    setResult(null)
    setError(null)
    setFileName(null)
    setLoading(false)
  }

  return (
    <>
      <Header />
      <main className="main">
        {!result && !loading && (
          <>
            <UploadZone onFile={handleFile} />
            {error && (
              <div className="error-box" style={{ marginTop: 16 }}>
                <span>⚠</span>
                <span><strong>Analysis failed:</strong> {error}</span>
              </div>
            )}
          </>
        )}
        {loading && <LoadingView />}
        {result && (
          <Dashboard
            result={result}
            fileName={fileName}
            onReset={handleReset}
            apiBase={API_BASE}
          />
        )}
      </main>
    </>
  )
}
