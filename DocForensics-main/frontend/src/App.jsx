import { useState, useRef } from 'react'
import './App.css'
import TrainPanel from './TrainPanel'

// Same-origin '/api' in production (FastAPI serves the built app); the Vite dev
// server proxies '/api' to the backend. Override with VITE_API_URL if needed.
const API = import.meta.env.VITE_API_URL || '/api'

const VERDICT = {
  'AUTHENTIC':    { icon: '✓', tone: 'good', note: 'No significant tampering signals detected.' },
  'TAMPERED':     { icon: '!', tone: 'bad',  note: 'This document shows signs of manipulation.' },
  'AI-GENERATED': { icon: '◆', tone: 'warn', note: 'This image appears to be AI-generated.' },
}

const DET_ICON = {
  ela: '🔬', noise: '📡', copy_move: '🧬', double_jpeg: '🗜️',
  font: '🔤', metadata: '🏷️', ai_generated: '🤖', model: '🧠',
}

function Dropzone({ onFile }) {
  const [drag, setDrag] = useState(false)
  const ref = useRef()
  return (
    <div
      className={`drop ${drag ? 'drag' : ''}`}
      onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]) }}
    >
      <input ref={ref} type="file" accept=".jpg,.jpeg,.png,.tiff,.pdf" hidden
             onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
      <div className="drop-icon">⬆</div>
      <div className="drop-title">Drop a document or image</div>
      <div className="drop-sub">JPG · PNG · TIFF · PDF — up to 20 MB</div>
      <div className="drop-cta">Choose file</div>
    </div>
  )
}

function Gauge({ value, tone }) {
  const pct = Math.round(value * 100)
  const r = 34, c = 2 * Math.PI * r
  const dash = (value * c)
  return (
    <div className="gauge">
      <svg width="84" height="84" viewBox="0 0 84 84">
        <circle cx="42" cy="42" r={r} className="gauge-bg" />
        <circle cx="42" cy="42" r={r} className={`gauge-fg ${tone}`}
                strokeDasharray={`${dash} ${c}`} transform="rotate(-90 42 42)" />
      </svg>
      <div className="gauge-label">
        <span className="gauge-num">{pct}<small>%</small></span>
        <span className="gauge-cap">suspicion</span>
      </div>
    </div>
  )
}

function Bar({ name, score }) {
  const pct = Math.round(score * 100)
  const tone = score > 0.6 ? 'bad' : score > 0.35 ? 'warn' : 'good'
  return (
    <div className="bar">
      <span className="bar-ico">{DET_ICON[name] || '•'}</span>
      <span className="bar-name">{name.replace(/_/g, ' ')}</span>
      <div className="bar-track"><div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
      <span className="bar-pct">{pct}%</span>
    </div>
  )
}

// Fixed display order per the workflow: Original -> AI Generated -> Forged
const CLASS_ORDER = ['Original', 'AI Generated', 'Forged']
const CLASS_ICON  = { 'Original': '✓', 'AI Generated': '◆', 'Forged': '!' }
const CLASS_TONE  = { 'Original': 'good', 'AI Generated': 'warn', 'Forged': 'bad' }

function ClassBreakdown({ scores }) {
  if (!scores) return null
  return (
    <div className="bars class-bars">
      {CLASS_ORDER.map(name => {
        const score = scores[name] ?? 0
        const pct   = Math.round(score * 100)
        const tone  = CLASS_TONE[name]
        return (
          <div className="bar" key={name}>
            <span className="bar-ico">{CLASS_ICON[name]}</span>
            <span className="bar-name class-bar-name">{name}</span>
            <div className="bar-track"><div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
            <span className="bar-pct">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('analyze')   // 'analyze' | 'train'
  const [file, setFile]   = useState(null)
  const [state, setState] = useState('idle')
  const [res, setRes]     = useState(null)
  const [err, setErr]     = useState('')
  const [view, setView]   = useState('heatmap')

  async function run(f) {
    setFile(f); setState('loading'); setRes(null); setView('heatmap')
    const form = new FormData(); form.append('file', f)
    try {
      const r = await fetch(`${API}/analyze`, { method: 'POST', body: form })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Server error ${r.status}`)
      setRes(await r.json()); setState('done')
    } catch (e) { setErr(e.message); setState('error') }
  }

  const reset = () => { setFile(null); setRes(null); setState('idle') }
  const v = res ? (VERDICT[res.label] || VERDICT.AUTHENTIC) : null

  return (
    <div className="page">
      <header className="nav">
        <div className="brand"><span className="brand-mark">🔍</span> Detection of AI-Generated and Forged Images Using deep learning</div>
        <div className="brand-sub">Deep learning-based forgery and AI-image detection</div>
        <div className="toggle nav-tabs">
          <button className={page === 'analyze' ? 'on' : ''} onClick={() => setPage('analyze')}>Analyze</button>
          <button className={page === 'train' ? 'on' : ''} onClick={() => setPage('train')}>Train</button>
        </div>
        <span className="nav-badge">8 detectors · CNN</span>
      </header>

      <main className="wrap">
        {page === 'train' && <TrainPanel />}

        {page === 'analyze' && state === 'idle' && (
          <section className="intro fade">
            <span className="pill">AI + forged image forensics</span>
            <h1>Detect AI-generated or forged images</h1>
            <p>Upload an image or PDF. Eight forensic detectors and a trained CNN
               inspect it for cloning, recompression, and AI-generated content.</p>
            <Dropzone onFile={run} />
            <div className="trust-row">
              <span>🧬 Copy-move</span><span>🔬 Error-level analysis</span>
              <span>🤖 AI detection</span><span>🧠 Neural localization</span>
            </div>
          </section>
        )}

        {page === 'analyze' && state === 'loading' && (
          <section className="status fade">
            <div className="ring" />
            <p className="status-main">Analyzing <b>{file?.name}</b></p>
            <p className="status-sub">Running detectors + CNN model…</p>
          </section>
        )}

        {page === 'analyze' && state === 'error' && (
          <section className="status fade">
            <div className="status-bad">⚠</div>
            <p className="status-main">{err}</p>
            <button className="btn" onClick={reset}>Try again</button>
          </section>
        )}

        {page === 'analyze' && state === 'done' && res && (
          <section className="result fade">
            <div className={`verdict ${v.tone}`}>
              <Gauge value={res.confidence} tone={v.tone} />
              <div className="verdict-text">
                <div className="verdict-row">
                  <span className="verdict-badge">{v.icon}</span>
                  <span className="verdict-label">{res.label}</span>
                </div>
                <div className="verdict-note">{v.note}</div>
              </div>
              <button className="btn ghost" onClick={reset}>New scan</button>
            </div>

            <div className="grid">
              <div className="card">
                <div className="card-head">
                  <h2>Evidence map</h2>
                  <div className="toggle">
                    <button className={view === 'heatmap' ? 'on' : ''} onClick={() => setView('heatmap')}>Heatmap</button>
                    <button className={view === 'original' ? 'on' : ''} onClick={() => setView('original')}>Original</button>
                  </div>
                </div>
                <div className="frame">
                  {file && <img className="frame-base" src={URL.createObjectURL(file)} alt="" />}
                  {view === 'heatmap' && res.heatmap_base64 &&
                    <img className="frame-heat" src={`data:image/png;base64,${res.heatmap_base64}`} alt="" />}
                </div>
                <p className="hint">Brighter / red areas indicate higher suspicion.</p>
              </div>

              <div className="card">
                <div className="card-head"><h2>Classification</h2></div>
                <ClassBreakdown scores={res.class_scores} />

                <div className="card-head sub"><h2>Detector breakdown</h2></div>
                <div className="bars">
                  {res.per_detector.map(d => <Bar key={d.name} name={d.name} score={d.score} />)}
                </div>
                {res.evidence?.length > 0 && (
                  <div className="evidence">
                    <h3>Findings</h3>
                    <ul>{res.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="foot">
        <div className="foot-meta">Built with FastAPI · PyTorch · React</div>
      </footer>
    </div>
  )
}
