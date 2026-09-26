import { useEffect, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL || '/api'

const STATUS_TONE = {
  idle: 'dim', running: 'warn', done: 'good', error: 'bad',
}

export default function TrainPanel() {
  const [status, setStatus] = useState(null)
  const [err, setErr]       = useState('')
  const pollRef = useRef(null)

  async function fetchStatus() {
    try {
      const r = await fetch(`${API}/train/status`)
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      setStatus(data)
      setErr('')
      return data
    } catch (e) {
      setErr(e.message)
      return null
    }
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const data = await fetchStatus()
      if (cancelled) return
      if (data) {
        // handled by the interval below too, this just paints immediately
      }
    })()
    pollRef.current = setInterval(async () => {
      const data = await fetchStatus()
      if (data && data.status !== 'running' && pollRef.current) {
        clearInterval(pollRef.current)
      }
    }, 2000)
    return () => { cancelled = true; pollRef.current && clearInterval(pollRef.current) }
  }, [])

  async function start() {
    setErr('')
    try {
      const r = await fetch(`${API}/train/start`, { method: 'POST' })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      await fetchStatus()
      if (!pollRef.current) {
        pollRef.current = setInterval(async () => {
          const data = await fetchStatus()
          if (data && data.status !== 'running' && pollRef.current) {
            clearInterval(pollRef.current)
          }
        }, 2000)
      }
    } catch (e) { setErr(e.message) }
  }

  const busy = status?.status === 'running'
  const canTrain = status && status.dataset.ai_generated > 0 &&
    status.dataset.forged > 0 && status.dataset.original > 0
  const pct = status && status.total_epochs
    ? Math.round((status.current_epoch / status.total_epochs) * 100) : 0

  return (
    <section className="train fade">
      <div className="train-head">
        <h2>Train the model</h2>
        <p>
          Dataset -&gt; Original / AI-generated / Forged classes -&gt; TamperNet
          (RGB + SRM noise streams, U-Net decoder, classification head) -&gt; <code>checkpoints/best.pt</code>.
        </p>
      </div>

      {err && <div className="train-err">{err}</div>}

      {status && (
        <>
          <div className="train-stats">
            <div className="stat">
              <span className="stat-num">{status.dataset.original}</span>
              <span className="stat-cap">Original</span>
            </div>
            <div className="stat">
              <span className="stat-num">{status.dataset.ai_generated}</span>
              <span className="stat-cap">AI-generated</span>
            </div>
            <div className="stat">
              <span className="stat-num">{status.dataset.forged}</span>
              <span className="stat-cap">Forged</span>
            </div>
            <div className="stat">
              <span className={`stat-num status-${STATUS_TONE[status.status]}`}>
                {status.status}
              </span>
              <span className="stat-cap">Run status</span>
            </div>
            <div className="stat">
              <span className="stat-num">{status.has_checkpoint ? 'yes' : 'no'}</span>
              <span className="stat-cap">Checkpoint saved</span>
            </div>
          </div>

          <div className="train-actions">
            <button className="btn" onClick={start} disabled={busy || !canTrain}>
              {busy ? 'Training…' : 'Start training'}
            </button>
            {!canTrain && (
              <span className="train-hint">
                Add images to <code>data/original/</code>, <code>data/aigenerated/</code>, and <code>data/forged/</code> first.
              </span>
            )}
          </div>

          {(busy || status.history.length > 0) && (
            <div className="train-progress">
              <div className="bar-track wide">
                <div className="bar-fill warn" style={{ width: `${pct}%` }} />
              </div>
              <span className="train-progress-label">
                Epoch {status.current_epoch} / {status.total_epochs}
              </span>
            </div>
          )}

          {status.error && <div className="train-err">{status.error}</div>}

          {status.history.length > 0 && (
            <div className="train-table-wrap">
              <table className="train-table">
                <thead>
                  <tr><th>Epoch</th><th>Loss</th><th>AUC</th><th>F1</th><th>Pixel IoU</th></tr>
                </thead>
                <tbody>
                  {status.history.slice(-12).reverse().map(h => (
                    <tr key={h.epoch}>
                      <td>{h.epoch}</td>
                      <td>{h.loss?.toFixed?.(4) ?? '—'}</td>
                      <td>{h.auc?.toFixed?.(4) ?? '—'}</td>
                      <td>{h.f1?.toFixed?.(4) ?? '—'}</td>
                      <td>{h.pixel_iou?.toFixed?.(4) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  )
}
