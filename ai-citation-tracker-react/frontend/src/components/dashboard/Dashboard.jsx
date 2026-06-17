import { useState, useEffect, useRef } from 'react'
import { api } from '../../lib/api'
import PromptDetailModal from './PromptDetailModal'

const MODEL_COLORS = {
  ChatGPT: '#10a37f',
  Perplexity: '#6366f1',
  Gemini: '#4285f4',
  Groq: '#f55036',
  Claude: '#d97706',
}

function CircleProgress({ pct, model, color }) {
  const r = 22
  const circ = 2 * Math.PI * r
  const offset = circ - (pct / 100) * circ

  return (
    <div className="circle-metric">
      <svg width="60" height="60" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r={r} fill="none" stroke="#e5e7eb" strokeWidth="5" />
        <circle
          cx="30" cy="30" r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 30 30)"
        />
        <text x="30" y="35" textAnchor="middle" fontSize="12" fontWeight="700" fill="var(--color-text-primary)">
          {pct}%
        </text>
      </svg>
      <span className="circle-label">{model}</span>
    </div>
  )
}

function BarProgress({ pct, color }) {
  return (
    <div className="bar-wrap">
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="bar-pct">{pct}%</span>
    </div>
  )
}

function TopicRow({ topicData, models, brandName }) {
  const [open, setOpen] = useState(true)
  const [selectedPrompt, setSelectedPrompt] = useState(null)

  return (
    <div className="topic-section">
      <div className="topic-row-header">
        <button className="topic-name-btn" onClick={() => setOpen(!open)}>
          <span className="topic-name-text">{topicData.topic_name}</span>
        </button>
        <div className="topic-model-bars">
          {models.map((model) => {
            const vis = topicData.visibility_by_model?.[model]
            return (
              <div key={model} className="topic-bar-cell">
                <BarProgress pct={vis?.pct || 0} color={MODEL_COLORS[model] || '#6b7280'} />
              </div>
            )
          })}
        </div>
        <button className="btn-analyze">+ Analyze & Improve</button>
      </div>

      {open && (
        <div className="prompt-rows">
          {topicData.prompts.map((prompt) => (
            <div
              key={prompt.prompt_id}
              className="prompt-result-row"
              onClick={() => setSelectedPrompt(prompt)}
            >
              <span className="prompt-result-text">+ {prompt.prompt_text}</span>
              <div className="model-badges">
                {models.map((model) => {
                  const r = prompt.results_by_model?.[model]
                  return (
                    <div key={model} className="badge-cell">
                      {r?.brand_mentioned ? (
                        <span className="badge-brand">Brand</span>
                      ) : (
                        <span className="badge-dash">—</span>
                      )}
                    </div>
                  )
                })}
              </div>
              <div className="prompt-meta">
                <span className="prompt-location">US</span>
                <span className="prompt-updated">Just now</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedPrompt && (
        <PromptDetailModal
          prompt={selectedPrompt}
          models={models}
          brandName={brandName}
          onClose={() => setSelectedPrompt(null)}
        />
      )}
    </div>
  )
}

export default function Dashboard({ runId, brandName, brandDomain, onNewRun }) {
  const [results, setResults] = useState(null)
  const [status, setStatus] = useState('pending')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const pollRef = useRef(null)

  useEffect(() => {
    pollStatus()
    return () => clearInterval(pollRef.current)
  }, [runId])

  const pollStatus = () => {
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getRunStatus(runId)
        setStatus(s.status)
        setProgress(s.progress)

        if (s.status === 'completed') {
          clearInterval(pollRef.current)
          fetchResults()
        } else if (s.status === 'failed') {
          clearInterval(pollRef.current)
          setError('Tracking run failed. Please try again.')
        }
      } catch (e) {
        clearInterval(pollRef.current)
        setError(e.message)
      }
    }, 2000)
  }

  const fetchResults = async () => {
    try {
      const data = await api.getRunResults(runId)
      setResults(data)
    } catch (e) {
      setError(e.message)
    }
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <p>{error}</p>
        <button className="btn-primary" onClick={onNewRun}>Start New Tracking</button>
      </div>
    )
  }

  if (status !== 'completed' || !results) {
    return (
      <div className="tracking-progress">
        <div className="progress-card">
          <div className="spinner-lg" />
          <h3>Running tracking...</h3>
          <p>Sending prompts to AI models and analyzing responses</p>
          <div className="progress-bar-wrap">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progress-pct">{progress}% complete</p>
        </div>
      </div>
    )
  }

  const models = results.models || []
  const topics = results.topics || []

  return (
    <div className="dashboard">
      {/* Top bar */}
      <div className="dash-topbar">
        <div className="dash-brand-info">
          <div className="dash-brand-avatar">{brandName?.[0]?.toUpperCase()}</div>
          <div>
            <h2 className="dash-brand-name">{brandName}</h2>
            <p className="dash-brand-domain">{brandDomain}</p>
          </div>
          <div className="dash-stats">
            <div className="dash-stat">
              <span className="dash-stat-num">{topics.length}</span>
              <span className="dash-stat-label">Topics</span>
            </div>
            <div className="dash-stat">
              <span className="dash-stat-num">{topics.reduce((a, t) => a + t.prompts.length, 0)}</span>
              <span className="dash-stat-label">Prompts</span>
            </div>
          </div>
        </div>

        <div className="dash-visibility">
          {models.map((model) => {
            const vis = results.overall_by_model?.[model]
            return (
              <CircleProgress
                key={model}
                model={model}
                pct={vis?.pct || 0}
                color={MODEL_COLORS[model] || '#6b7280'}
              />
            )
          })}
        </div>

        <button className="btn-secondary" onClick={onNewRun}>+ New Run</button>
      </div>

      {/* Results table */}
      <div className="dash-table">
        <div className="dash-table-header">
          <span className="col-topic">Topics {topics.length}, Prompts ({topics.reduce((a, t) => a + t.prompts.length, 0)})</span>
          {models.map((model) => (
            <span key={model} className="col-model">{model}</span>
          ))}
          <span className="col-actions" />
        </div>

        {topics.map((topicData) => (
          <TopicRow
            key={topicData.topic_id}
            topicData={topicData}
            models={models}
            brandName={brandName}
          />
        ))}
      </div>
    </div>
  )
}
