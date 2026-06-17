import { useState, useEffect } from 'react'
import { api } from '../../lib/api'

function TopicAccordion({ topic, brandId, apiKeys }) {
  const [open, setOpen] = useState(true)
  const [prompts, setPrompts] = useState([])
  const [loading, setLoading] = useState(true)
  const [newPrompt, setNewPrompt] = useState('')

  useEffect(() => {
    generatePrompts()
  }, [])

  const generatePrompts = async () => {
    setLoading(true)
    try {
      const data = await api.generatePrompts(topic.id, brandId, apiKeys)
      setPrompts(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const togglePrompt = async (prompt) => {
    try {
      await api.updatePrompt(prompt.id, !prompt.is_selected)
      setPrompts((prev) =>
        prev.map((p) => p.id === prompt.id ? { ...p, is_selected: !p.is_selected } : p)
      )
    } catch (e) {
      console.error(e)
    }
  }

  const addPrompt = async () => {
    const val = newPrompt.trim()
    if (!val) return
    try {
      const created = await api.addPrompt(topic.id, val)
      setPrompts((prev) => [...prev, created])
      setNewPrompt('')
    } catch (e) {
      console.error(e)
    }
  }

  const deletePrompt = async (promptId) => {
    try {
      await api.deletePrompt(promptId)
      setPrompts((prev) => prev.filter((p) => p.id !== promptId))
    } catch (e) {
      console.error(e)
    }
  }

  const selectedCount = prompts.filter((p) => p.is_selected).length

  return (
    <div className="accordion">
      <button className="accordion-header" onClick={() => setOpen(!open)}>
        <span className="accordion-title">
          {topic.name}
          <span className="prompt-count">({selectedCount} prompts)</span>
        </span>
        <span className={`accordion-arrow ${open ? 'open' : ''}`}>▲</span>
      </button>

      {open && (
        <div className="accordion-body">
          {loading ? (
            <div className="loading-inline">
              <div className="spinner-sm" /> Generating prompts...
            </div>
          ) : (
            <>
              {prompts.map((prompt) => (
                <div key={prompt.id} className={`prompt-row ${prompt.is_selected ? 'prompt-selected' : ''}`}>
                  <div className="prompt-left">
                    <button
                      className={`checkbox ${prompt.is_selected ? 'checked' : ''}`}
                      onClick={() => togglePrompt(prompt)}
                    >
                      {prompt.is_selected && '✓'}
                    </button>
                    <span className="prompt-text">{prompt.text}</span>
                  </div>
                  <button className="icon-btn" onClick={() => deletePrompt(prompt.id)}>×</button>
                </div>
              ))}

              <div className="add-prompt-row">
                <span className="add-icon">+</span>
                <input
                  className="input-inline"
                  value={newPrompt}
                  onChange={(e) => setNewPrompt(e.target.value)}
                  placeholder="Add prompt"
                  onKeyDown={(e) => e.key === 'Enter' && addPrompt()}
                />
                <button className="btn-add-sm" onClick={addPrompt}>+</button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function Step3Prompts({ topics, brandId, apiKeys, onStartTracking, onBack }) {
  const totalPrompts = topics.length * 5

  return (
    <div className="step-card">
      <div className="step3-header">
        <div>
          <p className="step-hint">
            The prompts under each topic are what we'll enter into each AI search tool to monitor your brand visibility.
          </p>
        </div>
        <div className="prompt-usage">
          <span>{totalPrompts} of {totalPrompts} prompts</span>
          <button className="btn-primary" onClick={onStartTracking}>
            Start Tracking →
          </button>
        </div>
      </div>

      <div className="accordions">
        {topics.map((topic) => (
          <TopicAccordion
            key={topic.id}
            topic={topic}
            brandId={brandId}
            apiKeys={apiKeys}
          />
        ))}
      </div>

      <div className="step-footer">
        <button className="btn-secondary" onClick={onBack}>← Back</button>
        <button className="btn-primary" onClick={onStartTracking}>
          Start Tracking →
        </button>
      </div>
    </div>
  )
}
