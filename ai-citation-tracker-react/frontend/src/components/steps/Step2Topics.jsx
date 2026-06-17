import { useState, useEffect } from 'react'
import { api } from '../../lib/api'

export default function Step2Topics({ brandId, apiKeys, onNext, onBack }) {
  const [topics, setTopics] = useState([])
  const [loading, setLoading] = useState(true)
  const [newTopic, setNewTopic] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    generateTopics()
  }, [])

  const generateTopics = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api.generateTopics(brandId, apiKeys)
      setTopics(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleTopic = async (topic) => {
    try {
      await api.updateTopic(topic.id, !topic.is_selected)
      setTopics((prev) =>
        prev.map((t) => t.id === topic.id ? { ...t, is_selected: !t.is_selected } : t)
      )
    } catch (e) {
      console.error(e)
    }
  }

  const addTopic = async () => {
    const val = newTopic.trim()
    if (!val) return
    try {
      const created = await api.addTopic(brandId, val)
      setTopics((prev) => [...prev, created])
      setNewTopic('')
    } catch (e) {
      console.error(e)
    }
  }

  const deleteTopic = async (topicId) => {
    try {
      await api.deleteTopic(topicId)
      setTopics((prev) => prev.filter((t) => t.id !== topicId))
    } catch (e) {
      console.error(e)
    }
  }

  const selectedCount = topics.filter((t) => t.is_selected).length

  const handleNext = () => {
    const selected = topics.filter((t) => t.is_selected)
    if (!selected.length) {
      setError('Please select at least one topic')
      return
    }
    onNext(selected)
  }

  return (
    <div className="step-card">
      <p className="step-hint">
        These are the topics for which we'll measure visibility. Choose which to keep, or add your own.
        In the next step, we'll choose specific prompts for each topic.
      </p>

      {loading ? (
        <div className="loading-state">
          <div className="spinner" />
          <p>Generating topics from your brand profile...</p>
        </div>
      ) : error ? (
        <div className="error-box">
          <p>{error}</p>
          <button className="btn-secondary" onClick={generateTopics}>Retry</button>
        </div>
      ) : (
        <>
          <div className="topics-header">
            <span className="topics-count">
              <span className="check-icon">✓</span> {selectedCount} topic{selectedCount !== 1 ? 's' : ''} selected
            </span>
          </div>

          <div className="topics-list">
            {topics.map((topic) => (
              <div key={topic.id} className={`topic-row ${topic.is_selected ? 'topic-selected' : ''}`}>
                <div className="topic-left">
                  <button
                    className={`checkbox ${topic.is_selected ? 'checked' : ''}`}
                    onClick={() => toggleTopic(topic)}
                  >
                    {topic.is_selected && '✓'}
                  </button>
                  {topic.is_ai_generated && <span className="ai-badge">✦</span>}
                  <span className="topic-name">{topic.name}</span>
                </div>
                <button className="icon-btn" onClick={() => deleteTopic(topic.id)}>×</button>
              </div>
            ))}
          </div>

          <div className="add-topic-row">
            <input
              className="input"
              value={newTopic}
              onChange={(e) => setNewTopic(e.target.value)}
              placeholder="Add topic"
              onKeyDown={(e) => e.key === 'Enter' && addTopic()}
            />
            <button className="btn-add" onClick={addTopic}>+</button>
          </div>
        </>
      )}

      <div className="step-footer">
        <button className="btn-secondary" onClick={onBack}>← Back</button>
        <button className="btn-primary" onClick={handleNext} disabled={loading}>
          Next →
        </button>
      </div>
    </div>
  )
}
