import { useState } from 'react'

const MODELS = [
  { key: 'ChatGPT', label: 'ChatGPT (OpenAI)', placeholder: 'sk-...', required: false },
  { key: 'Perplexity', label: 'Perplexity', placeholder: 'pplx-...', required: false },
  { key: 'Gemini', label: 'Gemini (Google)', placeholder: 'AI...', required: false },
  { key: 'Groq', label: 'Groq (Free)', placeholder: 'gsk_...', required: false },
  { key: 'Claude', label: 'Claude (Anthropic)', placeholder: 'sk-ant-...', required: false },
]

export default function ApiKeysStep({ onNext, onBack }) {
  const [keys, setKeys] = useState({})
  const [showKeys, setShowKeys] = useState({})
  const [selectedModels, setSelectedModels] = useState(['ChatGPT', 'Perplexity', 'Gemini'])

  const setKey = (model, val) => setKeys((k) => ({ ...k, [model]: val }))
  const toggleShow = (model) => setShowKeys((s) => ({ ...s, [model]: !s[model] }))
  const toggleModel = (model) => {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    )
  }

  const handleNext = () => {
    const activeKeys = {}
    selectedModels.forEach((m) => {
      if (keys[m]) activeKeys[m] = keys[m]
    })
    onNext({ apiKeys: activeKeys, models: selectedModels })
  }

  const hasAtLeastOneKey = selectedModels.some((m) => keys[m])

  return (
    <div className="step-card">
      <h2 className="step-title">Configure AI Models</h2>
      <p className="step-subtitle">
        Select which AI models to track and add your API keys. At least one key is required to run tracking.
      </p>

      <div className="api-keys-list">
        {MODELS.map((model) => (
          <div key={model.key} className={`api-key-row ${selectedModels.includes(model.key) ? 'key-row-active' : ''}`}>
            <div className="key-row-left">
              <button
                className={`checkbox ${selectedModels.includes(model.key) ? 'checked' : ''}`}
                onClick={() => toggleModel(model.key)}
              >
                {selectedModels.includes(model.key) && '✓'}
              </button>
              <label className="key-label">{model.label}</label>
              {model.key === 'Groq' && <span className="free-badge">Free</span>}
            </div>
            {selectedModels.includes(model.key) && (
              <div className="key-input-wrap">
                <input
                  type={showKeys[model.key] ? 'text' : 'password'}
                  className="input"
                  value={keys[model.key] || ''}
                  onChange={(e) => setKey(model.key, e.target.value)}
                  placeholder={model.placeholder}
                />
                <button className="toggle-show" onClick={() => toggleShow(model.key)}>
                  {showKeys[model.key] ? '🙈' : '👁'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      <p className="keys-note">
        Your API keys are sent directly to the models and never stored in our database.
      </p>

      <div className="step-footer">
        <button className="btn-secondary" onClick={onBack}>← Back</button>
        <button
          className="btn-primary"
          onClick={handleNext}
          disabled={!hasAtLeastOneKey}
          title={!hasAtLeastOneKey ? 'Add at least one API key' : ''}
        >
          Start Tracking →
        </button>
      </div>
    </div>
  )
}
