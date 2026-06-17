import { useState } from 'react'

const MODEL_COLORS = {
  ChatGPT: '#10a37f',
  Perplexity: '#6366f1',
  Gemini: '#4285f4',
  Groq: '#f55036',
  Claude: '#d97706',
}

export default function PromptDetailModal({ prompt, models, brandName, onClose }) {
  const [activeModel, setActiveModel] = useState(models[0])

  const result = prompt.results_by_model?.[activeModel]
  const linkedSites = result?.linked_sites || []
  const allBrands = result?.all_brands_detected || []
  const brandMentioned = result?.brand_mentioned
  const responseText = result?.response_text || ''
  const error = result?.error

  // Highlight brand in response text
  const highlightBrand = (text, brand) => {
    if (!brand || !text) return text
    const regex = new RegExp(`(${brand.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    return text.replace(regex, '<mark>$1</mark>')
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <h3 className="modal-prompt-text">{prompt.prompt_text}</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        {/* Model tabs */}
        <div className="modal-tabs">
          {models.map((model) => {
            const r = prompt.results_by_model?.[model]
            return (
              <button
                key={model}
                className={`modal-tab ${activeModel === model ? 'modal-tab-active' : ''}`}
                style={activeModel === model ? { borderColor: MODEL_COLORS[model], color: MODEL_COLORS[model] } : {}}
                onClick={() => setActiveModel(model)}
              >
                <span className="tab-dot" style={{ background: MODEL_COLORS[model] }} />
                {model}
                {r?.brand_mentioned && <span className="tab-badge">Brand</span>}
              </button>
            )
          })}
        </div>

        <div className="modal-body">
          {/* Left: Brand mentions */}
          <div className="modal-left">
            <h4 className="modal-section-title">Brand Mentions</h4>
            <table className="mentions-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Brand</th>
                </tr>
              </thead>
              <tbody>
                {brandMentioned ? (
                  <tr>
                    <td>#{result?.brand_position || 1}</td>
                    <td>
                      <span className="badge-brand">{brandName}</span>
                    </td>
                  </tr>
                ) : (
                  <tr>
                    <td colSpan={2} className="no-mentions">No brand mentions found.</td>
                  </tr>
                )}
              </tbody>
            </table>

            {/* AI Response text */}
            <h4 className="modal-section-title" style={{ marginTop: '1.5rem' }}>AI Response</h4>
            {error ? (
              <div className="response-error">Error: {error}</div>
            ) : (
              <div
                className="response-text"
                dangerouslySetInnerHTML={{
                  __html: highlightBrand(responseText, brandName)
                }}
              />
            )}
          </div>

          {/* Right: Linked sites */}
          <div className="modal-right">
            <h4 className="modal-section-title">Linked Sites</h4>
            {linkedSites.length > 0 ? (
              <table className="linked-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Title</th>
                    <th>URL</th>
                  </tr>
                </thead>
                <tbody>
                  {linkedSites.map((site) => (
                    <tr key={site.rank}>
                      <td>#{site.rank}</td>
                      <td>{site.title}</td>
                      <td>
                        <a href={site.url} target="_blank" rel="noreferrer" className="link">
                          {site.url.length > 40 ? site.url.slice(0, 40) + '...' : site.url}
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="no-mentions">No linked sites found.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
