const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE_URL}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

// Brands
export const api = {
  // ── Brands ──────────────────────────────────────────────────────────────
  createBrand: (data) => request('POST', '/api/brands/', data),
  getBrands: () => request('GET', '/api/brands/'),
  getBrand: (id) => request('GET', `/api/brands/${id}`),
  updateBrand: (id, data) => request('PUT', `/api/brands/${id}`, data),

  // ── Topics ───────────────────────────────────────────────────────────────
  generateTopics: (brandId, apiKeys) =>
    request('POST', `/api/brands/${brandId}/generate-topics`, { brand_id: brandId, api_keys: apiKeys, count: 5 }),
  getTopics: (brandId) => request('GET', `/api/brands/${brandId}/topics`),
  addTopic: (brandId, name) =>
    request('POST', `/api/brands/${brandId}/topics`, { name, is_ai_generated: false }),
  updateTopic: (topicId, isSelected) =>
    request('PATCH', `/api/topics/${topicId}`, { is_selected: isSelected }),
  deleteTopic: (topicId) => request('DELETE', `/api/topics/${topicId}`),

  // ── Prompts ──────────────────────────────────────────────────────────────
  generatePrompts: (topicId, brandId, apiKeys) =>
    request('POST', `/api/topics/${topicId}/generate-prompts`, {
      brand_id: brandId,
      topic_id: topicId,
      api_keys: apiKeys,
      count: 5,
    }),
  getPrompts: (topicId) => request('GET', `/api/topics/${topicId}/prompts`),
  addPrompt: (topicId, text) =>
    request('POST', `/api/topics/${topicId}/prompts`, { text, is_ai_generated: false }),
  updatePrompt: (promptId, isSelected) =>
    request('PATCH', `/api/prompts/${promptId}`, { is_selected: isSelected }),
  deletePrompt: (promptId) => request('DELETE', `/api/prompts/${promptId}`),

  // ── Tracking ─────────────────────────────────────────────────────────────
  startTracking: (brandId, models, apiKeys) =>
    request('POST', '/api/tracking/start', { brand_id: brandId, models, api_keys: apiKeys }),
  getRunStatus: (runId) => request('GET', `/api/tracking/status/${runId}`),
  getRunResults: (runId) => request('GET', `/api/tracking/results/${runId}`),
  getBrandRuns: (brandId) => request('GET', `/api/tracking/brand/${brandId}/runs`),
}
