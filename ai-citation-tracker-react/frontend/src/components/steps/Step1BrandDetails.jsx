import { useState } from 'react'

const BUSINESS_TYPES = [
  'SaaS / Software',
  'Service Business',
  'Other / Not Sure',
  'Ecommerce / DTC Brand',
  'Marketplace / Aggregator',
]

const COUNTRIES = [
  'United States', 'United Kingdom', 'Canada', 'Australia',
  'Germany', 'France', 'India', 'Pakistan', 'Global',
]

function TagInput({ label, hint, tags, onChange, placeholder }) {
  const [input, setInput] = useState('')

  const addTag = () => {
    const val = input.trim()
    if (val && !tags.includes(val)) {
      onChange([...tags, val])
    }
    setInput('')
  }

  const removeTag = (tag) => onChange(tags.filter((t) => t !== tag))

  return (
    <div className="field">
      <label className="field-label">{label}</label>
      {hint && <p className="field-hint">{hint}</p>}
      <div className="tag-input-box">
        {tags.map((tag) => (
          <span key={tag} className="tag">
            {tag}
            <button onClick={() => removeTag(tag)} className="tag-remove">×</button>
          </span>
        ))}
        <input
          className="tag-input"
          value={input}
          placeholder={placeholder}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ',') {
              e.preventDefault()
              addTag()
            }
          }}
        />
      </div>
    </div>
  )
}

export default function Step1BrandDetails({ onNext }) {
  const [form, setForm] = useState({
    name: '',
    domain: '',
    products: [],
    customers: [],
    key_features: [],
    business_type: 'SaaS / Software',
    country: 'United States',
  })
  const [errors, setErrors] = useState({})

  const set = (key, val) => setForm((f) => ({ ...f, [key]: val }))

  const validate = () => {
    const e = {}
    if (!form.name.trim()) e.name = 'Brand name is required'
    if (!form.domain.trim()) e.domain = 'Domain is required'
    return e
  }

  const handleNext = () => {
    const e = validate()
    if (Object.keys(e).length) { setErrors(e); return }
    onNext(form)
  }

  return (
    <div className="step-card">
      <h2 className="step-title">Brand Details</h2>
      <p className="step-subtitle">Enter your brand name, URL and topics you want to track</p>

      <div className="field-row">
        <div className="field">
          <label className="field-label">Brand name you would like to track <span className="required">*</span></label>
          <input
            className={`input ${errors.name ? 'input-error' : ''}`}
            value={form.name}
            onChange={(e) => set('name', e.target.value)}
            placeholder="e.g. Concurate"
          />
          {errors.name && <p className="error-msg">{errors.name}</p>}
        </div>

        <div className="field">
          <label className="field-label">Brand's domain/URL <span className="required">*</span></label>
          <input
            className={`input ${errors.domain ? 'input-error' : ''}`}
            value={form.domain}
            onChange={(e) => set('domain', e.target.value)}
            placeholder="e.g. concurate.com"
          />
          {errors.domain && <p className="error-msg">{errors.domain}</p>}
        </div>
      </div>

      <TagInput
        label="Your products and services"
        hint="List all possible ways customers may describe them"
        tags={form.products}
        onChange={(v) => set('products', v)}
        placeholder="Type and press Enter..."
      />

      <TagInput
        label="Your customers"
        hint="Briefly list your different ideal customer personas"
        tags={form.customers}
        onChange={(v) => set('customers', v)}
        placeholder="Type and press Enter..."
      />

      <TagInput
        label="Key Features"
        hint="List all important features, benefits and differentiators"
        tags={form.key_features}
        onChange={(v) => set('key_features', v)}
        placeholder="Type and press Enter..."
      />

      <div className="field">
        <label className="field-label">Business Type</label>
        <p className="field-hint">Which best describes your business?</p>
        <div className="radio-grid">
          {BUSINESS_TYPES.map((type) => (
            <label key={type} className={`radio-option ${form.business_type === type ? 'radio-selected' : ''}`}>
              <input
                type="radio"
                name="business_type"
                value={type}
                checked={form.business_type === type}
                onChange={() => set('business_type', type)}
              />
              {type}
            </label>
          ))}
        </div>
      </div>

      <div className="step-footer">
        <select className="select" value={form.country} onChange={(e) => set('country', e.target.value)}>
          {COUNTRIES.map((c) => <option key={c}>{c}</option>)}
        </select>
        <button className="btn-primary" onClick={handleNext}>
          Next →
        </button>
      </div>
    </div>
  )
}
