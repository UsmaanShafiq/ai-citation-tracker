import { useState } from 'react'
import Step1BrandDetails from './components/steps/Step1BrandDetails'
import Step2Topics from './components/steps/Step2Topics'
import Step3Prompts from './components/steps/Step3Prompts'
import ApiKeysStep from './components/steps/ApiKeysStep'
import Dashboard from './components/dashboard/Dashboard'
import { api } from './lib/api'
import './styles.css'

const STEPS = ['Brand Details', 'Topics', 'Prompts', 'API Keys', 'Dashboard']

export default function App() {
  const [step, setStep] = useState(0)
  const [brandId, setBrandId] = useState(null)
  const [brandName, setBrandName] = useState('')
  const [brandDomain, setBrandDomain] = useState('')
  const [selectedTopics, setSelectedTopics] = useState([])
  const [apiKeys, setApiKeys] = useState({})
  const [models, setModels] = useState([])
  const [runId, setRunId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Step 1: Create brand in DB then move to step 2
  const handleStep1 = async (formData) => {
    setLoading(true)
    setError('')
    try {
      const brand = await api.createBrand(formData)
      setBrandId(brand.id)
      setBrandName(brand.name)
      setBrandDomain(brand.domain)
      setStep(1)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Step 2: Topics selected
  const handleStep2 = (topics) => {
    setSelectedTopics(topics)
    setStep(2)
  }

  // Step 3: Prompts reviewed - go to API keys
  const handleStep3 = () => {
    setStep(3)
  }

  // API Keys step: start tracking
  const handleApiKeys = async ({ apiKeys: keys, models: selectedModels }) => {
    setApiKeys(keys)
    setModels(selectedModels)
    setLoading(true)
    setError('')
    try {
      const run = await api.startTracking(brandId, selectedModels, keys)
      setRunId(run.run_id)
      setStep(4)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleNewRun = () => {
    setStep(0)
    setBrandId(null)
    setBrandName('')
    setBrandDomain('')
    setSelectedTopics([])
    setApiKeys({})
    setModels([])
    setRunId(null)
  }

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">CitationTracker</span>
        </div>
        {step === 4 && (
          <nav className="sidebar-nav">
            <a className="nav-item active">Overview</a>
            <a className="nav-item">Competitors</a>
            <a className="nav-item">User Access</a>
          </nav>
        )}
        {step < 4 && (
          <div className="sidebar-steps">
            {STEPS.slice(0, 4).map((s, i) => (
              <div key={s} className={`sidebar-step ${i === step ? 'step-active' : ''} ${i < step ? 'step-done' : ''}`}>
                <span className="step-dot">{i < step ? '✓' : i + 1}</span>
                <span>{s}</span>
              </div>
            ))}
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="main">
        {error && (
          <div className="global-error">
            {error}
            <button onClick={() => setError('')}>×</button>
          </div>
        )}

        {loading && step !== 4 && (
          <div className="global-loading">
            <div className="spinner" /> Processing...
          </div>
        )}

        {step === 0 && <Step1BrandDetails onNext={handleStep1} />}

        {step === 1 && brandId && (
          <Step2Topics
            brandId={brandId}
            apiKeys={apiKeys}
            onNext={handleStep2}
            onBack={() => setStep(0)}
          />
        )}

        {step === 2 && (
          <Step3Prompts
            topics={selectedTopics}
            brandId={brandId}
            apiKeys={apiKeys}
            onStartTracking={handleStep3}
            onBack={() => setStep(1)}
          />
        )}

        {step === 3 && (
          <ApiKeysStep
            onNext={handleApiKeys}
            onBack={() => setStep(2)}
          />
        )}

        {step === 4 && runId && (
          <Dashboard
            runId={runId}
            brandName={brandName}
            brandDomain={brandDomain}
            onNewRun={handleNewRun}
          />
        )}
      </main>
    </div>
  )
}
