# AI Citation Tracker v2

Track your brand visibility across AI models (ChatGPT, Perplexity, Gemini, Groq, Claude).
Replicated from traqer.ai with a full SaaS-ready architecture.

---

## Stack

- **Frontend**: React + Vite (deploy on Vercel)
- **Backend**: FastAPI + Python (deploy on Railway)
- **Database**: PostgreSQL (on Railway)

---

## Project Structure

```
ai-citation-tracker/
├── backend/
│   ├── api/
│   │   ├── brands.py        # Brand CRUD endpoints
│   │   ├── topics.py        # Topic + prompt generation endpoints
│   │   └── tracking.py      # Run tracking + results endpoints
│   ├── core/
│   │   ├── ai_runner.py     # Runs prompts on each AI model
│   │   ├── brand_detector.py # Detects brand mentions + extracts links
│   │   └── generator.py     # AI topic + prompt generation
│   ├── db/
│   │   └── database.py      # SQLAlchemy session + engine
│   ├── models/
│   │   └── models.py        # All database table definitions
│   ├── config.py            # App settings from .env
│   ├── main.py              # FastAPI app entry point
│   ├── requirements.txt
│   ├── railway.toml         # Railway deployment config
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── steps/
    │   │   │   ├── Step1BrandDetails.jsx
    │   │   │   ├── Step2Topics.jsx
    │   │   │   ├── Step3Prompts.jsx
    │   │   │   └── ApiKeysStep.jsx
    │   │   └── dashboard/
    │   │       ├── Dashboard.jsx
    │   │       └── PromptDetailModal.jsx
    │   ├── lib/
    │   │   └── api.js        # All API calls to backend
    │   ├── App.jsx           # Main app with step flow
    │   ├── styles.css        # Global styles
    │   └── main.jsx
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── .env.example
```

---

## Local Development Setup

### 1. Clone the repo

```bash
git clone https://github.com/UsmaanShafiq/ai-citation-tracker.git
cd ai-citation-tracker
```

### 2. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file and fill in values
cp .env.example .env
# Edit .env with your DATABASE_URL and API keys

# Run the backend
uvicorn backend.main:app --reload --port 8000
```

Backend will be live at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 3. Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Copy env file
cp .env.example .env
# Leave VITE_API_URL empty for local dev (Vite proxy handles it)

# Run the frontend
npm run dev
```

Frontend will be live at: http://localhost:5173

### 4. Database

For local development, install PostgreSQL and create a database:

```sql
CREATE DATABASE citation_tracker;
```

Tables are created automatically when the backend starts.

---

## Deployment

### Backend on Railway

1. Go to railway.app and create a new project
2. Add a PostgreSQL database service
3. Add a new service from your GitHub repo
4. Set the root directory to `/backend`
5. Add environment variables (copy from .env.example):
   - `DATABASE_URL` (Railway auto-fills this from the PostgreSQL service)
   - `SECRET_KEY`
   - Any AI API keys you want as server-side fallbacks
   - `CORS_ORIGINS` with your Vercel frontend URL
6. Deploy. Railway uses the `railway.toml` config automatically.

### Frontend on Vercel

1. Go to vercel.com and import your GitHub repo
2. Set the root directory to `/frontend`
3. Add environment variable:
   - `VITE_API_URL` = your Railway backend URL (e.g. https://your-app.railway.app)
4. Deploy.

---

## How It Works

### User Flow

1. **Step 1 - Brand Details**: User enters brand name, domain, products, customers, key features, business type
2. **Step 2 - Topics**: AI generates 5 relevant search topics from the brand profile. User can add/remove.
3. **Step 3 - Prompts**: AI generates 5 prompts per topic (25 total). User reviews and edits.
4. **API Keys**: User enters their own API keys and selects which models to track
5. **Dashboard**: Results show visibility % per model, Brand badges per prompt per model, linked sites

### Detection Logic

For each prompt on each model:
- Response text is scanned for brand name and domain
- If found: shows "Brand" badge, records position and context
- If not found: shows dash
- All URLs in the response are extracted as "Linked Sites"
- Visibility % = (prompts where brand was mentioned / total prompts) x 100

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/brands/ | Create a brand |
| GET | /api/brands/ | List all brands |
| GET | /api/brands/{id} | Get a brand |
| POST | /api/brands/{id}/generate-topics | AI generate topics |
| GET | /api/brands/{id}/topics | Get topics for a brand |
| POST | /api/brands/{id}/topics | Add a topic manually |
| PATCH | /api/topics/{id} | Toggle topic selected |
| POST | /api/topics/{id}/generate-prompts | AI generate prompts |
| GET | /api/topics/{id}/prompts | Get prompts for a topic |
| POST | /api/topics/{id}/prompts | Add a prompt manually |
| PATCH | /api/prompts/{id} | Toggle prompt selected |
| POST | /api/tracking/start | Start a tracking run |
| GET | /api/tracking/status/{run_id} | Poll run progress |
| GET | /api/tracking/results/{run_id} | Get full results |
| GET | /api/tracking/brand/{brand_id}/runs | Get all runs for a brand |

---

## Adding New AI Models

In `backend/core/ai_runner.py`, add a new function and register it in `ALL_MODELS`:

```python
def run_my_model(query: str, api_key: str) -> str:
    # your implementation
    return response_text

ALL_MODELS["MyModel"] = {"fn": run_my_model, "key_env": "MY_MODEL_API_KEY"}
```

Then add it to the frontend `ApiKeysStep.jsx` MODELS array.
