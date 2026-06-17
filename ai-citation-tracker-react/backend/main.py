from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.db.database import engine, Base
from backend.api import brands, topics, tracking

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Citation Tracker API",
    description="Track your brand visibility across AI models",
    version="2.0.0",
)

# CORS - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(brands.router)
app.include_router(topics.router)
app.include_router(tracking.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Citation Tracker API v2"}


@app.get("/health")
def health():
    return {"status": "healthy"}
