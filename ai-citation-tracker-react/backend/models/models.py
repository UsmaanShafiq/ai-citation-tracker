from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from backend.db.database import Base

def gen_uuid():
    return str(uuid.uuid4())

class Brand(Base):
    __tablename__ = "brands"

    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    products = Column(JSON, default=list)       # ["Accounting Software", "Bookkeeping"]
    customers = Column(JSON, default=list)      # ["Accountants", "Bookkeepers"]
    key_features = Column(JSON, default=list)   # ["Send invoice", "Track records"]
    business_type = Column(String, default="SaaS / Software")
    country = Column(String, default="United States")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    topics = relationship("Topic", back_populates="brand", cascade="all, delete")
    tracking_runs = relationship("TrackingRun", back_populates="brand", cascade="all, delete")


class Topic(Base):
    __tablename__ = "topics"

    id = Column(String, primary_key=True, default=gen_uuid)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    name = Column(String, nullable=False)
    is_selected = Column(Boolean, default=True)
    is_ai_generated = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    brand = relationship("Brand", back_populates="topics")
    prompts = relationship("Prompt", back_populates="topic", cascade="all, delete")


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, default=gen_uuid)
    topic_id = Column(String, ForeignKey("topics.id"), nullable=False)
    text = Column(Text, nullable=False)
    is_selected = Column(Boolean, default=True)
    is_ai_generated = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    topic = relationship("Topic", back_populates="prompts")
    results = relationship("PromptResult", back_populates="prompt", cascade="all, delete")


class TrackingRun(Base):
    __tablename__ = "tracking_runs"

    id = Column(String, primary_key=True, default=gen_uuid)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_prompts = Column(Integer, default=0)
    completed_prompts = Column(Integer, default=0)

    brand = relationship("Brand", back_populates="tracking_runs")
    results = relationship("PromptResult", back_populates="run", cascade="all, delete")


class PromptResult(Base):
    __tablename__ = "prompt_results"

    id = Column(String, primary_key=True, default=gen_uuid)
    run_id = Column(String, ForeignKey("tracking_runs.id"), nullable=False)
    prompt_id = Column(String, ForeignKey("prompts.id"), nullable=False)
    model = Column(String, nullable=False)      # "ChatGPT", "Perplexity", "Gemini", etc.
    response_text = Column(Text, default="")
    brand_mentioned = Column(Boolean, default=False)
    brand_position = Column(Integer, nullable=True)  # rank if mentioned
    brand_context = Column(String, default="not_mentioned")  # recommended/mentioned/warned/not_mentioned
    linked_sites = Column(JSON, default=list)   # [{title, url, rank}]
    all_brands_detected = Column(JSON, default=list)
    error = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("TrackingRun", back_populates="results")
    prompt = relationship("Prompt", back_populates="results")
