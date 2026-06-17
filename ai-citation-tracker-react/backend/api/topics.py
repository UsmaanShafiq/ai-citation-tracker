from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from backend.db.database import get_db
from backend.models.models import Brand, Topic, Prompt
from backend.core.generator import generate_topics, generate_prompts_for_topic
import uuid

router = APIRouter(prefix="/api", tags=["topics"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class GenerateTopicsRequest(BaseModel):
    brand_id: str
    api_keys: dict = {}
    count: int = 5


class TopicCreate(BaseModel):
    name: str
    is_ai_generated: bool = False


class TopicUpdate(BaseModel):
    is_selected: bool


class GeneratePromptsRequest(BaseModel):
    brand_id: str
    topic_id: str
    api_keys: dict = {}
    count: int = 5


class PromptCreate(BaseModel):
    text: str
    is_ai_generated: bool = False


class PromptUpdate(BaseModel):
    is_selected: bool


# ── Topic Routes ──────────────────────────────────────────────────────────────

@router.post("/brands/{brand_id}/generate-topics")
def generate_topics_for_brand(brand_id: str, req: GenerateTopicsRequest, db: Session = Depends(get_db)):
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    brand_data = {
        "name": brand.name,
        "domain": brand.domain,
        "products": brand.products or [],
        "customers": brand.customers or [],
        "key_features": brand.key_features or [],
        "business_type": brand.business_type,
        "country": brand.country,
    }

    try:
        topic_names = generate_topics(brand_data, req.api_keys, req.count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Delete old AI-generated topics first
    db.query(Topic).filter(
        Topic.brand_id == brand_id,
        Topic.is_ai_generated == True
    ).delete()
    db.commit()

    # Save new topics
    topics = []
    for name in topic_names:
        topic = Topic(
            id=str(uuid.uuid4()),
            brand_id=brand_id,
            name=name,
            is_selected=True,
            is_ai_generated=True,
        )
        db.add(topic)
        topics.append(topic)

    db.commit()
    for t in topics:
        db.refresh(t)

    return [{"id": t.id, "name": t.name, "is_selected": t.is_selected, "is_ai_generated": t.is_ai_generated} for t in topics]


@router.get("/brands/{brand_id}/topics")
def get_topics(brand_id: str, db: Session = Depends(get_db)):
    topics = db.query(Topic).filter(Topic.brand_id == brand_id).all()
    return [{"id": t.id, "name": t.name, "is_selected": t.is_selected, "is_ai_generated": t.is_ai_generated} for t in topics]


@router.post("/brands/{brand_id}/topics")
def add_topic(brand_id: str, data: TopicCreate, db: Session = Depends(get_db)):
    topic = Topic(
        id=str(uuid.uuid4()),
        brand_id=brand_id,
        name=data.name,
        is_selected=True,
        is_ai_generated=data.is_ai_generated,
    )
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return {"id": topic.id, "name": topic.name, "is_selected": topic.is_selected, "is_ai_generated": topic.is_ai_generated}


@router.patch("/topics/{topic_id}")
def update_topic(topic_id: str, data: TopicUpdate, db: Session = Depends(get_db)):
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    topic.is_selected = data.is_selected
    db.commit()
    return {"ok": True}


@router.delete("/topics/{topic_id}")
def delete_topic(topic_id: str, db: Session = Depends(get_db)):
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    db.delete(topic)
    db.commit()
    return {"ok": True}


# ── Prompt Routes ─────────────────────────────────────────────────────────────

@router.post("/topics/{topic_id}/generate-prompts")
def generate_prompts_for_topic_route(topic_id: str, req: GeneratePromptsRequest, db: Session = Depends(get_db)):
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    brand = db.query(Brand).filter(Brand.id == topic.brand_id).first()
    brand_data = {
        "name": brand.name,
        "domain": brand.domain,
        "products": brand.products or [],
        "customers": brand.customers or [],
        "business_type": brand.business_type,
    }

    try:
        prompt_texts = generate_prompts_for_topic(topic.name, brand_data, req.api_keys, req.count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Delete old AI-generated prompts for this topic
    db.query(Prompt).filter(
        Prompt.topic_id == topic_id,
        Prompt.is_ai_generated == True
    ).delete()
    db.commit()

    prompts = []
    for text in prompt_texts:
        p = Prompt(
            id=str(uuid.uuid4()),
            topic_id=topic_id,
            text=text,
            is_selected=True,
            is_ai_generated=True,
        )
        db.add(p)
        prompts.append(p)

    db.commit()
    for p in prompts:
        db.refresh(p)

    return [{"id": p.id, "text": p.text, "is_selected": p.is_selected, "is_ai_generated": p.is_ai_generated} for p in prompts]


@router.get("/topics/{topic_id}/prompts")
def get_prompts(topic_id: str, db: Session = Depends(get_db)):
    prompts = db.query(Prompt).filter(Prompt.topic_id == topic_id).all()
    return [{"id": p.id, "text": p.text, "is_selected": p.is_selected, "is_ai_generated": p.is_ai_generated} for p in prompts]


@router.post("/topics/{topic_id}/prompts")
def add_prompt(topic_id: str, data: PromptCreate, db: Session = Depends(get_db)):
    p = Prompt(
        id=str(uuid.uuid4()),
        topic_id=topic_id,
        text=data.text,
        is_selected=True,
        is_ai_generated=False,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "text": p.text, "is_selected": p.is_selected, "is_ai_generated": p.is_ai_generated}


@router.patch("/prompts/{prompt_id}")
def update_prompt(prompt_id: str, data: PromptUpdate, db: Session = Depends(get_db)):
    p = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prompt not found")
    p.is_selected = data.is_selected
    db.commit()
    return {"ok": True}


@router.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str, db: Session = Depends(get_db)):
    p = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prompt not found")
    db.delete(p)
    db.commit()
    return {"ok": True}
