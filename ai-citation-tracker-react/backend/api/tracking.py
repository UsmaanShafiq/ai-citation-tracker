from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel
from backend.db.database import get_db
from backend.models.models import Brand, Topic, Prompt, TrackingRun, PromptResult
from backend.core.ai_runner import run_prompt_on_model
from backend.core.brand_detector import detect_brand
from backend.core.generator import calculate_visibility
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/tracking", tags=["tracking"])


class StartTrackingRequest(BaseModel):
    brand_id: str
    models: list = ["ChatGPT", "Perplexity", "Gemini"]
    api_keys: dict = {}


def _run_tracking_job(run_id: str, brand_id: str, models: list, api_keys: dict):
    """Background job that runs all prompts across all models."""
    from backend.db.database import SessionLocal
    db = SessionLocal()

    try:
        run = db.query(TrackingRun).filter(TrackingRun.id == run_id).first()
        if not run:
            return

        run.status = "running"
        db.commit()

        brand = db.query(Brand).filter(Brand.id == brand_id).first()

        # Get all selected prompts across all selected topics
        topics = db.query(Topic).filter(
            Topic.brand_id == brand_id,
            Topic.is_selected == True
        ).all()

        all_prompts = []
        for topic in topics:
            prompts = db.query(Prompt).filter(
                Prompt.topic_id == topic.id,
                Prompt.is_selected == True
            ).all()
            all_prompts.extend(prompts)

        run.total_prompts = len(all_prompts) * len(models)
        db.commit()

        # Run each prompt on each model
        for prompt in all_prompts:
            for model_name in models:
                try:
                    response_text = run_prompt_on_model(prompt.text, model_name, api_keys)

                    is_error = response_text.startswith("ERROR:")
                    error_msg = response_text.replace("ERROR:", "").strip() if is_error else None

                    detection = {}
                    if not is_error:
                        detection = detect_brand(response_text, brand.name, brand.domain)

                    result = PromptResult(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        prompt_id=prompt.id,
                        model=model_name,
                        response_text=response_text if not is_error else "",
                        brand_mentioned=detection.get("brand_mentioned", False),
                        brand_position=detection.get("brand_position"),
                        brand_context=detection.get("brand_context", "not_mentioned"),
                        linked_sites=detection.get("linked_sites", []),
                        all_brands_detected=detection.get("all_brands_detected", []),
                        error=error_msg,
                    )
                    db.add(result)

                    run.completed_prompts += 1
                    db.commit()

                except Exception as e:
                    result = PromptResult(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        prompt_id=prompt.id,
                        model=model_name,
                        response_text="",
                        brand_mentioned=False,
                        brand_context="not_mentioned",
                        linked_sites=[],
                        all_brands_detected=[],
                        error=str(e),
                    )
                    db.add(result)
                    run.completed_prompts += 1
                    db.commit()

        run.status = "completed"
        run.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        run = db.query(TrackingRun).filter(TrackingRun.id == run_id).first()
        if run:
            run.status = "failed"
            db.commit()
    finally:
        db.close()


@router.post("/start")
def start_tracking(req: StartTrackingRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    brand = db.query(Brand).filter(Brand.id == req.brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    run = TrackingRun(
        id=str(uuid.uuid4()),
        brand_id=req.brand_id,
        status="pending",
        completed_prompts=0,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(
        _run_tracking_job,
        run.id,
        req.brand_id,
        req.models,
        req.api_keys,
    )

    return {"run_id": run.id, "status": "pending"}


@router.get("/status/{run_id}")
def get_run_status(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TrackingRun).filter(TrackingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    progress = 0
    if run.total_prompts > 0:
        progress = round((run.completed_prompts / run.total_prompts) * 100)

    return {
        "run_id": run.id,
        "status": run.status,
        "total_prompts": run.total_prompts,
        "completed_prompts": run.completed_prompts,
        "progress": progress,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


@router.get("/results/{run_id}")
def get_run_results(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TrackingRun).filter(TrackingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    results = db.query(PromptResult).options(
        joinedload(PromptResult.prompt).joinedload(Prompt.topic)
    ).filter(PromptResult.run_id == run_id).all()

    brand = db.query(Brand).filter(Brand.id == run.brand_id).first()

    # Build results grouped by topic then prompt
    topics_map = {}
    for r in results:
        if not r.prompt or not r.prompt.topic:
            continue
        topic_name = r.prompt.topic.name
        topic_id = r.prompt.topic.id
        prompt_text = r.prompt.text
        prompt_id = r.prompt_id

        if topic_id not in topics_map:
            topics_map[topic_id] = {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "prompts": {},
                "visibility_by_model": {},
            }

        if prompt_id not in topics_map[topic_id]["prompts"]:
            topics_map[topic_id]["prompts"][prompt_id] = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "results_by_model": {},
            }

        topics_map[topic_id]["prompts"][prompt_id]["results_by_model"][r.model] = {
            "brand_mentioned": r.brand_mentioned,
            "brand_context": r.brand_context,
            "brand_position": r.brand_position,
            "response_text": r.response_text,
            "linked_sites": r.linked_sites or [],
            "all_brands_detected": r.all_brands_detected or [],
            "error": r.error,
        }

    # Calculate visibility per topic per model
    models = list(set(r.model for r in results))
    for topic_id, topic_data in topics_map.items():
        for model in models:
            topic_results = [
                r for r in results
                if r.prompt and r.prompt.topic and r.prompt.topic.id == topic_id and r.model == model
            ]
            mentions = sum(1 for r in topic_results if r.brand_mentioned)
            total = len(topic_results)
            topic_data["visibility_by_model"][model] = {
                "pct": round((mentions / total) * 100) if total > 0 else 0,
                "mentions": mentions,
                "total": total,
            }

    # Overall visibility per model
    overall_by_model = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        mentions = sum(1 for r in model_results if r.brand_mentioned)
        total = len(model_results)
        overall_by_model[model] = {
            "pct": round((mentions / total) * 100) if total > 0 else 0,
            "mentions": mentions,
            "total": total,
        }

    # Convert prompts dict to list
    for topic_data in topics_map.values():
        topic_data["prompts"] = list(topic_data["prompts"].values())

    return {
        "run_id": run_id,
        "brand": {"name": brand.name, "domain": brand.domain},
        "status": run.status,
        "models": models,
        "overall_by_model": overall_by_model,
        "topics": list(topics_map.values()),
        "completed_at": run.completed_at,
    }


@router.get("/brand/{brand_id}/runs")
def get_brand_runs(brand_id: str, db: Session = Depends(get_db)):
    runs = db.query(TrackingRun).filter(
        TrackingRun.brand_id == brand_id
    ).order_by(TrackingRun.started_at.desc()).all()

    return [
        {
            "run_id": r.id,
            "status": r.status,
            "started_at": r.started_at,
            "completed_at": r.completed_at,
            "total_prompts": r.total_prompts,
            "completed_prompts": r.completed_prompts,
        }
        for r in runs
    ]
