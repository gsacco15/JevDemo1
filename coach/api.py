"""HTTP API. Run: uvicorn coach.api:app --reload"""

from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import llm, personalization, pipeline, store
from .config import settings
from .schemas import CoachRequest, FeedbackRequest, RerankRequest, StyleProfile

app = FastAPI(title="Jev Dating Message Coach", version="0.1.0")
STATIC = Path(__file__).resolve().parent.parent / "static"


def uid(x_user_id: str | None) -> str:
    return (x_user_id or "default")[:64]


def _err(e: Exception):
    raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "backends": pipeline.backends(), "num_candidates": settings.num_candidates,
            "confidence_policy": {"auto": settings.conf_auto, "escalate_below": settings.conf_escalate},
            "retention_hours": settings.retention_hours, "screenshots": settings.use_claude}


@app.post("/api/coach")
async def coach(req: CoachRequest, x_user_id: str | None = Header(None)):
    try:
        return await pipeline.coach(req, uid(x_user_id))
    except pipeline.CoachError as e:
        _err(e)


@app.post("/api/rerank")
async def rerank(req: RerankRequest, x_user_id: str | None = Header(None)):
    try:
        return await pipeline.rerank(req.session_id, req.sliders, uid(x_user_id))
    except pipeline.CoachError as e:
        _err(e)


@app.post("/api/regenerate")
async def regenerate(req: RerankRequest, x_user_id: str | None = Header(None)):
    try:
        return await pipeline.regenerate(req.session_id, req.sliders, uid(x_user_id))
    except pipeline.CoachError as e:
        _err(e)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest, x_user_id: str | None = Header(None)):
    try:
        return pipeline.feedback(req.session_id, req.candidate_id, req.kind, req.edited_text, uid(x_user_id))
    except pipeline.CoachError as e:
        _err(e)


@app.post("/api/parse-screenshot")
async def parse_screenshot(file: UploadFile = File(...)):
    """Screenshot -> messages/profile. Image is processed in memory and never stored."""
    if not settings.use_claude:
        raise HTTPException(400, "Screenshot parsing needs ANTHROPIC_API_KEY (vision). Paste the text for now.")
    media_type = file.content_type or "image/png"
    if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        raise HTTPException(400, f"Unsupported image type {media_type}")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 8MB)")
    return await llm.parse_screenshot(data, media_type)


@app.get("/api/style")
def get_style(x_user_id: str | None = Header(None)):
    style, prefs = store.get_user(uid(x_user_id))
    return {"style": style or StyleProfile().model_dump(), "learned": personalization.describe(prefs),
            "feedback_counts": store.feedback_counts(uid(x_user_id))}


@app.put("/api/style")
def put_style(style: StyleProfile, x_user_id: str | None = Header(None)):
    store.save_user(uid(x_user_id), style=style.model_dump())
    return {"ok": True}


@app.delete("/api/preferences")
def reset_preferences(x_user_id: str | None = Header(None)):
    store.save_user(uid(x_user_id), prefs=personalization.empty())
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, x_user_id: str | None = Header(None)):
    store.delete_session(session_id, uid(x_user_id))
    return {"ok": True}


@app.delete("/api/me")
def delete_me(x_user_id: str | None = Header(None)):
    store.delete_user(uid(x_user_id))
    return {"ok": True}
