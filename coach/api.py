"""HTTP API. Run: uvicorn coach.api:app --reload"""

import hmac
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
import json

from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import llm, personalization, pipeline, store
from .config import settings
from .profile import from_model, parse_profile, profile_to_text
from .schemas import CoachRequest, FeedbackRequest, RerankRequest, StyleProfile

app = FastAPI(title="Jev Dating Message Coach", version="0.1.0")
STATIC = Path(__file__).resolve().parent.parent / "public"

# endpoints that spend model credits
EXPENSIVE = {"/api/coach", "/api/coach/stream", "/api/regenerate", "/api/parse-chat-shots", "/api/parse-profile-shots",
             "/api/jev-check"}
OPEN = {"/api/health", "/api/parse-profile"}  # parse-profile is pure text rules, no model calls
_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in OPEN:
        if settings.app_password:
            given = request.headers.get("x-app-key", "")
            if not hmac.compare_digest(given.encode(), settings.app_password.encode()):
                return JSONResponse({"detail": "password required", "locked": True}, status_code=401)
        if path in EXPENSIVE and settings.rate_limit_per_hour > 0:
            q, now = _hits[client_ip(request)], time.time()
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= settings.rate_limit_per_hour:
                wait = int(3600 - (now - q[0])) // 60 + 1
                return JSONResponse({"detail": f"Rate limit reached ({settings.rate_limit_per_hour} searches/hour). Try again in ~{wait} min."},
                                    status_code=429)
            q.append(now)
    return await call_next(request)


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
            "retention_hours": settings.retention_hours, "screenshots": settings.use_claude,
            "locked": bool(settings.app_password), "rate_limit_per_hour": settings.rate_limit_per_hour}


@app.get("/api/jev-check")
async def jev_check():
    return await pipeline.jev_check()


@app.post("/api/coach")
async def coach(req: CoachRequest, x_user_id: str | None = Header(None)):
    try:
        return await pipeline.coach(req, uid(x_user_id))
    except pipeline.CoachError as e:
        _err(e)


@app.post("/api/coach/stream")
async def coach_stream(req: CoachRequest, x_user_id: str | None = Header(None)):
    """Same as /api/coach, but streams every pipeline step as newline-delimited JSON."""
    user = uid(x_user_id)

    async def events():
        try:
            async for ev in pipeline.coach_events(req, user):
                yield json.dumps(ev) + "\n"
        except pipeline.CoachError as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "error", "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


class ProfileText(BaseModel):
    text: str = ""


@app.post("/api/parse-profile")
def parse_profile_text(body: ProfileText):
    return parse_profile(body.text[:20000])


MAX_SHOTS = 10


async def read_shots(files: list[UploadFile]) -> list[tuple[bytes, str]]:
    """Validate uploaded screenshots and read them into memory (never written anywhere)."""
    if not settings.use_claude:
        raise HTTPException(400, "Reading screenshots needs ANTHROPIC_API_KEY (vision). Paste the text for now.")
    if len(files) > MAX_SHOTS:
        raise HTTPException(400, f"Up to {MAX_SHOTS} screenshots at a time")
    images, total = [], 0
    for f in files:
        mt = f.content_type or "image/jpeg"
        if mt not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise HTTPException(400, f"Unsupported image type {mt}")
        data = await f.read()
        total += len(data)
        images.append((data, mt))
    if total > 20 * 1024 * 1024:
        raise HTTPException(400, "Images too large in total (max 20MB)")
    return images


@app.post("/api/parse-profile-shots")
async def parse_profile_shots(files: list[UploadFile] = File(...), existing: str = Form("")):
    """Several profile screenshots -> one merged profile as text."""
    images = await read_shots(files)
    parsed = from_model(await llm.profile_from_screenshots(images, existing))
    return {"profile_text": profile_to_text(parsed), "profile": parsed, "images": len(images)}


@app.post("/api/parse-chat-shots")
async def parse_chat_shots(files: list[UploadFile] = File(...), existing: str = Form(""),
                           pronoun: str = Form("they")):
    """Several chat screenshots -> the whole conversation as 'Me: ... / Her: ...' text."""
    images = await read_shots(files)
    out = await llm.chat_from_screenshots(images, existing)
    name = (out.get("match_name") or "").strip().split(" ")[0][:20]
    them = "Him" if pronoun == "he" else "Her"
    msgs = [m for m in out.get("messages", []) if (m.get("text") or "").strip()]
    text = "\n".join(f"{'Me' if m['speaker'] == 'user' else them}: {' '.join(m['text'].split())}" for m in msgs)
    return {"conversation_text": text, "messages": len(msgs), "match_name": name or None, "images": len(images)}


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
