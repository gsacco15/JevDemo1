"""The orchestrator: understand state -> choose strategy -> search many messages -> judge -> rank -> top 3."""

import asyncio
import logging
import re
import time
import uuid
from collections import Counter

from . import explain, llm, personalization, ranking, store, templates
from .config import settings
from .jev import EvalItem, HeuristicJudge, make_judge
from .jev import heuristics as H
from .parsing import parse_conversation, render_conversation
from .questions import (RISK_DIMS, CANDIDATE_QUESTIONS, HOOK_APPEARANCE, HOOK_QUESTION, PAIRWISE_QUESTION,
                        REASONING_DIMS, STATE_QUESTIONS, STRATEGY_QUESTION)
from .schemas import (Candidate, CoachRequest, ConversationState, Hook, Message, Sliders, StyleProfile)

log = logging.getLogger(__name__)

judge = make_judge()
_local = HeuristicJudge()


REJECT_NOULS = {"manipulative", "insulting", "pickup_line", "invented_info", "repeats", "asks_known_info"}
DECISION_SCORES = {"relevance", "cringe", "neediness", "sexual"}


class CoachError(ValueError):
    pass


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.stages: dict[str, float] = {}

    def lap(self, name: str):
        now = time.perf_counter()
        self.stages[name] = round((now - self.last) * 1000, 1)
        self.last = now

    def total(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)


async def jev_check() -> dict:
    """One tiny real call so you can confirm Jev is wired up (see /api/jev-check)."""
    from .jev.types import noul as _noul
    if judge.name != "jev":
        return {"ok": False, "judge": judge.name, "reason": "No TYPESAFE_API_KEY (or JEV_API_KEY) set - using the local stand-in."}
    q = _noul("urgent", "`message` conveys urgency.")
    t0 = time.perf_counter()
    before = judge.stats.errors
    res = await judge.judge([EvalItem("check", {"message": "Please help ASAP, I'm losing sales!"}, {"kind": "candidate", "candidate": ""})], [q])
    j = res["check"].get("urgent")
    ok = judge.stats.errors == before and j is not None and j.source == "jev"
    return {"ok": ok, "judge": "jev", "model": judge.stats.model_version, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "answer_p_urgent": j.value if j else None, "error": None if ok else judge.stats.last_error,
            "state_sent_as": "string" if getattr(judge, "_state_as_string", False) else "object"}


def backends() -> dict:
    return {
        "judge": judge.name,
        "generator": "claude:" + settings.generator_model if settings.use_claude else "templates",
        "reasoning": "claude:" + settings.reasoning_model if settings.use_claude else "off",
    }


# ---------------------------------------------------------------------------
# rendering app state for Jev
# ---------------------------------------------------------------------------

def _msgs_dicts(messages: list[Message]) -> list[dict]:
    return [m.model_dump() for m in messages]


def state_summary(state: ConversationState) -> dict:
    return state.model_dump(exclude={"hooks", "platform"})


ROLES = {"user": "the person we are helping write their next message", "match": "the person they matched with"}


def render_context(mode: str, messages: list[Message], profile: str | None, state: ConversationState | None,
                   style: StyleProfile | None, strategy: str | None = None) -> dict:
    """The state object Jev sees. Questions in questions.py refer to these field names."""
    ctx: dict = {"app": "dating app (Hinge)", "roles": ROLES, "mode": mode,
                 "conversation": _msgs_dicts(messages[-20:]), "match_profile": profile}
    if state:
        ctx["conversation_state"] = {k: v for k, v in state_summary(state).items() if k != "last_match_message"}
    if strategy:
        ctx["chosen_move"] = strategy
    if style:
        ctx["user_style"] = style.model_dump(exclude={"sample_messages"}) | {
            "example_messages_the_user_sent": style.sample_messages[:8]}
    return ctx


# ---------------------------------------------------------------------------
# 1. state
# ---------------------------------------------------------------------------

def split_profile(profile: str) -> list[tuple[str, str]]:
    out = []
    for line in re.split(r"[\n]+|(?<=[.!?])\s+", profile or ""):
        line = line.strip(" -•\t")
        if len(line.split()) < 2:
            continue
        src = "photo" if line.lower().startswith(("photo", "pic", "image")) else "prompt"
        out.append((src, re.sub(r"^(photo|pic|image)\s*[:\-]\s*", "", line, flags=re.I)))
    return out[:12]


async def extract_hooks(profile: str, context: str) -> tuple[list[Hook], int]:
    pieces = split_profile(profile)
    if not pieces:
        return [], 0
    items = [EvalItem(key=f"h{i}", state=context | {"profile_detail": t}, data={"kind": "hook", "hook_text": t})
             for i, (_, t) in enumerate(pieces)]
    res = await judge.judge(items, [HOOK_QUESTION, HOOK_APPEARANCE])
    hooks = []
    for (src, text), item in zip(pieces, items):
        j = res[item.key]
        strength = j["hook_strength"].value if "hook_strength" in j else 0.3
        appearance = j.get("hook_is_appearance") and j["hook_is_appearance"].value > 0.5
        if appearance:
            strength *= 0.4  # strongly penalise appearance-only hooks when better ones exist
        subject = templates.extract_topic(text) or text[:40]
        hooks.append(Hook(source=src, subject=subject, text=text, strength=round(strength, 3), is_appearance=bool(appearance)))
    hooks.sort(key=lambda h: -h.strength)
    return hooks, len(items) * 2


async def build_state(mode: str, messages: list[Message], profile: str | None, style: StyleProfile,
                      platform: str) -> tuple[ConversationState, dict, int]:
    md = _msgs_dicts(messages)
    ctx = render_context(mode, messages, profile, None, style)
    res = (await judge.judge([EvalItem("state", ctx, {"kind": "state", "messages": md, "profile": profile, "mode": mode})],
                             STATE_QUESTIONS))["state"]
    n_judgments = len(res)

    recent = messages[-6:]
    user_text = " ".join(m.text for m in messages if m.speaker == "user")
    word_counts = Counter(w for m in messages for w in H.content_words(m.text) if len(w) > 3)
    topics = [w for w, c in word_counts.most_common(6) if c >= 2][:4]
    for m in messages:
        topics += [p for p in H.proper_nouns(" " + m.text) if p not in topics]
    last_match = next((m.text for m in reversed(messages) if m.speaker == "match"), None)

    tone_probs = res["tone"].probs if "tone" in res else {}
    tones = [t for t, p in sorted(tone_probs.items(), key=lambda kv: -kv[1]) if p > 0.2][:2] or ([res["tone"].choice] if "tone" in res else [])

    def val(k, d=0.5):
        return res[k].value if k in res else d

    state = ConversationState(
        platform=platform,
        stage=res["stage"].choice if "stage" in res else "EARLY_CONVERSATION",
        message_count=len(messages),
        match_engagement=round(val("match_engagement"), 3),
        user_engagement=round(val("user_engagement"), 3),
        tone=tones,
        match_behavior=res["match_behavior"].choice if "match_behavior" in res else "sharing",
        topics=topics[:6],
        questions_recently_asked_by_user=sum(H.is_question(m.text) for m in recent if m.speaker == "user"),
        questions_recently_asked_by_match=sum(H.is_question(m.text) for m in recent if m.speaker == "match"),
        flirt_level=round(val("flirt_level", 0.3), 3),
        escalation_readiness=round(val("escalation_readiness", 0.3), 3),
        need_question=round(val("need_question"), 3),
        number_requested=H.hits(user_text, H.NUMBER) > 0,
        date_requested=H.hits(user_text, H.DATE) > 0,
        number_appropriate=round(val("number_appropriate", 0.2), 3),
        date_appropriate=round(val("date_appropriate", 0.2), 3),
        conversation_momentum=res["momentum"].choice if "momentum" in res else "neutral",
        last_match_message=last_match,
    )
    if mode == "opener":
        state.stage = "PROFILE_OPENER"
    if profile:
        state.hooks, hj = await extract_hooks(profile, ctx)
        n_judgments += hj
    raw = {k: j.model_dump(include={"choice", "value", "confidence", "source"}) for k, j in res.items()}
    return state, raw, n_judgments


# ---------------------------------------------------------------------------
# 2. strategy (Jev choice + programmatic rules)
# ---------------------------------------------------------------------------

async def choose_strategy(mode: str, messages, profile, state: ConversationState, style) -> tuple[str, list[str], dict]:
    ctx = render_context(mode, messages, profile, state, style)
    res = (await judge.judge([EvalItem("strategy", ctx, {"kind": "strategy", "state": state_summary(state), "mode": mode})],
                             [STRATEGY_QUESTION]))["strategy"]["strategy"]
    probs = dict(res.probs)
    rules = []
    if state.number_requested:
        probs["ASK_FOR_NUMBER"] = 0
        rules.append("number already requested -> no ASK_FOR_NUMBER")
    if state.date_requested and state.stage != "DATE_PLANNING":
        probs["ASK_FOR_DATE"] *= 0.3
    if state.message_count < 6:
        for k in ("ASK_FOR_NUMBER", "ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE"):
            probs[k] = probs.get(k, 0) * 0.3
        rules.append("early conversation -> damp number/date asks")
    trailing_user = 0
    for m in reversed(messages):
        if m.speaker != "user":
            break
        trailing_user += 1
    if trailing_user >= 2:
        probs["WAIT"] = probs.get("WAIT", 0) + 0.5
        probs["PULL_BACK"] = probs.get("PULL_BACK", 0) + 0.3
        rules.append("user already double-texted -> favour WAIT / PULL_BACK")
    if mode == "opener":
        for k in list(probs):
            if k not in ("TEASE", "ASK_QUESTION", "CONTINUE_TOPIC", "FLIRT"):
                probs[k] = 0
    z = sum(probs.values()) or 1
    probs = {k: v / z for k, v in probs.items()}
    ranked = sorted(probs, key=lambda k: -probs[k])
    chosen = ranked[0]
    alts = [s for s in ranked[1:3] if probs[s] > 0.03]
    bold_alt = "ASK_FOR_DATE" if state.date_appropriate >= state.number_appropriate else "ASK_FOR_NUMBER"
    if mode == "reply" and state.escalation_readiness > 0.45 and bold_alt not in alts and bold_alt != chosen and probs.get(bold_alt, 0) > 0:
        alts.append(bold_alt)
    info = {"chosen": chosen, "confidence": round(res.confidence, 3), "source": res.source,
            "probs": {k: round(probs[k], 3) for k in ranked[:6]}, "rules_applied": rules}
    return chosen, alts, info


# ---------------------------------------------------------------------------
# 3. generation
# ---------------------------------------------------------------------------

def _gen_ctx(mode, messages, profile, state, strategy, alts, style, sliders, n, pronoun, platform, avoid) -> dict:
    return {
        "mode": mode, "platform": platform, "pronoun": pronoun, "profile": profile,
        "conversation": render_conversation(messages) if messages else "",
        "state_summary": {k: v for k, v in state_summary(state).items() if k != "last_match_message"},
        "strategy": strategy, "alt_strategies": alts, "style": style.model_dump(),
        "sliders": sliders.model_dump(), "n": n, "hooks": [h.model_dump() for h in state.hooks],
        "last_match": state.last_match_message, "avoid_texts": avoid,
    }


def _batch_plan(strategy: str, alts: list[str], n: int) -> list[tuple[str, str, int]]:
    """Split generation into parallel batches, each exploring a different region of message space."""
    per = max(4, -(-n // 3))
    alt = ", ".join(alts) or "other moves that could work"
    return [
        ("balanced", f"the recommended move ({strategy}) done well, balanced boldness", per),
        ("bold", f"bolder options: more confident or forward takes on {strategy}, plus {alt}", per),
        ("chill", f"lower-risk options: relaxed, short, easy to reply to; {strategy} or {alt}", n - 2 * per if n - 2 * per >= 4 else per),
    ]


async def generate_batches(mode, messages, profile, state, strategy, alts, style, sliders, n, pronoun, platform,
                           avoid: list[str]):
    """Yields (batch_name, raw_candidates, note) as each batch finishes."""
    base = _gen_ctx(mode, messages, profile, state, strategy, alts, style, sliders, n, pronoun, platform, avoid)
    if not settings.use_claude:
        yield "templates", templates.generate(base, seed=len(avoid)), None
        return

    async def one(name, focus, k):
        try:
            return name, await llm.generate_candidates(base | {"n": k, "focus": focus}), None
        except Exception as e:  # noqa: BLE001 - keep the demo alive
            log.exception("Claude generation batch %s failed", name)
            return name, templates.generate(base | {"n": k}, seed=hash(name) % 1000), \
                f"Claude batch '{name}' failed ({type(e).__name__}); used templates for it"

    tasks = [asyncio.create_task(one(*b)) for b in _batch_plan(strategy, alts, n)]
    for fut in asyncio.as_completed(tasks):
        yield await fut


async def generate(mode, messages, profile, state, strategy, alts, style, sliders, n, pronoun, platform,
                   avoid: list[str]) -> tuple[list[dict], str | None]:
    raw, notes = [], []
    async for _, batch, note in generate_batches(mode, messages, profile, state, strategy, alts, style, sliders, n,
                                                 pronoun, platform, avoid):
        raw += batch
        if note:
            notes.append(note)
    return raw, "; ".join(notes) or None


def make_candidates(raw: list[dict], start: int = 0) -> list[Candidate]:
    out, seen = [], set()
    for i, r in enumerate(raw):
        text = (r.get("text") or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(Candidate(
            id=f"c{start + i}", text=text, intended_strategy=r.get("strategy"), intended_boldness=r.get("boldness"),
            features={"words": len(text.split()), "emojis": H.emoji_count(text), "question_marks": text.count("?")},
        ))
    return out


# ---------------------------------------------------------------------------
# 4. judging candidates
# ---------------------------------------------------------------------------

def candidate_items(cands: list[Candidate], mode, messages, profile, state, strategy, style) -> list[EvalItem]:
    ctx = render_context(mode, messages, profile, state, style, strategy)
    hook = state.hooks[0].text if mode == "opener" and state.hooks else None
    base = {"kind": "candidate", "messages": _msgs_dicts(messages), "profile": profile, "last_match": state.last_match_message,
            "state": state_summary(state), "style": style.model_dump(), "strategy": strategy, "hook": hook}
    return [EvalItem(c.id, ctx | {"candidate_message": c.text}, base | {"candidate": c.text}) for c in cands]


def apply_judgments(c: Candidate, res: dict) -> int:
    c.judgments = res
    for dim in ranking.TONE_DIMS:
        c.features[dim] = round(res[dim].value, 3) if dim in res else 0.5
    return len(res)


async def judge_candidates(cands: list[Candidate], mode, messages, profile, state, strategy, style) -> int:
    items = candidate_items(cands, mode, messages, profile, state, strategy, style)
    res = await judge.judge(items, CANDIDATE_QUESTIONS)
    return sum(apply_judgments(c, res[c.id]) for c in cands)


async def escalate(cands: list[Candidate], context: dict) -> tuple[int, int]:
    """System Two: re-ask low-confidence judgments on contenders. Returns (low_conf_count, escalated_count)."""
    # Only escalate judgments that can change the outcome (TypeSafe guidance: low confidence on a
    # harmless preference is fine; ignore uncertainty that doesn't affect a decision).
    asks = []
    for c in cands:
        for qid, j in c.judgments.items():
            q = REASONING_DIMS.get(qid)
            if not q:
                continue
            if j.kind == "noul":
                unsure = qid in REJECT_NOULS and 0.45 < j.value < 0.8  # near a rejection threshold
            else:
                unsure = qid in DECISION_SCORES and j.confidence < settings.conf_escalate
            if unsure:
                asks.append({"id": f"{c.id}:{qid}", "candidate": c.text, "question": q.instructions,
                             "levels": list(q.levels) if q.kind == "score" else None, "kind": j.kind})
    if not asks or not settings.use_claude:
        return len(asks), 0
    try:
        answers = await llm.resolve_low_confidence(context, asks[:settings.escalation_cap])
    except Exception:  # noqa: BLE001
        log.exception("reasoning escalation failed")
        return len(asks), 0
    by_id = {c.id: c for c in cands}
    for key, value in answers.items():
        cid, qid = key.split(":", 1)
        if cid in by_id and qid in by_id[cid].judgments:
            j = by_id[cid].judgments[qid]
            j.value, j.source, j.confidence = value, "reasoning", 0.9
            if qid in ranking.TONE_DIMS:
                by_id[cid].features[qid] = round(value, 3)
    return len(asks), len(answers)


# ---------------------------------------------------------------------------
# 5. ranking
# ---------------------------------------------------------------------------

async def pairwise_probs(finalists: list[Candidate], context: dict, use_judge: bool) -> dict[tuple[str, str], float]:
    pairs = [(a, b) for i, a in enumerate(finalists) for b in finalists[i + 1:]]
    if not pairs:
        return {}
    items = [EvalItem(f"{a.id}|{b.id}", context | {"candidate_a": a.text, "candidate_b": b.text},
                      {"kind": "pairwise", "a_total": a.total, "b_total": b.total}) for a, b in pairs]
    res = await (judge if use_judge else _local).judge(items, [PAIRWISE_QUESTION])
    out = {}
    for (a, b), item in zip(pairs, items):
        j = res[item.key].get("pairwise")
        if j:
            out[(a.id, b.id)] = j.probs.get("A", 0.5)
    return out


def score_pool(cands, state, strategy, alts, desired, weights, mode):
    for c in cands:
        c.reject_reasons = ranking.hard_reject(c, state, strategy, desired, mode)
        c.rejected = bool(c.reject_reasons)
        ranking.score_candidate(c, state, strategy, alts, desired, weights)
        c.final, c.pairwise_winrate = c.total, None


async def rank(cands, state, strategy, alts, desired, weights, mode, context, sliders: Sliders, use_judge=True):
    score_pool(cands, state, strategy, alts, desired, weights, mode)
    alive = sorted([c for c in cands if not c.rejected], key=lambda c: -c.total)
    finalists = alive[: settings.tournament_size]
    probs = await pairwise_probs(finalists, context, use_judge)
    ranking.apply_tournament(finalists, probs)
    finalists.sort(key=lambda c: -c.final)
    picks = ranking.select_diverse(finalists, reserve=[c for c in alive if c.total > 45], bold_slider=sliders.bold)
    return alive, finalists, picks, probs


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def tournament_view(finalists: list[Candidate], probs: dict) -> dict:
    return {"ids": [c.id for c in finalists],
            "p": {f"{a}|{b}": round(p, 3) for (a, b), p in probs.items()}}


QUESTION_META = [{"id": q.id, "kind": q.kind, "risk": q.id in RISK_DIMS, "instructions": q.instructions}
                 for q in CANDIDATE_QUESTIONS]


def build_output(session_id, mode, state, strategy_info, desired, cands, finalists, picks, stats, pronoun, notes,
                 probs: dict | None = None):
    slot_of = {c.id: slot for slot, c in picks}
    top_fit = max((c.components.get("context_fit", 0) for c in finalists[:6]), default=0)
    return {
        "session_id": session_id,
        "mode": mode,
        "read": explain.conversation_read(state, strategy_info["chosen"], pronoun, mode),
        "state": state.model_dump(),
        "strategy": strategy_info,
        "desired_state": desired,
        "picks": [
            {"slot": slot, "id": c.id, "text": c.text, "why": explain.why(c, slot, strategy_info["chosen"], desired, pronoun, state.match_behavior),
             "labels": explain.labels(c, desired), "scores": explain.display_scores(c), "final": c.final}
            for slot, c in picks
        ],
        "pool": [
            {"id": c.id, "text": c.text, "intended_strategy": c.intended_strategy, "total": c.total, "final": c.final,
             "pairwise_winrate": c.pairwise_winrate, "rejected": c.rejected, "reject_reasons": c.reject_reasons,
             "components": c.components, "penalties": c.penalties, "boldness": round(ranking.boldness(c), 3),
             "scores": explain.display_scores(c),
             "low_confidence": [k for k, j in c.judgments.items() if j.confidence < settings.conf_escalate and j.source != "reasoning"],
             "escalated": [k for k, j in c.judgments.items() if j.source == "reasoning"],
             "judgments": {k: round(j.value, 3) for k, j in c.judgments.items()},
             "finalist": c in finalists, "slot": slot_of.get(c.id)}
            for c in sorted(cands, key=lambda c: (c.rejected, -c.final))
        ],
        "questions": QUESTION_META,
        "tournament": tournament_view(finalists, probs or {}),
        "pool_exhausted": top_fit < 0.55,
        "stats": stats,
        "notes": notes,
    }


def jev_report(cands: list[Candidate], errors_before: int) -> tuple[dict, list[str]]:
    """Make Jev failures visible instead of silently using the stand-in."""
    if judge.name != "jev":
        return {}, []
    fell_back = sum(1 for c in cands if any(j.source == "heuristic" for j in c.judgments.values()))
    new_errors = judge.stats.errors - errors_before
    info = {"jev_model": judge.stats.model_version, "jev_errors": new_errors, "jev_fallback_candidates": fell_back,
            "jev_last_error": judge.stats.last_error if new_errors else None}
    notes = []
    if new_errors:
        notes.append(f"Jev failed on {new_errors} request(s) ({fell_back}/{len(cands)} candidates used the local "
                     f"stand-in). Last error: {judge.stats.last_error}")
    return info, notes


def _load_user(user_id: str) -> tuple[StyleProfile, dict]:
    style, prefs = store.get_user(user_id)
    return StyleProfile(**style) if style else StyleProfile(), prefs or personalization.empty()


def _session_payload(req_mode, messages, profile, pronoun, platform, state, strategy_info, alts, cands, sliders):
    return {
        "mode": req_mode, "messages": _msgs_dicts(messages), "profile": profile, "pronoun": pronoun,
        "platform": platform, "state": state.model_dump(), "strategy_info": strategy_info, "alts": alts,
        "candidates": [c.model_dump() for c in cands], "sliders": sliders.model_dump(),
    }


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

async def coach(req: CoachRequest, user_id: str) -> dict:
    result = None
    async for ev in coach_events(req, user_id):
        if ev["type"] == "result":
            result = ev["result"]
    return result


async def coach_events(req: CoachRequest, user_id: str):
    """The whole search, as a stream of events the UI can animate. Last event: {"type": "result"}."""
    store.purge_expired()
    t = Timer()
    messages = req.messages or parse_conversation(req.conversation_text or "")
    profile = (req.profile_text or "").strip() or None
    if req.mode == "reply" and not messages:
        raise CoachError("Paste a conversation (e.g. 'Me: ...' / 'Her: ...') or switch to opener mode.")
    if req.mode == "opener" and not profile:
        raise CoachError("Opener mode needs profile text (prompts, captions, photo notes).")
    style, prefs = _load_user(user_id)
    errors_before = judge.stats.errors
    t.lap("load")

    def ev(type_, **kw):
        return {"type": type_, "t": t.total(), **kw}

    yield ev("start", backends=backends(), questions=QUESTION_META, n_state_questions=len(STATE_QUESTIONS))

    # 1-2. understand the conversation, choose a move
    state, state_raw, n_state = await build_state(req.mode, messages, profile, style, req.platform)
    t.lap("state")
    yield ev("state", read=explain.conversation_read(state, "CONTINUE_TOPIC", req.match_pronoun, req.mode),
             judgments=n_state, ms=t.stages["state"],
             hooks=[h.model_dump() for h in state.hooks])
    strategy, alts, strategy_info = await choose_strategy(req.mode, messages, profile, state, style)
    t.lap("strategy")
    yield ev("strategy", strategy=strategy_info, alts=alts, ms=t.stages["strategy"],
             read=explain.conversation_read(state, strategy, req.match_pronoun, req.mode))

    # 3-4. generate in parallel batches; judge each candidate as soon as it exists
    n = req.num_candidates or settings.num_candidates
    cands: list[Candidate] = []
    seen: set[str] = set()
    queue: asyncio.Queue = asyncio.Queue()
    gen_notes: list[str] = []
    n_cand = 0

    async def generate_all():
        async for name, raw, note in generate_batches(req.mode, messages, profile, state, strategy, alts, style,
                                                      req.sliders, n, req.match_pronoun, req.platform, avoid=[]):
            await queue.put(("batch", name, raw, note))
        await queue.put(("gen_done", None, None, None))

    async def judge_one(c: Candidate, item: EvalItem):
        res = await judge.judge([item], CANDIDATE_QUESTIONS)
        await queue.put(("judged", c, res[item.key], None))

    gen_task = asyncio.create_task(generate_all())
    pending_judges, gen_done = 0, False
    while not gen_done or pending_judges:
        kind, a, b, c_ = await queue.get()
        if kind == "batch":
            fresh = [r for r in b if (r.get("text") or "").strip().lower() not in seen]
            new = make_candidates(fresh, start=len(cands))
            seen |= {c.text.lower() for c in new}
            cands += new
            if c_:
                gen_notes.append(c_)
            yield ev("candidates", batch=a, items=[{"id": c.id, "text": c.text, "strategy": c.intended_strategy,
                                                   "boldness": c.intended_boldness} for c in new])
            for c, item in zip(new, candidate_items(new, req.mode, messages, profile, state, strategy, style)):
                pending_judges += 1
                asyncio.create_task(judge_one(c, item))
        elif kind == "gen_done":
            gen_done = True
            t.lap("generate+judge")
        else:
            pending_judges -= 1
            n_cand += apply_judgments(a, b)
            yield ev("judged", id=a.id, values={k: round(j.value, 3) for k, j in b.items()},
                     source=next(iter(b.values())).source if b else None)
    await gen_task
    t.lap("jev_candidates")

    # 5-6. hard rejection + contextual scoring
    desired = ranking.desired_state(state, strategy, style, req.sliders, prefs)
    weights = ranking.stage_weights(state.stage, prefs, req.sliders)
    ctx = render_context(req.mode, messages, profile, state, style, strategy)
    score_pool(cands, state, strategy, alts, desired, weights, req.mode)
    yield ev("scored", rows=[{"id": c.id, "total": c.total, "rejected": c.rejected, "reasons": c.reject_reasons}
                             for c in cands])

    # 7. System Two on the contenders
    contenders = sorted([c for c in cands if not c.rejected], key=lambda c: -c.total)[: settings.escalation_top]
    low_conf, escalated = await escalate(contenders, ctx)
    t.lap("escalation")
    yield ev("escalation", checked=len(contenders), unsure=low_conf, escalated=escalated, ms=t.stages["escalation"])

    # 8-9. tournament + diverse top 3
    alive, finalists, picks, probs = await rank(cands, state, strategy, alts, desired, weights, req.mode, ctx, req.sliders)
    t.lap("rank_tournament")
    yield ev("tournament", **tournament_view(finalists, probs))

    session_id = uuid.uuid4().hex[:12]
    store.save_session(session_id, user_id, _session_payload(req.mode, messages, profile, req.match_pronoun,
                                                             req.platform, state, strategy_info, alts, cands, req.sliders))
    t.lap("save")
    stats = {
        "generated": len(cands), "rejected": len(cands) - len(alive), "finalists": len(finalists),
        "jev_judgments": n_state + n_cand + 1 + len(probs), "candidate_judgments": n_cand,
        "pairwise_comparisons": len(probs), "low_confidence_judgments": low_conf, "escalated_to_reasoning": escalated,
        "timings_ms": t.stages | {"total": t.total()}, "backends": backends(), "state_judgments": state_raw,
        "judge_latency": judge.stats.snapshot(),
    }
    jev_info, jev_notes = jev_report(cands, errors_before)
    stats |= jev_info
    result = build_output(session_id, req.mode, state, strategy_info, desired, cands, finalists, picks, stats,
                          req.match_pronoun, jev_notes + gen_notes, probs)
    yield ev("result", result=result)


def _restore(sess: dict):
    state = ConversationState(**sess["state"])
    cands = [Candidate(**c) for c in sess["candidates"]]
    messages = [Message(**m) for m in sess["messages"]]
    return state, cands, messages


async def rerank(session_id: str, sliders: Sliders, user_id: str) -> dict:
    """Instant: no generation, no Jev calls - reuse stored judgments with a new desired state."""
    t = Timer()
    sess = store.get_session(session_id, user_id)
    if not sess:
        raise CoachError("Session expired or deleted - run the coach again.")
    state, cands, messages = _restore(sess)
    style, prefs = _load_user(user_id)
    strategy, alts = sess["strategy_info"]["chosen"], sess["alts"]
    desired = ranking.desired_state(state, strategy, style, sliders, prefs)
    weights = ranking.stage_weights(state.stage, prefs, sliders)
    alive, finalists, picks, probs = await rank(cands, state, strategy, alts, desired, weights, sess["mode"], {},
                                                  sliders, use_judge=False)
    sess["sliders"] = sliders.model_dump()
    store.save_session(session_id, user_id, sess | {"candidates": [c.model_dump() for c in cands]})
    stats = {"generated": len(cands), "rejected": len(cands) - len(alive), "finalists": len(finalists),
             "jev_judgments": 0, "pairwise_comparisons": len(probs), "reranked_only": True,
             "timings_ms": {"rerank": t.total(), "total": t.total()}, "backends": backends()}
    return build_output(session_id, sess["mode"], state, sess["strategy_info"], desired, cands, finalists, picks,
                        stats, sess["pronoun"], [], probs)


async def regenerate(session_id: str, sliders: Sliders, user_id: str) -> dict:
    """Pool lacks good matches for these sliders: generate more, judge only the new ones, merge, rank."""
    t = Timer()
    sess = store.get_session(session_id, user_id)
    if not sess:
        raise CoachError("Session expired or deleted - run the coach again.")
    state, cands, messages = _restore(sess)
    style, prefs = _load_user(user_id)
    strategy, alts, mode = sess["strategy_info"]["chosen"], sess["alts"], sess["mode"]
    raw, note = await generate(mode, messages, sess["profile"], state, strategy, alts, style, sliders,
                               max(12, settings.num_candidates // 2), sess["pronoun"], sess["platform"],
                               avoid=[c.text for c in cands])
    errors_before = judge.stats.errors
    new = make_candidates(raw, start=len(cands) + 1000)
    t.lap("generate")
    n_new = await judge_candidates(new, mode, messages, sess["profile"], state, strategy, style)
    t.lap("jev_candidates")
    cands += new
    desired = ranking.desired_state(state, strategy, style, sliders, prefs)
    weights = ranking.stage_weights(state.stage, prefs, sliders)
    ctx = render_context(mode, messages, sess["profile"], state, style, strategy)
    alive, finalists, picks, probs = await rank(cands, state, strategy, alts, desired, weights, mode, ctx, sliders)
    t.lap("rank_tournament")
    store.save_session(session_id, user_id, sess | {"candidates": [c.model_dump() for c in cands],
                                                    "sliders": sliders.model_dump()})
    stats = {"generated": len(cands), "new_candidates": len(new), "rejected": len(cands) - len(alive),
             "finalists": len(finalists), "jev_judgments": n_new + len(probs), "pairwise_comparisons": len(probs),
             "timings_ms": t.stages | {"total": t.total()}, "backends": backends()}
    jev_info, jev_notes = jev_report(new, errors_before)
    stats |= jev_info
    return build_output(session_id, mode, state, sess["strategy_info"], desired, cands, finalists, picks, stats,
                        sess["pronoun"], jev_notes + ([note] if note else []), probs)


def feedback(session_id: str, candidate_id: str, kind: str, edited_text: str | None, user_id: str) -> dict:
    sess = store.get_session(session_id, user_id)
    if not sess:
        raise CoachError("Session expired or deleted.")
    cand = next((Candidate(**c) for c in sess["candidates"] if c["id"] == candidate_id), None)
    if not cand:
        raise CoachError("Unknown suggestion.")
    style, prefs = _load_user(user_id)
    state = ConversationState(**sess["state"])
    desired = ranking.desired_state(state, sess["strategy_info"]["chosen"], style, Sliders(**sess["sliders"]), prefs)
    prefs = personalization.update(prefs, kind, cand.features, desired, cand.components, edited_text, cand.text)
    # what the user actually sends is the best evidence of their voice
    if kind in ("copy", "edited", "select"):
        sent = edited_text if kind == "edited" and edited_text else cand.text
        style.sample_messages = ([sent] + [s for s in style.sample_messages if s != sent])[:20]
    store.save_user(user_id, style=style.model_dump(), prefs=prefs)
    store.add_feedback(user_id, session_id, candidate_id, kind, edited_text or cand.text, cand.features)
    return {"ok": True, "preferences": personalization.describe(prefs), "raw": prefs}
