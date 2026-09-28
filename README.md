# Jev Dating Message Coach (MVP backend + test UI)

> Find the strongest message that still sounds like you.

This app works as a search and ranking engine, not a chatbot. For each request it:

```
conversation / profile
  → state extraction      (Jev: stage, tone, engagement, momentum, readiness …)
  → strategy selection    (Jev choice over 14 moves + deterministic rules)
  → generation            (~30 candidates, spread across moves and boldness)
  → Jev evaluation        (26 narrow Noul/Score questions per candidate ≈ 780 judgments)
  → hard rejection        (manipulation, insults, pickup lines, invented facts, …)
  → contextual scoring    (distance from a *desired state*, stage-dependent weights, penalties)
  → System Two escalation (low-confidence judgments on contenders → reasoning model)
  → pairwise tournament   (top 12 round-robin, Jev choice A/B)
  → diversity selection   (BEST / BOLDER / MORE RELAXED)
```

The tone sliders **re-rank the stored pool without any model calls**, which takes about 15 ms. "Search more candidates" generates more only when the pool has no good fit.

## Run it

```bash
pip install -r requirements.txt
uvicorn coach.api:app --reload      # open http://localhost:8000
pytest -q
```

It works with **no keys**, using local stand-ins:

| Layer | With key | Without key |
|---|---|---|
| Judge (Jev) | `JEV_API_KEY` (+ `JEV_BASE_URL`) → TypeSafe Jev | `heuristic` lexical judge in `coach/jev/heuristics.py` |
| Generator | `ANTHROPIC_API_KEY` → Claude (`GENERATOR_MODEL`, default `claude-opus-5`) | template generator in `coach/templates.py` |
| Reasoning fallback | Claude (`REASONING_MODEL`) | off (low-confidence judgments only get down-weighted) |
| Screenshot import | Claude vision | disabled |

The stand-ins exist so the whole pipeline can be exercised and inspected. They are not meant to be good at social judgment: the heuristic judge is keyword-based, and the template generator has a fixed pool that includes deliberately bad lines, so the rejection layer has something to catch. Judge the product's quality with real keys.

**Public deploys:** set `APP_PASSWORD` to require a password (the page asks once and remembers it in that browser). Searches are rate limited per IP with `RATE_LIMIT_PER_HOUR` (default 60).

Other settings: `NUM_CANDIDATES` (30), `CONF_AUTO` (0.90), `CONF_ESCALATE` (0.65), `RETENTION_HOURS` (24), `COACH_DB`, `COACH_FORCE_MOCK=1`.

## Jev integration

`TypeSafeJevJudge` (`coach/jev/client.py`) calls the documented System One API:

```
POST https://api.typesafe.ai/v1/systemone      Authorization: Bearer $TYPESAFE_API_KEY
{"state": {...named fields...}, "model": "jev-latest",
 "questions": {"flirt": {"type": "score", "instructions": "...", "criteria": ["...", "..."]},
               "stage": {"type": "choice", "instructions": "...", "criteria": {"KEY": "description"}},
               "generic": {"type": "noul", "instructions": "..."}}}
```

- All questions for one candidate go in **one request**, and the 30 candidates run in parallel.
- Score answers (a level index) are normalised to 0..1. Choice and Score answers carry `confidence`. Noul answers have no confidence field, so "unsure" for them means a probability near 0.5.
- If a call fails, that item falls back to the local stand-in, **and the UI shows a red banner with the error**. Failures are never silent.
- **`GET /api/jev-check`** makes one tiny real call and reports the model version, latency and any error. Use it to confirm a deploy is wired up.
- Env vars: `TYPESAFE_API_KEY` (or `JEV_API_KEY`), `JEV_MODEL` (default `jev-latest`), `JEV_BASE_URL`, `JEV_CONCURRENCY`.

All questions and criteria live in `coach/questions.py`. That's the file to review and edit together.

## Where things live

| File | What |
|---|---|
| `coach/questions.py` | **The Jev evaluation schema**: every narrow question. Tune wording here. |
| `coach/ranking.py` | Desired state, stage weights, confidence policy, hard rejects, scoring, tournament, diversity |
| `coach/pipeline.py` | Orchestrator plus `coach` / `rerank` / `regenerate` / `feedback` |
| `coach/personalization.py` | Learns tone offsets, weight multipliers, and length preference from feedback |
| `coach/explain.py` | Conversation read, move labels, and per-pick "why". Describes fit, never predicts outcomes. |
| `coach/llm.py` | Claude generation, low-confidence escalation, screenshot parsing |
| `coach/store.py` | SQLite. Sessions auto-expire, screenshots are never stored, one call deletes all data |
| `public/index.html` | Single-page test UI, including an "Under the hood" view of every candidate and score |

## Deploying on Vercel

`pyproject.toml` sets `[tool.vercel] entrypoint = "coach.api:app"`, and Vercel's CDN serves `public/index.html` at `/`. Add `ANTHROPIC_API_KEY` and `JEV_API_KEY` as environment variables in the project settings.

Caveat: on Vercel, SQLite lives in `/tmp`, which is per instance and temporary. Sessions and learned preferences can vanish, and a slider re-rank can hit "Session expired" if it lands on a different instance. That's fine for a demo. For real use, move `coach/store.py` to Postgres (for example Supabase).

## API

- `POST /api/coach`: `{mode: "reply"|"opener", conversation_text, profile_text, sliders, match_pronoun}`
- `POST /api/rerank`: `{session_id, sliders}` (instant, no model calls)
- `POST /api/regenerate`: `{session_id, sliders}` (adds candidates to the pool)
- `POST /api/feedback`: `{session_id, candidate_id, kind}` where kind is copy, like, dislike, too_much, too_boring, too_cheesy, not_me, too_long, more_direct, less_direct, or edited
- `POST /api/parse-screenshot`: multipart image
- `GET/PUT /api/style`, `DELETE /api/preferences`, `DELETE /api/sessions/{id}`, `DELETE /api/me`

Conversations use "Me:" lines for the user. Any other label is the match. The data model is platform-agnostic (`platform` field), and Hinge is the default.
