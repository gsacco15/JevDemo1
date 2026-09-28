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

Other settings: `NUM_CANDIDATES` (30), `CONF_AUTO` (0.90), `CONF_ESCALATE` (0.65), `RETENTION_HOURS` (24), `COACH_DB`, `COACH_FORCE_MOCK=1`.

## ⚠️ Jev API mapping is a best guess

TypeSafe's docs were not reachable while this was built. `TypeSafeJevJudge` in `coach/jev/client.py` sends:

```json
POST {JEV_BASE_URL}/v1/judge
{"model": "jev", "input": "<rendered app state>",
 "questions": [{"id": "flirt", "type": "score", "question": "...", "levels": ["none","low","medium","high","very_high"]},
               {"id": "generic", "type": "noul", "question": "..."},
               {"id": "strategy", "type": "choice", "question": "...", "options": ["TEASE", "..."]}]}
```

and expects `answers: [{id, probabilities | probability | score, confidence}]`. To match the real API, change only `_request_body` and `_parse_answer`. If a call fails, it falls back to the heuristic for that item, so a demo keeps working.

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
| `static/index.html` | Single-page test UI, including an "Under the hood" view of every candidate and score |

## API

- `POST /api/coach`: `{mode: "reply"|"opener", conversation_text, profile_text, sliders, match_pronoun}`
- `POST /api/rerank`: `{session_id, sliders}` (instant, no model calls)
- `POST /api/regenerate`: `{session_id, sliders}` (adds candidates to the pool)
- `POST /api/feedback`: `{session_id, candidate_id, kind}` where kind is copy, like, dislike, too_much, too_boring, too_cheesy, not_me, too_long, more_direct, less_direct, or edited
- `POST /api/parse-screenshot`: multipart image
- `GET/PUT /api/style`, `DELETE /api/preferences`, `DELETE /api/sessions/{id}`, `DELETE /api/me`

Conversations use "Me:" lines for the user. Any other label is the match. The data model is platform-agnostic (`platform` field), and Hinge is the default.
