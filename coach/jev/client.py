"""Judge clients: the real TypeSafe System One API (Jev), and a local heuristic stand-in.

Both implement the same interface:

    await judge.judge(items, questions) -> {item.key: {question.id: Judgment}}

so the rest of the app never knows which one it is talking to.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from ..schemas import Judgment
from . import heuristics as H
from .types import EvalItem, Question

log = logging.getLogger(__name__)


@dataclass
class JudgeStats:
    calls: int = 0
    judgments: int = 0
    errors: int = 0
    last_error: str | None = None
    model_version: str | None = None
    input_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms[-500:])
        p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1) if lat else None  # noqa: E731
        return {"calls": self.calls, "judgments": self.judgments, "errors": self.errors,
                "last_error": self.last_error, "model_version": self.model_version,
                "input_tokens": self.input_tokens, "latency_p50_ms": p(0.5), "latency_p95_ms": p(0.95)}


def score_to_unit(index: float, n_levels: int) -> float:
    """Jev returns a Score as a (probability-weighted) level index; we use 0..1 internally."""
    return max(0.0, min(1.0, index / max(1, n_levels - 1)))


def distribution_around(n: int, value: float, sharpness: float = 6.0) -> dict[str, float]:
    """Heuristic judge only: fabricate a level distribution centred on `value` (0..1)."""
    centre = value * (n - 1)
    w = [pow(2.718, -sharpness * ((i - centre) / max(1, n - 1)) ** 2 * 4) for i in range(n)]
    z = sum(w)
    return {str(i): wi / z for i, wi in enumerate(w)}


class HeuristicJudge:
    """Local lexical judge. Deterministic, instant, free - and much dumber than Jev."""

    name = "heuristic"

    def __init__(self):
        self.stats = JudgeStats()

    async def judge(self, items: list[EvalItem], questions: list[Question]) -> dict[str, dict[str, Judgment]]:
        out: dict[str, dict[str, Judgment]] = {}
        for item in items:
            t0 = time.perf_counter()
            out[item.key] = self._judge_item(item, questions)
            self.stats.calls += 1
            self.stats.judgments += len(questions)
            self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def _judge_item(self, item: EvalItem, questions: list[Question]) -> dict[str, Judgment]:
        d = item.data
        kind = d.get("kind")
        if kind == "state":
            raw = H.state_judgments(d)
        elif kind == "strategy":
            probs, conf = H.strategy_probs(d["state"], d.get("mode", "reply"))
            raw = {"strategy": (probs, conf)}
        elif kind == "hook":
            v, conf, appearance = H.hook_strength(d["hook_text"])
            raw = {"hook_strength": (v, conf), "hook_is_appearance": (0.9 if appearance else 0.1, 1.0)}
        elif kind == "pairwise":
            raw = {"pairwise": H.pairwise(d["a_total"], d["b_total"])}
        else:
            raw = {q.id: H.CANDIDATE_FNS[q.id](d) for q in questions if q.id in H.CANDIDATE_FNS}

        res: dict[str, Judgment] = {}
        for q in questions:
            if q.id not in raw:
                continue
            val, conf = raw[q.id]
            if q.kind == "choice":
                probs = {o: float(val.get(o, 0.0)) for o in q.options}
                top = max(probs, key=probs.get)
                res[q.id] = Judgment(question_id=q.id, kind="choice", choice=top, value=probs[top],
                                     probs=probs, confidence=conf, source="heuristic")
            elif q.kind == "score":
                v = float(val)
                res[q.id] = Judgment(question_id=q.id, kind="score", value=round(v, 4),
                                     probs=distribution_around(len(q.levels), v), confidence=conf, source="heuristic")
            else:
                res[q.id] = Judgment(question_id=q.id, kind="noul", value=round(float(val), 4),
                                     confidence=1.0, source="heuristic")
        return res


class TypeSafeJevJudge:
    """Client for TypeSafe's System One API.

        POST {base}/v1/systemone
        {"state": ..., "model": "jev-latest", "questions": {id: {type, instructions, criteria}}}
     -> {"model": "jev-x.y.z", "answers": {id: {...}}, "usage": {...}}

    All questions for one item go in one request (they run in parallel server-side).
    If a request fails, that item falls back to the heuristic judge and the error is
    recorded in `stats.last_error` so the UI can show it - failures are never silent.
    """

    name = "jev"

    def __init__(self, api_key: str, base_url: str, model: str = "jev-latest", concurrency: int = 16):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.stats = JudgeStats()
        self.fallback = HeuristicJudge()
        self._client = httpx.AsyncClient(timeout=20.0)
        self._state_as_string = False  # flipped if the API rejects object state

    def request_body(self, item: EvalItem, questions: list[Question]) -> dict:
        state = json.dumps(item.state, ensure_ascii=False) if self._state_as_string else item.state
        return {"state": state, "model": self.model, "questions": {q.id: q.to_api() for q in questions}}

    @staticmethod
    def parse_answer(q: Question, a: dict) -> Judgment:
        if q.kind == "noul":
            # Noul returns only P(yes); there is no separate confidence.
            return Judgment(question_id=q.id, kind="noul", value=float(a["noul"]), confidence=1.0, source="jev")
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        conf = float(a.get("confidence", 0.0))
        if q.kind == "choice":
            top = a.get("choice") or max(probs, key=probs.get)
            return Judgment(question_id=q.id, kind="choice", choice=top, value=probs.get(top, 0.0),
                            probs=probs, confidence=conf, source="jev")
        return Judgment(question_id=q.id, kind="score", value=score_to_unit(float(a["score"]), len(q.levels)),
                        probs=probs, confidence=conf, source="jev")

    async def _post(self, item: EvalItem, questions: list[Question]) -> httpx.Response:
        return await self._client.post(
            f"{self.base_url}/v1/systemone",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=self.request_body(item, questions),
        )

    async def _one(self, item: EvalItem, questions: list[Question]) -> dict[str, Judgment]:
        async with self.sem:
            t0 = time.perf_counter()
            try:
                r = await self._post(item, questions)
                if r.status_code in (400, 422) and not self._state_as_string:
                    # Some deployments may only accept a string state; retry once that way.
                    self._state_as_string = True
                    r = await self._post(item, questions)
                if r.status_code >= 400:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:300]}", request=r.request, response=r)
                body = r.json()
                answers = body.get("answers") or {}
                self.stats.model_version = body.get("model", self.stats.model_version)
                self.stats.input_tokens += int((body.get("usage") or {}).get("input_tokens", 0))
                out = {q.id: self.parse_answer(q, answers[q.id]) for q in questions if q.id in answers}
                missing = [q.id for q in questions if q.id not in answers]
                if missing:
                    log.warning("Jev omitted answers for %s", missing)
                self.stats.judgments += len(out)
                return out
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
                self.stats.errors += 1
                self.stats.last_error = f"{type(e).__name__}: {e}"[:400]
                log.warning("Jev call failed (%s); falling back to heuristic for %s", self.stats.last_error, item.key)
                return (await self.fallback.judge([item], questions))[item.key]
            finally:
                self.stats.calls += 1
                self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)

    async def judge(self, items: list[EvalItem], questions: list[Question]) -> dict[str, dict[str, Judgment]]:
        results = await asyncio.gather(*(self._one(i, questions) for i in items))
        return {i.key: r for i, r in zip(items, results)}
