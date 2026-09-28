"""Judge clients: the real TypeSafe Jev API, and a local heuristic stand-in.

Both implement the same interface:

    await judge.judge(items, questions) -> {item.key: {question.id: Judgment}}

so the rest of the app never knows which one it is talking to.
"""

import asyncio
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
    latencies_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms)
        p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1) if lat else None  # noqa: E731
        return {"calls": self.calls, "judgments": self.judgments, "errors": self.errors,
                "latency_p50_ms": p(0.5), "latency_p95_ms": p(0.95)}


def score_from_distribution(levels: tuple[str, ...], probs: dict[str, float]) -> float:
    n = len(levels)
    if n <= 1:
        return 0.0
    return sum(probs.get(l, 0.0) * i for i, l in enumerate(levels)) / (n - 1)


def distribution_around(levels: tuple[str, ...], value: float, sharpness: float = 6.0) -> dict[str, float]:
    """Heuristic judge only: fabricate a level distribution centred on `value`."""
    n = len(levels)
    centre = value * (n - 1)
    w = [pow(2.718, -sharpness * ((i - centre) / max(1, n - 1)) ** 2 * 4) for i in range(n)]
    z = sum(w)
    return {l: wi / z for l, wi in zip(levels, w)}


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
        res: dict[str, Judgment] = {}
        if kind == "state":
            raw = H.state_judgments(d)
        elif kind == "strategy":
            probs, conf = H.strategy_probs(d["state"], d.get("mode", "reply"))
            raw = {"strategy": (probs, conf)}
        elif kind == "hook":
            v, conf, appearance = H.hook_strength(d["hook_text"])
            raw = {"hook_strength": (v, conf), "hook_is_appearance": (0.9 if appearance else 0.1, 0.8)}
        elif kind == "pairwise":
            raw = {"pairwise": H.pairwise(d["a_total"], d["b_total"])}
        else:
            raw = {q.id: H.CANDIDATE_FNS[q.id](d) for q in questions if q.id in H.CANDIDATE_FNS}

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
                                     probs=distribution_around(q.levels, v), confidence=conf, source="heuristic")
            else:
                res[q.id] = Judgment(question_id=q.id, kind="noul", value=round(float(val), 4),
                                     confidence=conf, source="heuristic")
        return res


class TypeSafeJevJudge:
    """HTTP client for TypeSafe's Jev.

    NOTE: the request/response mapping below is our best guess from public write-ups
    (Choice / Score / Noul questions, per-option probabilities, 0..1 confidence).
    Confirm against the official API reference and adjust `_request_body` /
    `_parse_answer` - nothing else in the app needs to change.
    If a call fails, that item falls back to the heuristic judge so a demo never dies.
    """

    name = "jev"

    def __init__(self, api_key: str, base_url: str, model: str = "jev", concurrency: int = 16):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.stats = JudgeStats()
        self.fallback = HeuristicJudge()
        self._client = httpx.AsyncClient(timeout=10.0)

    def _request_body(self, item: EvalItem, questions: list[Question]) -> dict:
        qs = []
        for q in questions:
            spec = {"id": q.id, "type": q.kind, "question": q.prompt}
            if q.kind == "choice":
                spec["options"] = list(q.options)
            elif q.kind == "score":
                spec["levels"] = list(q.levels)
            qs.append(spec)
        return {"model": self.model, "input": item.text, "questions": qs}

    def _parse_answer(self, q: Question, a: dict) -> Judgment:
        conf = float(a.get("confidence", 0.0))
        if q.kind == "noul":
            p = a.get("probability", a.get("p_true", a.get("value")))
            return Judgment(question_id=q.id, kind="noul", value=float(p), confidence=conf or abs(float(p) - 0.5) * 2, source="jev")
        probs = a.get("probabilities") or a.get("distribution") or {}
        if isinstance(probs, list):  # [{"option": ..., "probability": ...}]
            probs = {x.get("option") or x.get("level"): float(x["probability"]) for x in probs}
        if q.kind == "choice":
            top = a.get("choice") or max(probs, key=probs.get)
            return Judgment(question_id=q.id, kind="choice", choice=top, value=probs.get(top, 0.0),
                            probs=probs, confidence=conf, source="jev")
        value = a.get("score")
        if value is None:
            value = score_from_distribution(q.levels, probs)
        return Judgment(question_id=q.id, kind="score", value=float(value), probs=probs, confidence=conf, source="jev")

    async def _one(self, item: EvalItem, questions: list[Question]) -> dict[str, Judgment]:
        async with self.sem:
            t0 = time.perf_counter()
            try:
                r = await self._client.post(
                    f"{self.base_url}/v1/judge",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._request_body(item, questions),
                )
                r.raise_for_status()
                body = r.json()
                answers = body.get("answers") or body.get("results") or []
                by_id = {a.get("id"): a for a in answers}
                out = {q.id: self._parse_answer(q, by_id[q.id]) for q in questions if q.id in by_id}
                self.stats.judgments += len(out)
                return out
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
                self.stats.errors += 1
                log.warning("Jev call failed (%s); falling back to heuristic for %s", e, item.key)
                return (await self.fallback.judge([item], questions))[item.key]
            finally:
                self.stats.calls += 1
                self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)

    async def judge(self, items: list[EvalItem], questions: list[Question]) -> dict[str, dict[str, Judgment]]:
        results = await asyncio.gather(*(self._one(i, questions) for i in items))
        return {i.key: r for i, r in zip(items, results)}
