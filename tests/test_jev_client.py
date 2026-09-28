"""The TypeSafe client against a fake server that speaks the documented /v1/systemone format."""

import asyncio
import json

import httpx

from coach.jev.client import TypeSafeJevJudge
from coach.jev.types import EvalItem
from coach.questions import CANDIDATE_QUESTIONS, PAIRWISE_QUESTION, STATE_QUESTIONS


def fake_answer(spec: dict) -> dict:
    if spec["type"] == "noul":
        return {"type": "noul", "noul": 0.8}
    if spec["type"] == "choice":
        keys = list(spec["criteria"])
        return {"type": "choice", "choice": keys[0], "confidence": 0.78,
                "probabilities": {k: (0.85 if i == 0 else 0.15 / (len(keys) - 1)) for i, k in enumerate(keys)}}
    n = len(spec["criteria"])
    return {"type": "score", "score": float(n - 1), "confidence": 0.95,
            "legend": {str(i): c for i, c in enumerate(spec["criteria"])},
            "probabilities": {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)}}


def make_judge(handler) -> TypeSafeJevJudge:
    j = TypeSafeJevJudge("test-key", "https://api.typesafe.ai")
    j._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return j


def test_request_and_response_match_documented_format():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], body
        return httpx.Response(200, json={"model": "jev-1.13.0",
                                         "answers": {k: fake_answer(v) for k, v in body["questions"].items()},
                                         "usage": {"input_tokens": 392, "output_tokens": 65}})

    judge = make_judge(handler)
    qs = CANDIDATE_QUESTIONS + STATE_QUESTIONS + [PAIRWISE_QUESTION]
    item = EvalItem("c1", {"conversation": [{"speaker": "match", "text": "hi"}], "candidate_message": "hey"})
    res = asyncio.run(judge.judge([item], qs))["c1"]

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "jev-latest" and body["state"]["candidate_message"] == "hey"
    assert isinstance(body["questions"], dict)
    assert body["questions"]["flirt"]["type"] == "score" and isinstance(body["questions"]["flirt"]["criteria"], list)
    assert body["questions"]["stage"]["type"] == "choice" and isinstance(body["questions"]["stage"]["criteria"], dict)
    assert set(body["questions"]["generic"]) == {"type", "instructions"}

    assert len(res) == len(qs) and all(j.source == "jev" for j in res.values())
    assert res["flirt"].value == 1.0 and res["flirt"].confidence == 0.95  # top level -> 1.0
    assert res["generic"].value == 0.8
    assert res["stage"].choice == "PROFILE_OPENER" and res["stage"].confidence == 0.78
    assert judge.stats.model_version == "jev-1.13.0" and judge.stats.errors == 0


def test_failure_falls_back_but_is_recorded():
    judge = make_judge(lambda r: httpx.Response(401, json={"error": "invalid api key"}))
    item = EvalItem("c1", {"candidate_message": "hey"},
                    {"kind": "candidate", "candidate": "hey", "messages": [], "state": {}, "style": {}})
    res = asyncio.run(judge.judge([item], CANDIDATE_QUESTIONS))["c1"]
    assert all(j.source == "heuristic" for j in res.values())
    assert judge.stats.errors == 1 and "401" in judge.stats.last_error


def test_string_state_retry_on_422():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(type(body["state"]).__name__)
        if isinstance(body["state"], dict):
            return httpx.Response(422, json={"detail": "state must be a string"})
        return httpx.Response(200, json={"model": "jev-1.13.0",
                                         "answers": {k: fake_answer(v) for k, v in body["questions"].items()}})

    judge = make_judge(handler)
    res = asyncio.run(judge.judge([EvalItem("c1", {"candidate_message": "hey"})], CANDIDATE_QUESTIONS[:3]))["c1"]
    assert calls == ["dict", "str"] and all(j.source == "jev" for j in res.values())
