import asyncio
import os

os.environ["COACH_DB"] = ":memory:"
os.environ["COACH_FORCE_MOCK"] = "1"

from coach import pipeline, ranking  # noqa: E402
from coach.parsing import parse_conversation  # noqa: E402
from coach.schemas import CoachRequest, Sliders  # noqa: E402

TEASE_CONV = "Me: ok but I'm dangerously good at go karts\nHer: Lol you definitely couldn't keep up"
PROFILE = "I get way too competitive at Mario Kart.\nPhoto: hiking in Patagonia\nYou're cute"


def run(coro):
    return asyncio.run(coro)


def test_parse_conversation():
    msgs = parse_conversation("Me: hi\nHer: hey\nthis continues\nJess: lol")
    assert [m.speaker for m in msgs] == ["user", "match", "match"]
    assert msgs[1].text == "hey this continues"


def test_reply_flow_returns_three_distinct_picks():
    r = run(pipeline.coach(CoachRequest(conversation_text=TEASE_CONV, match_pronoun="she"), "t1"))
    assert r["strategy"]["chosen"] == "TEASE"
    assert r["read"]["summary"].startswith("She's")
    assert len(r["picks"]) == 3
    assert len({p["text"] for p in r["picks"]}) == 3
    assert r["stats"]["jev_judgments"] > 500
    assert r["stats"]["pairwise_comparisons"] > 0


def test_bad_candidates_never_win():
    r = run(pipeline.coach(CoachRequest(conversation_text=TEASE_CONV), "t2"))
    rejected = {c["text"] for c in r["pool"] if c["rejected"]}
    assert any("Tennessee" in t for t in rejected)
    picked = {p["text"] for p in r["picks"]}
    assert not any("Tennessee" in t or "Most girls" in t or "hey beautiful" in t for t in picked)


def test_rerank_is_instant_and_changes_desired_state():
    r = run(pipeline.coach(CoachRequest(conversation_text=TEASE_CONV), "t3"))
    bold = run(pipeline.rerank(r["session_id"], Sliders(bold=1, flirty=1), "t3"))
    chill = run(pipeline.rerank(r["session_id"], Sliders(bold=0, flirty=0), "t3"))
    assert bold["stats"]["jev_judgments"] == 0
    assert bold["desired_state"]["escalation"] > chill["desired_state"]["escalation"]
    assert bold["desired_state"]["flirt"] > chill["desired_state"]["flirt"]


def test_opener_prefers_specific_hook_over_appearance():
    r = run(pipeline.coach(CoachRequest(mode="opener", profile_text=PROFILE), "t4"))
    hooks = r["state"]["hooks"]
    assert "Mario Kart" in hooks[0]["text"]
    assert hooks[-1]["is_appearance"]


def test_feedback_updates_preferences():
    r = run(pipeline.coach(CoachRequest(conversation_text=TEASE_CONV), "t5"))
    pick = r["picks"][0]["id"]
    out = pipeline.feedback(r["session_id"], pick, "too_much", None, "t5")
    assert out["raw"]["offsets"]["escalation"] < 0


def test_distance_scoring_penalises_overshoot():
    # same candidate, two desired states: fit should be better when target matches
    from coach.schemas import Candidate, ConversationState, Judgment
    c = Candidate(id="x", text="t", features={"words": 8})
    for d, val in {"flirt": 0.9, "confidence": 0.8, "escalation": 0.7, "playfulness": 0.5, "warmth": 0.5, "directness": 0.6}.items():
        c.judgments[d] = Judgment(question_id=d, kind="score", value=val, confidence=0.95)
    st = ConversationState()
    hot = {"flirt": 0.9, "confidence": 0.8, "escalation": 0.7, "playfulness": 0.5, "warmth": 0.5, "directness": 0.6, "words": 8, "sexual_tolerance": 0.2}
    cold = hot | {"flirt": 0.2, "escalation": 0.1}
    w = ranking.stage_weights("EARLY_CONVERSATION", {})
    ranking.score_candidate(c, st, "TEASE", [], hot, w)
    fit_hot = c.components["context_fit"]
    ranking.score_candidate(c, st, "TEASE", [], cold, w)
    assert fit_hot > c.components["context_fit"] + 0.3


def test_stream_endpoint_emits_every_stage():
    import json as _json

    from fastapi.testclient import TestClient

    from coach.api import app

    with TestClient(app) as client:
        r = client.post("/api/coach/stream", json={"conversation_text": TEASE_CONV, "match_pronoun": "she"})
        events = [_json.loads(line) for line in r.text.strip().splitlines()]
    types = [e["type"] for e in events]
    for t in ("start", "state", "strategy", "candidates", "judged", "scored", "escalation", "tournament", "result"):
        assert t in types, t
    assert types[-1] == "result"
    result = events[-1]["result"]
    assert result["questions"] and result["tournament"]["ids"]
    assert all("judgments" in c for c in result["pool"])
    assert types.count("judged") == result["stats"]["generated"]
