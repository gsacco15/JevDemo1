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


def _cand(cid, total):
    from coach.schemas import Candidate
    c = Candidate(id=cid, text=cid)
    c.total = total
    return c


def test_dominant_tournament_winner_can_take_best():
    a, b = _cand("a", 80), _cand("b", 76)
    ranking.apply_tournament([a, b], {("a", "b"): 0.1})  # Jev strongly prefers b
    assert b.final > a.final


def test_escalation_target_follows_readiness():
    from coach.schemas import ConversationState, StyleProfile
    base = dict(stage="ACTIVE_FLIRTING", message_count=6, flirt_level=0.4)
    ready = ranking.desired_state(ConversationState(**base, escalation_readiness=0.8, date_appropriate=0.8),
                                  "TEASE", StyleProfile(), Sliders(), {})
    cold = ranking.desired_state(ConversationState(**base, escalation_readiness=0.05, date_appropriate=0.05),
                                 "TEASE", StyleProfile(), Sliders(), {})
    assert ready["escalation"] > cold["escalation"] + 0.2


def test_change_topic_is_not_punished_for_asking_a_question():
    from coach.schemas import Candidate, ConversationState, Judgment
    c = Candidate(id="x", text="t", features={"words": 8})
    c.judgments["asks_question"] = Judgment(question_id="asks_question", kind="noul", value=0.95, confidence=1)
    st = ConversationState(questions_recently_asked_by_user=2, need_question=0.7)
    d = ranking.desired_state(st, "CHANGE_TOPIC", __import__("coach.schemas", fromlist=["StyleProfile"]).StyleProfile(), Sliders(), {})
    ranking.score_candidate(c, st, "CHANGE_TOPIC", [], d, ranking.stage_weights(st.stage, {}))
    assert "question_overload" not in c.penalties


def test_double_text_recommends_waiting():
    conv = TEASE_CONV + "\nMe: Ok\nMe: i don't think you could even try"
    r = run(pipeline.coach(CoachRequest(conversation_text=conv, match_pronoun="she"), "t9"))
    assert r["strategy"]["chosen"] == "WAIT"
    assert r["strategy"]["trailing_user"] == 2


def test_turn_awareness():
    replied = TEASE_CONV + "\nMe: I'll allow the head start"
    r = run(pipeline.coach(CoachRequest(conversation_text=replied, match_pronoun="she"), "t10"))
    assert r["strategy"]["chosen"] == "WAIT" and r["state"]["followup_reason"] == "none"
    typo = "Me: we should grab a drink sometime\nHer: yes! I'm free thursday or saturday\nMe: ok want to go firday at 7pm"
    r = run(pipeline.coach(CoachRequest(conversation_text=typo, match_pronoun="she"), "t11"))
    assert r["strategy"]["chosen"] == "FOLLOW_UP" and r["state"]["followup_reason"] == "fix_mistake"


def test_no_premature_date_penalty_when_she_is_planning():
    conv = "Me: we should grab a drink sometime\nHer: yes! I'm free thursday or saturday"
    r = run(pipeline.coach(CoachRequest(conversation_text=conv, match_pronoun="she"), "t12"))
    assert not any("premature_date" in c["penalties"] for c in r["pool"])


def test_feedback_learns_against_neutral_target_not_sliders():
    r = run(pipeline.coach(CoachRequest(conversation_text=TEASE_CONV, sliders=Sliders(bold=0, flirty=0, direct=0)), "t13"))
    pick = r["picks"][0]["id"]
    out = pipeline.feedback(r["session_id"], pick, "copy", None, "t13")
    # extreme-chill sliders must not make a normal pick look "more flirty/forward" than the user wants
    assert out["raw"]["offsets"]["flirt"] < 0.05 and out["raw"]["offsets"]["escalation"] < 0.05


def test_password_and_rate_limit():
    from fastapi.testclient import TestClient

    from coach import api
    from coach.config import settings
    old = (settings.app_password, settings.rate_limit_per_hour)
    settings.app_password, settings.rate_limit_per_hour = "s3cret", 2
    api._hits.clear()
    try:
        with TestClient(api.app) as c:
            assert c.get("/api/health").status_code == 200
            assert c.get("/api/style").status_code == 401
            h = {"x-app-key": "s3cret"}
            assert c.get("/api/style", headers=h).status_code == 200
            body = {"conversation_text": TEASE_CONV}
            assert c.post("/api/coach", json=body, headers=h).status_code == 200
            assert c.post("/api/coach", json=body, headers=h).status_code == 200
            assert c.post("/api/coach", json=body, headers=h).status_code == 429
    finally:
        settings.app_password, settings.rate_limit_per_hour = old
        api._hits.clear()


def test_hinge_style_profile_parsing_keeps_prompts_and_photos():
    from coach.pipeline import split_profile
    prof = ("Name: Lauren\nAge: 30\nHeight: 5'6\"\nPhoto set\n#\tImage description\tGeneration prompt\n"
            "2\tBarton Springs photo\tSame fictional woman swimming at Barton Springs in Austin, standing waist deep\n"
            "Hinge prompts\nTogether we could:\nFind the best margarita in Austin and become way too opinionated about it.\n"
            "The way to win me over is:\nMake a plan. Pick a place. Tell me what time.")
    hooks = split_profile(prof)
    texts = [t for _, t in hooks]
    assert texts[0].startswith("Together we could: Find the best margarita")
    assert any(t.startswith("Photo: Barton Springs photo - swimming") for t in texts)
    assert not any("Photo set" in t or "Image description" in t for t in texts)
    assert not any(t.startswith("Height") for t in texts)  # height isn't something to open on
    r = run(pipeline.coach(CoachRequest(mode="opener", profile_text=prof), "t14"))
    assert "margarita" in r["state"]["hooks"][0]["text"].lower() or "barton" in r["state"]["hooks"][0]["text"].lower()
    assert r["stats"]["state_judgments"] == {}  # no conversation questions asked for an opener


MIXED_PROFILE = """Here's another completely fictional Hinge profile as raw information, intentionally in mixed order:
Maddie, 29
The key to my heart is
Tacos, good banter, and actually making plans instead of talking about making plans
5'4"
Austin, Texas
Account Executive
Drinks sometimes
University of Colorado Boulder
Dating intention
Long term relationship
Typical Sunday
Pilates, an unnecessarily expensive coffee, convincing myself I'll meal prep, then somehow ending up at a patio with friends
Woman
Originally from Denver, Colorado
Has a dog
Gemini
Together we could
Book a flight because it was cheap and figure out the rest when we get there
Wants children
Photo captions/details visible on profile:
Standing outside a restaurant in a black dress with two friends
Holding a golden retriever at Zilker Park"""


def test_unlabeled_mixed_order_profile_is_sorted():
    from coach.profile import parse_profile
    p = parse_profile(MIXED_PROFILE)
    assert (p["name"], p["age"]) == ("Maddie", "29")
    assert [q for q, _ in p["prompts"]] == ["The key to my heart is", "Typical Sunday", "Together we could"]
    assert p["prompts"][0][1].startswith("Tacos")
    b = p["basics"]
    assert b["Location"] == "Austin, Texas" and b["Hometown"] == "Denver, Colorado" and b["Job"] == "Account Executive"
    assert b["Height"] == "5'4\"" and b["Zodiac"] == "Gemini" and b["Dating intention"] == "Long term relationship"
    assert b["Education"].startswith("University") and b["Pets"] == "Has a dog" and b["Kids"] == "Wants children"
    assert len(p["photos"]) == 2 and p["other"] == []
    from fastapi.testclient import TestClient
    from coach.api import app
    with TestClient(app) as c:
        r = c.post("/api/parse-profile", json={"text": MIXED_PROFILE}).json()
    assert r["name"] == "Maddie" and len(r["prompts"]) == 3


def test_profiles_from_other_apps_and_messy_notes():
    from coach.profile import parse_profile, looks_weak, from_model
    tinder = parse_profile("Jess 26\n📍 Lives in Brooklyn\n🎓 NYU\n💼 Nurse at Mount Sinai\n3 miles away\nAbout me\n"
                           "Professional brunch critic. Will judge your coffee order.\nInterests\nHiking · Ramen · Live music\n"
                           "Lifestyle\nSocial drinker\nNever smoke")
    assert (tinder["name"], tinder["basics"]["Location"], tinder["basics"]["Job"]) == ("Jess", "Brooklyn", "Nurse at Mount Sinai")
    assert tinder["basics"]["Interests"] == "Hiking, Ramen, Live music" and tinder["prompts"][0][0] == "About me"
    messy = parse_profile("omg ok so this girl on bumble\nname - Priya\n28\nlives in seattle but from chicago\n"
                          "3 photos: 1) at a concert 2) skiing 3) with a corgi\ninto: skiing, true crime podcasts\n"
                          "height 5'5, drinks socially, no smoking, dog lover")
    assert (messy["name"], messy["age"]) == ("Priya", "28") and messy["photos"] == ["at a concert", "skiing", "with a corgi"]
    assert messy["basics"]["Hometown"] == "chicago" and messy["basics"]["Height"] == "5'5" and not looks_weak(messy, "x")
    para = parse_profile("Emma 27 London. Works in marketing. Loves climbing and sourdough. Photos: her at a climbing wall, one on a boat")
    assert para["basics"]["Location"] == "London" and len(para["photos"]) == 2
    okc = parse_profile("Rachel · 33 · Chicago, IL\nMy self-summary\nTeacher by day, baker by night.\n"
                        "What I'm doing with my life\nVisiting every national park.")
    assert okc["basics"]["Location"] == "Chicago, IL" and [q for q, _ in okc["prompts"]] == ["About me", "What I'm doing with my life"]
    # the model fallback's output lands in the same shape
    m = from_model({"name": "Ana", "age": "30", "basics": [{"label": "Job", "value": "Chef"}],
                    "prompts": [{"title": "", "answer": "I make pasta"}], "photos": ["surfing"]})
    assert m["basics"] == {"Job": "Chef"} and m["prompts"] == [["About me", "I make pasta"]]


def test_profile_screenshots_merge_into_profile_text(monkeypatch):
    from fastapi.testclient import TestClient
    from coach import api, llm
    from coach.config import settings
    seen = {}

    async def fake(images, existing=""):
        seen["n"], seen["existing"] = len(images), existing
        return {"name": "Maddie", "age": "29", "basics": [{"label": "Location", "value": "Austin, Texas"}],
                "prompts": [{"title": "Typical Sunday", "answer": "Pilates and a patio"}],
                "photos": ["holding a golden retriever at Zilker Park"]}
    monkeypatch.setattr(llm, "profile_from_screenshots", fake)
    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    monkeypatch.setattr(settings, "force_mock", False)
    api._hits.clear()
    with TestClient(api.app) as c:
        files = [("files", (f"s{i}.jpg", b"\xff\xd8fake", "image/jpeg")) for i in range(3)]
        r = c.post("/api/parse-profile-shots", files=files, data={"existing": "Has a dog"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert seen == {"n": 3, "existing": "Has a dog"} and body["images"] == 3
        assert body["profile_text"].startswith("Maddie, 29\nLocation: Austin, Texas")
        assert "Photo: holding a golden retriever at Zilker Park" in body["profile_text"]
        too_many = [("files", (f"s{i}.jpg", b"x", "image/jpeg")) for i in range(11)]
        assert c.post("/api/parse-profile-shots", files=too_many).status_code == 400
        assert c.post("/api/parse-profile-shots", files=[("files", ("a.txt", b"x", "text/plain"))]).status_code == 400


def test_chat_screenshots_become_conversation_text(monkeypatch):
    from fastapi.testclient import TestClient
    from coach import api, llm
    from coach.config import settings
    from coach.parsing import parse_conversation
    replies = {"match_name": "Maddie Smith", "messages": [
        {"speaker": "user", "text": "tacos or   nothing"}, {"speaker": "match", "text": "breakfast tacos obviously 🌮"},
        {"speaker": "match", "text": ""}, {"speaker": "user", "text": "correct answer"}]}
    seen = {}

    async def fake(images, existing=""):
        seen["n"], seen["existing"] = len(images), existing
        return replies
    monkeypatch.setattr(llm, "chat_from_screenshots", fake)
    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    monkeypatch.setattr(settings, "force_mock", False)
    api._hits.clear()
    with TestClient(api.app) as c:
        files = [("files", (f"s{i}.jpg", b"\xff\xd8fake", "image/jpeg")) for i in range(4)]
        body = c.post("/api/parse-chat-shots", files=files, data={"existing": "Me: hey", "pronoun": "she"}).json()
        assert seen == {"n": 4, "existing": "Me: hey"} and body["messages"] == 3 and body["match_name"] == "Maddie"
        assert body["conversation_text"] == "Me: tacos or nothing\nHer: breakfast tacos obviously 🌮\nMe: correct answer"
        msgs = parse_conversation(body["conversation_text"])
        assert [m.speaker for m in msgs] == ["user", "match", "user"]
        body = c.post("/api/parse-chat-shots", files=files[:1], data={"pronoun": "he"}).json()
        assert body["conversation_text"].splitlines()[1].startswith("Him: ")
    # typed or pasted chats with any name still alternate correctly
    typed = parse_conversation("Me: hey\nZoë: hiii\nMe: tacos?\nMaddie 🌮: obviously\nChloé-Anne: wait who\nMe: ok")
    assert [m.speaker for m in typed] == ["user", "match", "user", "match", "match", "user"]
