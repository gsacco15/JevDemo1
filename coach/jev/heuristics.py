"""Local stand-in for Jev.

A deliberately simple lexical judge that answers the same typed questions Jev would,
so the entire pipeline (rejection, scoring, tournament, personalization) can be
exercised without an API key. It is NOT meant to be good at social judgment; swap in
the real Jev client for that. Confidence values are lower when signals are weak, so
the System One -> System Two escalation path gets exercised too.
"""

import math
import re
from typing import Any, Callable

WORD_RE = re.compile(r"[a-z0-9']+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

STOPWORDS = set(
    """a an the and or but if so to of in on at for with from by is are was were be been am i me my
    you your yours we us our it its this that these those just really very too also not no yes do
    does did have has had can could would should will what whats how why when where who which
    there here then than lol haha hahaha omg im i'm you're youre it's dont don't gonna wanna u ur
    oh ok okay yeah yea like get got one about up out all some more much any be as""".split()
)

FLIRT = ["cute", "gorgeous", "beautiful", "sexy", "kiss", "charming", "handsome", "attractive",
         "dangerous", "trouble", "blush", "flirt", "smile", "crush", "wife", "husband", "stealing",
         "impressed", "dreamy", "date", "drinks", "😏", "😉", "😘", "😍", "❤️", "🥵"]
SEXUAL = ["sexy", "bed", "naked", "nudes", "hookup", "hook up", "come over", "my place", "your place",
          "body", "🥵", "🍆", "tonight at mine", "sleep over", "sleepover"]
NEEDY = ["please", "sorry", "why aren't you", "why didnt you", "why didn't you", "did i do something",
         "hello?", "miss you", "thought we had", "just checking", "don't ignore", "dont ignore",
         "if you want", "no pressure but", "you probably", "i know i'm not", "i hope that's ok",
         "you there", "still there", "???"]
HEDGES = ["maybe", "i guess", "idk", "sorry", "kinda", "sort of", "if that's ok", "not sure",
          "i think", "probably", "just wondering", "no worries if not"]
CONFIDENT = ["bet", "prepared to", "clearly", "obviously", "one way to find out", "i'll", "we'll",
             "let's", "big claim", "settle this", "confident", "i need", "important question",
             "i'm going to", "defend", "accusation", "evidence", "verdict", "i'd win", "call it"]
PLAYFUL = ["haha", "lol", "😂", "🤣", "bold", "big claim", "suspicious", "dangerous", "accusation",
           "evidence", "prove", "defend", "settle", "bet", "confess", "tragic", "unacceptable",
           "judge", "hot take", "red flag", "green flag", "controversial", "tested", "zero evidence",
           "rematch", "rivalry", "respectfully", "plot twist", "villain", "bribe", "negotiate"]
WARM = ["love", "awesome", "amazing", "glad", "fun", "sweet", "nice", "honestly", "appreciate",
        "sounds great", "that's great", "cool", "😊", "🙂", "❤️"]
PRESSURE = ["tonight", "right now", "asap", "you have to", "you should", "why not", "just say yes",
            "come on", "don't leave me", "answer me", "you owe", "now"]
NUMBER = ["number", "digits", "text me", "your #", "whatsapp", "phone", "insta", "instagram", "snap"]
DATE = ["drinks", "a drink", "coffee", "dinner", "grab a", "meet up", "meet you", "this week",
        "this weekend", "saturday", "friday", "sunday", "thursday", "take you", "hang out",
        "in person", "go on a date", "let's go", "we should go", "settle this properly", "irl"]
GENERIC = ["hey", "hi", "hello", "what's up", "whats up", "wyd", "how are you", "how's your day",
           "hows your day", "hey beautiful", "hey gorgeous", "you're cute", "youre cute", "nice pics",
           "you're gorgeous", "haha yeah", "cool", "nice", "lol", "same", "how was your weekend"]
PICKUP = ["are you a", "because you", "did it hurt", "is your name", "believe in love at first",
          "must be tired", "only 10", "fell from heaven", "parking ticket", "google", "are you from",
          "if you were a", "on a scale"]
MANIPULATIVE = ["you'd be lucky", "most girls", "most guys", "prove you're not", "you owe", "if you really",
                "don't be boring", "unlike other", "you're not like", "i usually don't", "you should feel"]
INSULT = ["ugly", "stupid", "dumb", "fat", "loser", "shut up", "annoying", "idiot", "boring person",
          "basic", "cringe", "pathetic", "desperate"]
CHALLENGE = ["couldn't", "couldnt", "can't", "cant", "bet", "definitely", "prove", "doubt", "you wish",
             "yeah right", "sure you", "no way", "keep up", "beat", "win", "better than", "not a chance",
             "we'll see", "big talk", "lose", "never"]
# explicit invitations only - "this weekend?" on its own is small talk, not a date ask
DATE_ASKS = ["grab a drink", "grab drinks", "get a drink", "get drinks", "grab coffee", "get coffee", "grab dinner",
             "take you", "meet up", "go on a date", "let's get", "let's go", "we should go", "we should grab",
             "we should get", "want to meet", "wanna meet", "drinks thursday", "drinks friday", "drinks saturday"]
INVITES = ["best spot", "i know a", "i know the best", "you have to try", "you'd love", "we should",
           "you should come", "take you to", "show you"]
APPEARANCE = ["pretty", "beautiful", "gorgeous", "cute", "hot", "eyes", "smile", "hair", "outfit",
              "dress", "photo", "pic", "looks", "attractive", "stunning"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS and len(w) > 2}


def stem(w: str) -> str:
    for suf in ("ing", "ers", "er", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def stems(text: str) -> set[str]:
    return {stem(w) for w in content_words(text)}


def hits(text: str, lexicon: list[str]) -> int:
    t = f" {(text or '').lower()} "
    n = 0
    for term in lexicon:
        if term.isalpha() and " " not in term:
            n += len(re.findall(rf"\b{re.escape(term)}\b", t))
        else:
            n += t.count(term)
    return n


def emoji_count(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sat(x: float, k: float = 1.0) -> float:
    """Saturating 0..1 transform of a non-negative count."""
    return 1 - math.exp(-k * max(0.0, x))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def conf_from_signal(strength: float) -> float:
    """Weak lexical evidence -> low confidence (so escalation can kick in)."""
    return round(clamp(0.58 + 0.38 * strength, 0.5, 0.97), 3)


def is_question(text: str) -> bool:
    t = (text or "").strip().lower()
    return "?" in t or bool(re.match(r"^(what|how|where|when|why|who|which|do|does|did|are|is|would|could|have|has)\b", t))


def proper_nouns(text: str) -> set[str]:
    toks = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text or "")
    common = {"I", "Me", "You", "The", "And", "But", "Okay", "Big", "Send", "Guess", "Dangerous",
              "Important", "Hot", "Scale", "Well", "Fair", "Also", "Honestly", "Wait", "Respectfully",
              "Plot", "Counterpoint", "Hey", "Hi", "Hello", "What", "How", "Why", "Hmm", "Lol", "Haha",
              "That", "This", "Not", "Now", "Mine", "Yours", "Tennessee"}
    return {t.lower() for t in toks if t not in common}


# ---------------------------------------------------------------------------
# candidate judgments: fn(data) -> (value 0..1, confidence)
# ---------------------------------------------------------------------------

def _anchor(data: dict) -> str:
    """What the candidate should be responding to."""
    if data.get("hook"):
        return data["hook"]
    msgs = data.get("messages") or []
    if msgs and msgs[-1]["speaker"] == "user":
        return msgs[-1]["text"]
    return data.get("last_match") or data.get("profile") or ""


def c_relevance(d):
    cand, anchor = d["candidate"], _anchor(d)
    overlap = len(stems(cand) & stems(anchor))
    behavior = d.get("state", {}).get("match_behavior")
    fit = 0.0
    if behavior == "teasing_challenge" and (hits(cand, PLAYFUL) + hits(cand, CONFIDENT)) > 0:
        fit = 0.72
    elif behavior == "asking_question" and not is_question(cand.split("?")[0] + "?") and re.search(r"\b(i|i'm|my|me)\b", cand.lower()):
        fit = 0.6
    elif behavior == "flirting" and hits(cand, FLIRT) + hits(cand, PLAYFUL) > 0:
        fit = 0.6
    ov = sat(overlap, 0.9)
    generic_hit = hits(cand, GENERIC) > 0 and len(words(cand)) <= 5
    v = max(ov, fit) * (0.4 if generic_hit else 1.0)
    # no lexical evidence either way -> genuinely unsure (a job for System Two)
    conf = 0.6 if (ov == 0 and fit == 0) else conf_from_signal(max(ov, fit))
    return clamp(v * 0.95 + 0.03), conf


def c_specificity(d):
    cand = d["candidate"]
    ctx = " ".join(m["text"] for m in d.get("messages", [])) + " " + (d.get("profile") or "") + " " + (d.get("hook") or "")
    overlap = len(stems(cand) & stems(ctx))
    generic = hits(cand, GENERIC) + hits(cand, PICKUP)
    v = clamp(0.2 + 0.25 * overlap + 0.03 * len(content_words(cand)) - 0.3 * generic)
    return v, conf_from_signal(min(1, 0.3 * overlap + 0.4 * generic + 0.2))


def c_naturalness(d):
    cand = d["candidate"]
    n = len(words(cand))
    v = 0.8
    if n > 40:
        v -= 0.3
    if emoji_count(cand) > 2:
        v -= 0.25
    if cand.count("!") > 2:
        v -= 0.2
    if hits(cand, PICKUP):
        v -= 0.35
    if re.search(r"\b(furthermore|additionally|moreover|delightful|endeavor|indeed)\b", cand.lower()):
        v -= 0.3
    return clamp(v), 0.72


def c_replyability(d):
    cand = d["candidate"]
    n = len(words(cand))
    v = 0.35
    if is_question(cand):
        v += 0.3
    if hits(cand, PLAYFUL) or hits(cand, CONFIDENT):
        v += 0.2  # a claim or tease invites a comeback
    if hits(cand, ["only one way", "prove", "defend", "explain", "tell me", "settle", "main", "verdict"]):
        v += 0.15
    if n <= 3 and not is_question(cand):
        v -= 0.25
    if n > 35:
        v -= 0.15
    if hits(cand, ["haha yeah", "cool", "nice", "same"]) and n <= 3:
        v -= 0.2
    return clamp(v), conf_from_signal(0.5)


def c_confidence(d):
    cand = d["candidate"]
    c, h = hits(cand, CONFIDENT), hits(cand, HEDGES) + hits(cand, NEEDY)
    v = clamp(0.55 + 0.15 * c - 0.2 * h + (0.05 if not is_question(cand) else -0.05))
    return v, conf_from_signal(min(1, 0.3 * (c + h)))


def c_playfulness(d):
    cand = d["candidate"]
    p = hits(cand, PLAYFUL) + 0.5 * emoji_count(cand)
    return clamp(0.15 + sat(p, 0.8) * 0.8), conf_from_signal(min(1, 0.35 * p))


def c_flirt(d):
    cand = d["candidate"]
    f = hits(cand, FLIRT) + 0.5 * hits(cand, DATE) + 0.4 * hits(cand, NUMBER)
    return clamp(0.08 + sat(f, 0.7) * 0.85), conf_from_signal(min(1, 0.3 * f + 0.2))


def c_warmth(d):
    cand = d["candidate"]
    w = hits(cand, WARM) - hits(cand, INSULT)
    return clamp(0.4 + 0.15 * w), 0.7


def c_humor(d):
    cand = d["candidate"]
    h = hits(cand, PLAYFUL) + hits(cand, ["zero evidence", "accusation", "friendship ending", "suspicious"])
    return clamp(0.15 + sat(h, 0.6) * 0.75), conf_from_signal(min(1, 0.3 * h))


def c_directness(d):
    cand = d["candidate"]
    x = hits(cand, NUMBER) + hits(cand, DATE) + hits(cand, ["i want", "i need", "let's", "send me", "give me"])
    return clamp(0.3 + sat(x, 0.8) * 0.65 - 0.1 * hits(cand, HEDGES)), conf_from_signal(min(1, 0.35 * x + 0.2))


def c_escalation(d):
    cand = d["candidate"]
    e = 1.4 * hits(cand, NUMBER) + 1.2 * hits(cand, DATE) + 0.5 * hits(cand, FLIRT) + 1.2 * hits(cand, SEXUAL)
    return clamp(0.05 + sat(e, 0.6) * 0.9), conf_from_signal(min(1, 0.3 * e + 0.25))


def c_neediness(d):
    cand = d["candidate"]
    n = hits(cand, NEEDY) + 0.5 * hits(cand, HEDGES) + (0.5 if cand.count("?") > 2 else 0)
    last_speakers = [m["speaker"] for m in d.get("messages", [])[-2:]]
    if last_speakers and all(s == "user" for s in last_speakers):
        n += 0.7  # double-texting
    return clamp(0.05 + sat(n, 0.9) * 0.9), conf_from_signal(min(1, 0.4 * n + 0.2))


def c_pressure(d):
    cand = d["candidate"]
    p = hits(cand, PRESSURE) + 0.3 * hits(cand, NUMBER) + (0.5 if cand.count("?") >= 3 else 0)
    return clamp(0.05 + sat(p, 0.8) * 0.85), conf_from_signal(min(1, 0.35 * p + 0.2))


def c_cringe(d):
    cand = d["candidate"]
    c = 1.5 * hits(cand, PICKUP) + 0.6 * max(0, emoji_count(cand) - 2) + 0.8 * hits(cand, ["m'lady", "queen", "angel", "princess"]) \
        + (0.5 if cand.count("!") > 2 else 0) + 0.6 * hits(cand, NEEDY)
    return clamp(0.05 + sat(c, 0.7) * 0.9), conf_from_signal(min(1, 0.35 * c + 0.2))


def c_sexual(d):
    cand = d["candidate"]
    s = hits(cand, SEXUAL) + 0.3 * hits(cand, ["hot", "🥵"])
    return clamp(0.02 + sat(s, 1.0) * 0.95), conf_from_signal(min(1, 0.45 * s + 0.3))


def c_style_fit(d):
    cand, style = d["candidate"], d.get("style") or {}
    n = len(words(cand))
    target = {"short": 10, "medium": 20, "long": 35}.get(style.get("average_message_length", "short"), 12)
    v = 1 - clamp(abs(n - target) / (target * 2.5))
    em_rate = style.get("emoji_frequency", 0.15)
    if emoji_count(cand) and em_rate < 0.1:
        v -= 0.2
    avoids = set(style.get("avoids", []))
    if "pickup_line_style" in avoids and hits(cand, PICKUP):
        v -= 0.4
    if "excessive_emojis" in avoids and emoji_count(cand) > 1:
        v -= 0.2
    if "long_explanations" in avoids and n > 30:
        v -= 0.2
    samples = style.get("sample_messages") or []
    if samples:
        lower_pref = sum(s[:1].islower() for s in samples if s) / max(1, len(samples))
        if (cand[:1].islower()) == (lower_pref > 0.5):
            v += 0.08
        vocab = set().union(*(content_words(s) for s in samples))
        v += 0.1 * sat(len(content_words(cand) & vocab), 0.8)
    humor = set(style.get("humor_style", []))
    if humor & {"teasing", "playful", "dry"} and hits(cand, PLAYFUL):
        v += 0.08
    return clamp(v), 0.66 if samples else 0.6


def c_generic(d):
    cand = d["candidate"]
    ctx = " ".join(m["text"] for m in d.get("messages", [])) + " " + (d.get("profile") or "") + " " + (d.get("hook") or "")
    overlap = len(stems(cand) & stems(ctx))
    g = hits(cand, GENERIC) * (2 if len(words(cand)) <= 5 else 1) + hits(cand, PICKUP)
    playful_callback = hits(cand, PLAYFUL) + hits(cand, CONFIDENT)
    p = clamp(0.15 + 0.3 * g - 0.18 * overlap - 0.08 * playful_callback)
    return p, conf_from_signal(min(1, 0.35 * (g + overlap)))


def c_repeats(d):
    cand = stems(d["candidate"])
    prior = [stems(m["text"]) for m in d.get("messages", []) if m["speaker"] == "user"]
    best = max((jaccard(cand, p) for p in prior), default=0)
    return clamp(best * 1.3), conf_from_signal(best + 0.3)


def c_invented_info(d):
    cand = d["candidate"]
    ctx = (" ".join(m["text"] for m in d.get("messages", [])) + " " + (d.get("profile") or "")).lower()
    names = [n for n in proper_nouns(cand) if n not in ctx]
    you_claims = re.findall(r"\byour (\w+)", cand.lower())
    unsupported = [w for w in you_claims if w not in ctx and w in {"dog", "cat", "sister", "brother", "job",
                                                                    "trip", "tattoo", "car", "kids", "ex"}]
    v = clamp(0.05 + 0.35 * len(names) + 0.35 * len(unsupported))
    return v, conf_from_signal(0.3 + 0.3 * (len(names) + len(unsupported)))


def c_manipulative(d):
    m = hits(d["candidate"], MANIPULATIVE)
    return clamp(0.03 + sat(m, 1.2) * 0.9), conf_from_signal(0.4 + 0.4 * m)


def c_insulting(d):
    i = hits(d["candidate"], INSULT)
    return clamp(0.03 + sat(i, 1.2) * 0.9), conf_from_signal(0.4 + 0.4 * i)


def c_pickup_line(d):
    p = hits(d["candidate"], PICKUP)
    return clamp(0.03 + sat(p, 1.5) * 0.95), conf_from_signal(0.4 + 0.4 * p)


def c_asks_known_info(d):
    cand = d["candidate"].lower()
    if not is_question(cand):
        return 0.03, 0.9
    ctx = (" ".join(m["text"] for m in d.get("messages", [])) + " " + (d.get("profile") or "")).lower()
    qs = [("what do you do", ["work", "job", "i'm a", "engineer", "nurse", "teacher"]),
          ("where are you from", ["from", "grew up"]),
          ("do you have a dog", ["dog"]), ("do you like to travel", ["travel"])]
    for q, keys in qs:
        if q in cand and any(k in ctx for k in keys):
            return 0.85, 0.8
    return 0.08, 0.7


def c_asks_question(d):
    return (0.95, 0.95) if is_question(d["candidate"]) else (0.05, 0.9)


def c_asks_number(d):
    n = hits(d["candidate"], NUMBER)
    return (0.9, 0.9) if n else (0.04, 0.88)


def c_asks_date(d):
    n = hits(d["candidate"], DATE)
    return (clamp(0.5 + 0.3 * n), conf_from_signal(0.3 + 0.35 * n)) if n else (0.05, 0.85)


CANDIDATE_FNS: dict[str, Callable[[dict], tuple[float, float]]] = {
    "relevance": c_relevance, "specificity": c_specificity, "naturalness": c_naturalness,
    "replyability": c_replyability, "confidence": c_confidence, "playfulness": c_playfulness,
    "flirt": c_flirt, "warmth": c_warmth, "humor": c_humor, "directness": c_directness,
    "escalation": c_escalation, "neediness": c_neediness, "pressure": c_pressure, "cringe": c_cringe,
    "sexual": c_sexual, "style_fit": c_style_fit, "generic": c_generic, "repeats": c_repeats,
    "invented_info": c_invented_info, "manipulative": c_manipulative, "insulting": c_insulting,
    "pickup_line": c_pickup_line, "asks_known_info": c_asks_known_info, "asks_question": c_asks_question,
    "asks_number": c_asks_number, "asks_date": c_asks_date,
}


# ---------------------------------------------------------------------------
# conversation-level judgments
# ---------------------------------------------------------------------------

def _msgs(d, who=None):
    return [m for m in d.get("messages", []) if who is None or m["speaker"] == who]


def _engagement(ms: list[dict]) -> float:
    if not ms:
        return 0.0
    recent = ms[-4:]
    avg_len = sum(len(words(m["text"])) for m in recent) / len(recent)
    q = sum(is_question(m["text"]) for m in recent) / len(recent)
    energy = sum(hits(m["text"], PLAYFUL) + emoji_count(m["text"]) for m in recent) / len(recent)
    return clamp(0.15 + sat(avg_len, 0.12) * 0.45 + 0.2 * q + 0.2 * sat(energy, 1.0))


def s_match_behavior(d) -> dict[str, float]:
    last = next((m["text"] for m in reversed(_msgs(d)) if m["speaker"] == "match"), None)
    if not last:
        return {"no_message": 1.0}
    scores = {
        "teasing_challenge": 0.6 * hits(last, CHALLENGE) + 0.3 * hits(last, PLAYFUL),
        "asking_question": 1.0 if is_question(last) else 0.0,
        "flirting": 0.6 * hits(last, FLIRT),
        "short_reply": 1.0 if len(words(last)) <= 3 and not is_question(last) else 0.0,
        "logistics": 0.7 * hits(last, DATE) + 0.7 * hits(last, NUMBER) + 0.9 * hits(last, INVITES),
        "compliment": 0.6 * hits(last, ["love your", "you're funny", "cute", "nice", "great taste", "like your"]),
        "sharing": 0.35,
    }
    return _softmax(scores, 3.0)


def s_tone(d) -> dict[str, float]:
    ms = _msgs(d, "match")[-4:]
    text = " ".join(m["text"] for m in ms)
    if not ms:
        return {"curious": 1.0}
    avg = sum(len(words(m["text"])) for m in ms) / len(ms)
    scores = {
        "playful": 0.4 * hits(text, PLAYFUL) + 0.2 * emoji_count(text),
        "teasing": 0.5 * hits(text, CHALLENGE),
        "flirty": 0.6 * hits(text, FLIRT),
        "warm": 0.4 * hits(text, WARM),
        "curious": 0.5 * sum(is_question(m["text"]) for m in ms),
        "serious": 0.3 if avg > 25 else 0.0,
        "dry": 0.2,
        "low_effort": 1.0 if avg <= 3 else 0.0,
    }
    return _softmax(scores, 2.5)


def s_momentum(d) -> dict[str, float]:
    ms = _msgs(d, "match")
    if len(ms) < 2:
        e = _engagement(ms)
        return _softmax({"positive": e, "neutral": 0.5, "fading": 0.2}, 3)
    early, late = _engagement(ms[:-2] or ms[:1]), _engagement(ms[-2:])
    trailing_user = 0
    for m in reversed(_msgs(d)):
        if m["speaker"] != "user":
            break
        trailing_user += 1
    return _softmax({"positive": late + 0.2 * (late - early), "neutral": 0.45,
                     "fading": (early - late) + 0.3 * trailing_user + (0.4 if late < 0.3 else 0)}, 4)


def s_stage(d, extra: dict) -> dict[str, float]:
    n = len(_msgs(d))
    if d.get("mode") == "opener" or n == 0:
        return {"PROFILE_OPENER": 0.97, "OPENING_EXCHANGE": 0.03}
    all_text = " ".join(m["text"] for m in _msgs(d))
    flirt = extra["flirt_level"]
    eng = extra["match_engagement"]
    mom = extra["momentum"]
    scores = {s: 0.0 for s in ["OPENING_EXCHANGE", "EARLY_CONVERSATION", "BUILDING_RAPPORT", "ACTIVE_FLIRTING",
                               "READY_TO_ESCALATE", "NUMBER_EXCHANGE", "DATE_PLANNING", "LOW_MOMENTUM", "RECOVERY"]}
    scores["OPENING_EXCHANGE"] = 1.0 if n <= 3 else 0.0
    scores["EARLY_CONVERSATION"] = 0.8 if 3 < n <= 9 else 0.2
    scores["BUILDING_RAPPORT"] = 0.7 if n > 9 else 0.1
    scores["ACTIVE_FLIRTING"] = 1.4 * flirt + (0.3 if n > 5 else 0)
    scores["READY_TO_ESCALATE"] = (1.0 if n >= 10 else 0.3) * (eng + flirt) * 0.8
    scores["NUMBER_EXCHANGE"] = 1.2 if hits(all_text, NUMBER) else 0.0
    scores["DATE_PLANNING"] = 1.2 if hits(all_text, DATE) >= 2 else 0.0
    scores["LOW_MOMENTUM"] = 1.5 * mom.get("fading", 0)
    return _softmax(scores, 3.0)


def state_judgments(d: dict) -> dict[str, tuple[Any, float]]:
    """Returns {question_id: (value_or_probs, confidence)} for STATE_QUESTIONS."""
    match_ms, user_ms = _msgs(d, "match"), _msgs(d, "user")
    all_text = " ".join(m["text"] for m in _msgs(d))
    n = len(_msgs(d))
    eng_m, eng_u = _engagement(match_ms), _engagement(user_ms)
    flirt_level = clamp(0.1 + sat(hits(all_text, FLIRT) + 0.5 * hits(all_text, PLAYFUL), 0.25) * 0.8)
    momentum = s_momentum(d)
    readiness = clamp(0.1 + 0.4 * eng_m + 0.3 * flirt_level + 0.25 * sat(n - 4, 0.15) - 0.4 * momentum.get("fading", 0))
    last_match = next((m["text"] for m in reversed(_msgs(d)) if m["speaker"] == "match"), "")
    recent_user_qs = sum(is_question(m["text"]) for m in _msgs(d)[-6:] if m["speaker"] == "user")
    need_q = clamp(0.55 - (0.35 if is_question(last_match) else 0) - 0.12 * recent_user_qs
                   + (0.25 if match_ms and len(words(last_match)) <= 4 else 0)
                   - (0.25 if hits(last_match, CHALLENGE) else 0))
    extra = {"flirt_level": flirt_level, "match_engagement": eng_m, "momentum": momentum}
    conf_len = conf_from_signal(sat(n, 0.25))
    return {
        "stage": (s_stage(d, extra), conf_len),
        "tone": (s_tone(d), conf_len * 0.95),
        "match_behavior": (s_match_behavior(d), 0.85 if last_match else 0.95),
        "momentum": (momentum, conf_len),
        "match_engagement": (eng_m, conf_len),
        "user_engagement": (eng_u, conf_len),
        "flirt_level": (flirt_level, conf_len * 0.95),
        "escalation_readiness": (readiness, conf_len * 0.9),
        "need_question": (need_q, 0.75),
        "number_appropriate": (clamp(readiness * 1.1 - 0.15 - (0.3 if n < 8 else 0)), conf_len * 0.9),
        "date_appropriate": (clamp(readiness * 1.05 - 0.2 - (0.3 if n < 10 else 0)), conf_len * 0.9),
    }


def strategy_probs(state: dict, mode: str) -> tuple[dict[str, float], float]:
    s = {k: 0.0 for k in [
        "CONTINUE_TOPIC", "TEASE", "ASK_QUESTION", "ANSWER_AND_REDIRECT", "FLIRT", "ESCALATE_FLIRT",
        "CHANGE_TOPIC", "ASK_FOR_NUMBER", "ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE", "PULL_BACK", "WAIT",
        "CLARIFY", "END_CONVERSATION"]}
    if mode == "opener":
        s.update({"TEASE": 1.0, "ASK_QUESTION": 1.1, "CONTINUE_TOPIC": 0.6, "FLIRT": 0.3})
        return _softmax(s, 2.0), 0.8
    b = state.get("match_behavior")
    s["CONTINUE_TOPIC"] = 0.6
    s["ASK_QUESTION"] = 0.3 + state.get("need_question", 0.5)
    if b == "teasing_challenge":
        s["TEASE"] = 1.6
    elif b == "asking_question":
        s["ANSWER_AND_REDIRECT"] = 1.5
    elif b == "flirting":
        s["FLIRT"] = 1.2
        s["ESCALATE_FLIRT"] = 0.6 + state.get("flirt_level", 0)
    elif b == "short_reply":
        s["CHANGE_TOPIC"] = 0.9
        s["ASK_QUESTION"] += 0.3
    elif b == "logistics":
        s["SUGGEST_SPECIFIC_DATE"] = 1.4
    elif b == "compliment":
        s["FLIRT"] = 0.9
        s["TEASE"] = 0.8
    if state.get("stage") == "READY_TO_ESCALATE":
        s["ASK_FOR_DATE"] = 0.9 + state.get("date_appropriate", 0)
        s["ASK_FOR_NUMBER"] = 0.8 + state.get("number_appropriate", 0)
    if state.get("conversation_momentum") == "fading":
        s["PULL_BACK"] = 0.8
        s["WAIT"] = 0.9
        s["CHANGE_TOPIC"] += 0.5
    probs = _softmax(s, 2.5)
    top = sorted(probs.values(), reverse=True)
    return probs, conf_from_signal(clamp((top[0] - top[1]) * 3))


def hook_strength(text: str) -> tuple[float, float, bool]:
    appearance = hits(text, APPEARANCE) > 0 and len(content_words(text)) <= 4
    specific = len(content_words(text)) + 2 * len(proper_nouns(" " + text)) + 2 * hits(text, ["too", "way too", "unpopular", "controversial", "never", "always", "obsessed", "worst", "best"])
    v = clamp(0.2 + sat(specific, 0.2) * 0.75 - (0.45 if appearance else 0))
    return v, conf_from_signal(sat(specific, 0.3)), appearance


def pairwise(a_total: float, b_total: float) -> tuple[dict[str, float], float]:
    pa = 1 / (1 + math.exp(-(a_total - b_total) / 6))
    return {"A": pa, "B": 1 - pa}, conf_from_signal(abs(pa - 0.5) * 2)


def _softmax(scores: dict[str, float], temp: float) -> dict[str, float]:
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) * temp) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}
