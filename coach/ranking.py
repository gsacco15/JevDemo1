"""Deterministic decision layer: judgments -> rejections -> contextual scores -> tournament -> top 3.

Key ideas from the spec:
  * score by DISTANCE from a desired state, not by raw "more is better"
  * weights depend on conversation stage (and are personalised)
  * low-confidence judgments count for less
  * the final three should be meaningfully different (BEST / BOLDER / CHILLER)
"""

import math

from .config import settings
from .jev.heuristics import content_words, jaccard
from .schemas import Candidate, ConversationState, Sliders, StyleProfile

TONE_DIMS = ["flirt", "confidence", "escalation", "playfulness", "warmth", "directness"]
TONE_DIM_WEIGHTS = {"flirt": 1.2, "confidence": 0.8, "escalation": 1.3, "playfulness": 1.0, "warmth": 0.6, "directness": 0.8}

STAGE_TARGETS = {
    #                    flirt conf  esc  play warm direct
    "PROFILE_OPENER":    (.30, .70, .10, .60, .50, .40),
    "OPENING_EXCHANGE":  (.35, .70, .15, .60, .55, .40),
    "EARLY_CONVERSATION": (.40, .70, .25, .60, .55, .45),
    "BUILDING_RAPPORT":  (.45, .72, .35, .55, .60, .50),
    "ACTIVE_FLIRTING":   (.65, .80, .45, .70, .50, .55),
    "READY_TO_ESCALATE": (.60, .85, .65, .55, .50, .70),
    "NUMBER_EXCHANGE":   (.50, .80, .60, .50, .55, .70),
    "DATE_PLANNING":     (.50, .80, .60, .45, .60, .75),
    "POST_DATE":         (.55, .75, .50, .50, .70, .60),
    "LOW_MOMENTUM":      (.30, .70, .15, .60, .50, .35),
    "RECOVERY":          (.30, .70, .15, .50, .60, .40),
}

BASE_WEIGHTS = {
    "context_fit": 0.25, "replyability": 0.15, "style_fit": 0.15, "naturalness": 0.15,
    "strategy_alignment": 0.15, "originality": 0.05, "confidence_fit": 0.05, "brevity_fit": 0.05,
}
STAGE_WEIGHT_TWEAKS = {
    "PROFILE_OPENER": {"originality": 0.15, "replyability": 0.18, "strategy_alignment": 0.07},
    "OPENING_EXCHANGE": {"originality": 0.10},
    "ACTIVE_FLIRTING": {"context_fit": 0.28},
    "READY_TO_ESCALATE": {"strategy_alignment": 0.22},
    "LOW_MOMENTUM": {"replyability": 0.22, "brevity_fit": 0.08},
}

LENGTH_WORDS = {"short": 9, "medium": 16, "long": 28}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# desired state
# ---------------------------------------------------------------------------

def desired_state(state: ConversationState, strategy: str, style: StyleProfile, sliders: Sliders, prefs: dict) -> dict:
    t = dict(zip(TONE_DIMS, STAGE_TARGETS.get(state.stage, STAGE_TARGETS["EARLY_CONVERSATION"])))

    if strategy == "TEASE":
        t["playfulness"] += 0.15
        t["confidence"] += 0.05
    elif strategy == "FLIRT":
        t["flirt"] += 0.15
    elif strategy == "ESCALATE_FLIRT":
        t["flirt"] += 0.25
        t["escalation"] += 0.15
    elif strategy in ("ASK_FOR_NUMBER", "ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE"):
        t["escalation"] = max(t["escalation"], 0.7)
        t["directness"] = max(t["directness"], 0.75)
    elif strategy in ("PULL_BACK", "WAIT"):
        t["escalation"] -= 0.1
        t["flirt"] -= 0.1

    # (2) let Jev's read of readiness move the escalation target (the stage table is only a prior)
    if state.stage != "PROFILE_OPENER" and state.message_count:
        ready = max(state.escalation_readiness, 0.85 * state.date_appropriate)
        t["escalation"] = 0.5 * t["escalation"] + 0.5 * ready
    # follow the match's lead a little
    if state.stage != "PROFILE_OPENER":
        t["flirt"] += 0.3 * (state.flirt_level - t["flirt"])
    # the user's own style
    t["flirt"] = 0.75 * t["flirt"] + 0.25 * style.preferred_flirt_level
    t["directness"] = 0.75 * t["directness"] + 0.25 * style.preferred_directness
    if set(style.humor_style) & {"teasing", "playful", "dry", "sarcastic"}:
        t["playfulness"] += 0.05

    # learned preferences
    for k, v in (prefs.get("offsets") or {}).items():
        if k in t:
            t[k] += v

    # sliders (0.5 = no change)
    b, f, s, d = sliders.bold - 0.5, sliders.flirty - 0.5, sliders.serious - 0.5, sliders.direct - 0.5
    t["escalation"] += 0.7 * b + 0.3 * d
    t["confidence"] += 0.3 * b + 0.3 * d
    t["flirt"] += 0.8 * f
    t["warmth"] -= 0.2 * f
    t["playfulness"] -= 0.8 * s
    t["directness"] += 0.8 * d + 0.2 * b
    t = {k: round(clamp(v, 0.02, 0.98), 3) for k, v in t.items()}

    base_words = LENGTH_WORDS.get(style.average_message_length, 10)
    if state.last_match_message:
        mw = len(state.last_match_message.split())
        base_words = 0.7 * base_words + 0.3 * clamp(mw, 4, 30)
    words = base_words * (2 ** ((sliders.expressive - 0.5) * 2)) * (prefs.get("length_mult") or 1.0)
    t["words"] = round(clamp(words, 3, 60), 1)
    t["sexual_tolerance"] = 0.35 if state.stage in ("ACTIVE_FLIRTING", "POST_DATE") else 0.15
    if state.stage in ("ACTIVE_FLIRTING", "POST_DATE") and sliders.flirty > 0.7:
        t["sexual_tolerance"] = 0.5
    return t


def stage_weights(stage: str, prefs: dict, sliders: Sliders | None = None) -> dict:
    w = dict(BASE_WEIGHTS)
    w.update(STAGE_WEIGHT_TWEAKS.get(stage, {}))
    if sliders:
        # the further the user pushes the sliders, the more tone fit matters vs. our defaults
        s = sliders.model_dump()
        intensity = sum(abs(x - 0.5) * 2 for x in s.values()) / len(s)
        w["context_fit"] += 0.25 * intensity
    for k, m in (prefs.get("weights") or {}).items():
        if k in w:
            w[k] *= m
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


# ---------------------------------------------------------------------------
# judgment access with confidence policy
# ---------------------------------------------------------------------------

def conf_weight(conf: float) -> float:
    """>= auto threshold: full weight. Between: scaled. Below escalate threshold: 0.4."""
    if conf >= settings.conf_auto:
        return 1.0
    if conf >= settings.conf_escalate:
        span = settings.conf_auto - settings.conf_escalate
        return 0.6 + 0.4 * (conf - settings.conf_escalate) / span
    return 0.4


def v(c: Candidate, dim: str, neutral: float = 0.5) -> float:
    j = c.judgments.get(dim)
    if j is None:
        return neutral
    if j.source in ("jev", "reasoning") or j.kind == "noul":
        # A Jev Score is already a probability-weighted position, and a Noul is already a
        # probability: shrinking by confidence again would double-count the uncertainty.
        # Confidence is used only to decide what to escalate (pipeline.escalate).
        return j.value
    w = conf_weight(j.confidence)  # heuristic stand-in: its made-up values deserve shrinkage
    return w * j.value + (1 - w) * neutral


def trusted(c: Candidate, dim: str) -> bool:
    j = c.judgments.get(dim)
    return bool(j) and (j.source in ("jev", "reasoning") or j.kind == "noul" or j.confidence >= settings.conf_escalate)


def boldness(c: Candidate) -> float:
    return 0.4 * v(c, "escalation") + 0.3 * v(c, "flirt") + 0.3 * v(c, "directness")


# ---------------------------------------------------------------------------
# hard rejection
# ---------------------------------------------------------------------------

def hard_reject(c: Candidate, state: ConversationState, strategy: str, desired: dict, mode: str) -> list[str]:
    reasons = []

    def over(dim, thr, label):
        j = c.judgments.get(dim)
        if j and j.value > thr and trusted(c, dim):
            reasons.append(label)

    over("manipulative", 0.7, "manipulative framing")
    over("insulting", 0.7, "insult risk")
    over("sexual", max(0.7, desired["sexual_tolerance"] + 0.35), "unnecessarily sexual")
    over("repeats", 0.75, "repeats something already said")
    over("invented_info", 0.7, "invents details about the match")
    over("pickup_line", 0.7, "canned pickup line")
    over("asks_known_info", 0.75, "asks for info already given")
    if c.features.get("words", 0) > max(3 * desired["words"], 45):
        reasons.append("far too long for this conversation")
    rel = c.judgments.get("relevance")
    if (mode == "reply" and strategy not in ("CHANGE_TOPIC", "PULL_BACK", "WAIT") and state.last_speaker != "user"
            and rel and rel.value < 0.1 and trusted(c, "relevance")):
        reasons.append("doesn't respond to the conversation")
    g, sp = c.judgments.get("generic"), c.judgments.get("specificity")
    g_cut, sp_cut = (0.9, 0.15) if strategy == "CHANGE_TOPIC" else (0.8, 0.25)  # a fresh topic is allowed to be broad
    if g and sp and g.value > g_cut and sp.value < sp_cut and trusted(c, "generic"):
        reasons.append("generic line that could go to anyone")
    return reasons


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def strategy_alignment(c: Candidate, strategy: str) -> float:
    rel, q = v(c, "relevance"), v(c, "asks_question", 0.3)
    f = {
        "TEASE": lambda: 0.6 * v(c, "playfulness") + 0.4 * v(c, "confidence"),
        "ASK_QUESTION": lambda: q,
        "CLARIFY": lambda: q,
        "ANSWER_AND_REDIRECT": lambda: 0.5 * rel + 0.5 * q,
        "FLIRT": lambda: 1 - abs(v(c, "flirt") - 0.6),
        "ESCALATE_FLIRT": lambda: v(c, "flirt"),
        "CONTINUE_TOPIC": lambda: rel,
        "CHANGE_TOPIC": lambda: 0.5 * (1 - rel) + 0.5 * q,
        "ASK_FOR_NUMBER": lambda: v(c, "asks_number", 0.1),
        "ASK_FOR_DATE": lambda: v(c, "asks_date", 0.1),
        "SUGGEST_SPECIFIC_DATE": lambda: v(c, "asks_date", 0.1),
        "PULL_BACK": lambda: 1 - 0.5 * (v(c, "neediness") + v(c, "pressure")),
        "WAIT": lambda: 1 - 0.5 * (v(c, "neediness") + v(c, "pressure")),
        "END_CONVERSATION": lambda: v(c, "warmth"),
    }.get(strategy, lambda: rel)()
    return clamp(f + (0.15 if c.intended_strategy == strategy else 0))


def score_candidate(c: Candidate, state: ConversationState, strategy: str, alt_strategies: list[str],
                    desired: dict, weights: dict) -> None:
    # tone distance -> context fit
    dist = sum(TONE_DIM_WEIGHTS[k] * abs(v(c, k) - desired[k]) for k in TONE_DIMS) / sum(TONE_DIM_WEIGHTS.values())
    context_fit = clamp(1 - 1.8 * dist)
    align = max([strategy_alignment(c, strategy)] + [0.8 * strategy_alignment(c, s) for s in alt_strategies[:2]])
    words = max(1, c.features.get("words", 1))
    ratio = words / desired["words"]
    brevity_fit = math.exp(-(math.log(ratio) ** 2) / (2 * 0.55 ** 2))
    comps = {
        "context_fit": context_fit,
        "replyability": v(c, "replyability"),
        "style_fit": 0.7 * v(c, "style_fit") + 0.3 * brevity_fit,
        "naturalness": v(c, "naturalness"),
        "strategy_alignment": align,
        "originality": v(c, "specificity") * (1 - v(c, "generic", 0.3)),
        "confidence_fit": 1 - abs(v(c, "confidence") - desired["confidence"]),
        "brevity_fit": brevity_fit,
    }

    def pen(dim, scale, floor=0.15, neutral=0.2):
        x = v(c, dim, neutral)
        return scale * max(0.0, x - floor) / (1 - floor)

    asking = strategy in ("ASK_FOR_NUMBER", "ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE")
    pens = {
        "cringe": pen("cringe", 25),
        "neediness": pen("neediness", 20),
        "genericness": pen("generic", 9 if strategy == "CHANGE_TOPIC" else 18, floor=0.25),
        "pressure": pen("pressure", 8 if asking else 15),
        "repetition": pen("repeats", 25, floor=0.3),
        "pickup_line": pen("pickup_line", 20, floor=0.2),
        "sexual": 20 * max(0.0, v(c, "sexual", 0.1) - desired["sexual_tolerance"]),
        "invented_info": pen("invented_info", 20, floor=0.3),
        "asks_known_info": pen("asks_known_info", 15, floor=0.3),
    }
    if (v(c, "asks_question", 0.3) > 0.5 and state.questions_recently_asked_by_user >= 2
            and strategy not in ("CHANGE_TOPIC", "ASK_QUESTION", "ANSWER_AND_REDIRECT") and state.need_question < 0.6):
        pens["question_overload"] = 8.0
    # asking for more is only "premature" relative to how bold the user asked us to be
    boldness_ask = clamp(desired["escalation"] - 0.3, 0, 0.6) / 0.6
    if v(c, "asks_number", 0.1) > 0.5:
        base = 8 if strategy == "ASK_FOR_NUMBER" else 20
        pens["premature_number"] = base * (1 - state.number_appropriate) * (1 - 0.6 * boldness_ask)
    if v(c, "asks_date", 0.1) > 0.5:
        base = 8 if strategy in ("ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE") else 20
        pens["premature_date"] = base * (1 - state.date_appropriate) * (1 - 0.6 * boldness_ask)
    if state.stage == "READY_TO_ESCALATE" and v(c, "escalation") < 0.2:
        pens["missed_escalation"] = 5.0

    c.components = {k: round(x, 3) for k, x in comps.items()}
    c.penalties = {k: round(x, 2) for k, x in pens.items() if x > 0.05}
    c.total = round(100 * sum(weights[k] * comps[k] for k in weights) - sum(c.penalties.values()), 2)


# ---------------------------------------------------------------------------
# tournament + diversity
# ---------------------------------------------------------------------------

def apply_tournament(finalists: list[Candidate], pair_probs: dict[tuple[str, str], float]) -> None:
    """pair_probs[(a,b)] = P(a beats b). Final = total + win-rate bonus."""
    for c in finalists:
        others = [o for o in finalists if o.id != c.id]
        if not others:
            c.pairwise_winrate, c.final = 1.0, c.total
            continue
        wins = 0.0
        for o in others:
            if (c.id, o.id) in pair_probs:
                wins += pair_probs[(c.id, o.id)]
            elif (o.id, c.id) in pair_probs:
                wins += 1 - pair_probs[(o.id, c.id)]
            else:
                wins += 0.5
        c.pairwise_winrate = round(wins / len(others), 3)
        c.final = round(c.total + settings.tournament_weight * (c.pairwise_winrate - 0.5), 2)


def similarity(a: Candidate, b: Candidate) -> float:
    return jaccard(content_words(a.text), content_words(b.text))


def select_diverse(ranked: list[Candidate], reserve: list[Candidate] | None = None,
                   bold_slider: float = 0.5) -> list[tuple[str, Candidate]]:
    """BEST from the tournament; BOLDER / CHILLER from the finalists, else from the wider surviving pool."""
    if not ranked:
        return []
    best = ranked[0]
    bb = boldness(best)
    picks = [("BEST", best)]
    used = {best.id}

    def pick(label, pred, lean=0.0):
        source = ranked + [c for c in (reserve or []) if c not in ranked]
        pool = [c for c in source if c.id not in used and pred(c)
                and all(similarity(c, p) < 0.5 for _, p in picks)]
        if pool:
            # BOLDER / CHILLER should be noticeably different, not just the next-best score
            chosen = max(pool, key=lambda c: c.final + lean * (boldness(c) - bb))
            picks.append((label, chosen))
            used.add(chosen.id)
            return True
        return False

    if not pick("BOLDER", lambda c: boldness(c) > bb + 0.07, lean=40 + 120 * max(0.0, bold_slider - 0.5)):
        pick("BOLDER", lambda c: boldness(c) > bb, lean=40 + 120 * max(0.0, bold_slider - 0.5))
    if not pick("CHILLER", lambda c: boldness(c) < bb - 0.07, lean=-40 - 120 * max(0.0, 0.5 - bold_slider)):
        pick("CHILLER", lambda c: boldness(c) < bb, lean=-40 - 120 * max(0.0, 0.5 - bold_slider))
    while len(picks) < 3 and pick("ALTERNATIVE", lambda c: True):
        pass
    order = {"BEST": 0, "BOLDER": 1, "CHILLER": 2, "ALTERNATIVE": 3}
    return sorted(picks, key=lambda p: order[p[0]])
