"""Human-facing language: conversation read, recommended move, and per-suggestion reasons.

Rule: describe fit, never promise outcomes ("91% chance of reply" is off limits).
"""

from .ranking import v
from .schemas import Candidate, ConversationState

PRONOUNS = {
    "she": {"subj": "she", "obj": "her", "poss": "her", "be": "she's", "refl": "herself"},
    "he": {"subj": "he", "obj": "him", "poss": "his", "be": "he's", "refl": "himself"},
    "they": {"subj": "they", "obj": "them", "poss": "their", "be": "they're", "refl": "themself"},
}

STRATEGY_LABEL = {
    "CONTINUE_TOPIC": "Keep the current topic going",
    "TEASE": "Tease back",
    "ASK_QUESTION": "Ask a question",
    "ANSWER_AND_REDIRECT": "Answer, then turn it back",
    "FLIRT": "Flirt a little",
    "ESCALATE_FLIRT": "Turn up the flirting",
    "CHANGE_TOPIC": "Change the topic",
    "ASK_FOR_NUMBER": "Ask for the number",
    "ASK_FOR_DATE": "Suggest meeting up",
    "SUGGEST_SPECIFIC_DATE": "Suggest a specific plan",
    "PULL_BACK": "Ease off a bit",
    "WAIT": "Wait before sending anything",
    "CLARIFY": "Clarify what they meant",
    "END_CONVERSATION": "Wrap it up kindly",
}

BEHAVIOR_PHRASE = {
    "teasing_challenge": "teasing you",
    "asking_question": "asking you something",
    "flirting": "flirting",
    "short_reply": "giving short replies",
    "logistics": "talking plans",
    "compliment": "complimenting you",
    "sharing": "sharing about {refl}",
    "no_message": "",
}


def level(x: float) -> str:
    return "High" if x >= 0.66 else "Medium" if x >= 0.4 else "Low"


def conversation_read(state: ConversationState, strategy: str, pronoun: str, mode: str) -> dict:
    p = PRONOUNS[pronoun]
    if mode == "opener":
        top = state.hooks[0].text if state.hooks else None
        summary = f"Best hook: “{top}”" if top else "No strong hook found - keep it light and specific."
        reason = "Open on something specific from the profile instead of a generic compliment."
    else:
        eng = state.match_engagement
        eng_word = "really engaged" if eng >= 0.66 else "engaged" if eng >= 0.45 else "not very engaged right now"
        beh = BEHAVIOR_PHRASE.get(state.match_behavior, "").format(**p)
        summary = f"{p['be'].capitalize()} {eng_word}" + (f" and {beh}." if beh else ".")
        if state.conversation_momentum == "fading":
            summary += " Momentum is dipping."
        reason = _move_reason(state, strategy, p)
    vibe = (state.tone[0] if state.tone else "neutral").replace("_", " ").title()
    return {
        "summary": summary,
        "vibe": vibe,
        "momentum": {"positive": "Strong", "neutral": "Steady", "fading": "Fading"}.get(state.conversation_momentum, "Steady"),
        "recommended_move": STRATEGY_LABEL.get(strategy, strategy.title()),
        "move_reason": reason,
        "escalation": level(state.escalation_readiness),
    }


def _move_reason(state: ConversationState, strategy: str, p: dict) -> str:
    no_q = state.need_question < 0.4
    if strategy == "TEASE":
        return "Match the playful energy." + (" No need to ask another question yet." if no_q else "")
    if strategy == "ANSWER_AND_REDIRECT":
        return f"{p['subj'].capitalize()} asked you something - answer it, then hand the conversation back."
    if strategy in ("ASK_FOR_NUMBER", "ASK_FOR_DATE", "SUGGEST_SPECIFIC_DATE"):
        return "The conversation has enough momentum to move it forward."
    if strategy == "CHANGE_TOPIC":
        return "This thread has run its course - give them something easier to reply to."
    if strategy == "WAIT":
        return ("Best move: don't send anything yet - you sent the last message. "
                "If you really want to follow up, these are the lowest-pressure options.")
    if strategy == "PULL_BACK":
        return "Don't chase. Keep the next message short and low-pressure."
    if strategy == "FLIRT":
        return f"{p['be'].capitalize()} opening the door - a little flirting fits here."
    if strategy == "ASK_QUESTION":
        return "Give them an easy, specific thing to respond to."
    return "Stay on what's working."


def why(c: Candidate, slot: str, strategy: str, desired: dict, pronoun: str, match_behavior: str | None = None) -> str:
    p = PRONOUNS[pronoun]
    parts = []
    if match_behavior in ("teasing_challenge", "flirting") and v(c, "playfulness") > 0.55:
        parts.append(f"matches {p['poss']} {'teasing' if match_behavior == 'teasing_challenge' else 'flirty'} energy")
    elif strategy == "TEASE" and v(c, "playfulness") > 0.55 and v(c, "relevance") > 0.6:
        parts.append(f"playfully riffs on what {p['subj']} said")
    elif strategy == "TEASE" and v(c, "playfulness") > 0.55:
        parts.append("adds some playful energy")
    elif v(c, "relevance") > 0.6:
        parts.append(f"picks up directly on what {p['subj']} said")
    elif v(c, "specificity") > 0.6:
        parts.append(f"stays specific to {p['obj']} instead of a copy-paste line")
    if v(c, "replyability") > 0.6:
        parts.append(f"gives {p['obj']} an easy way to keep it going")
    if v(c, "asks_number", 0.1) > 0.5:
        parts.append("goes for the number")
    elif v(c, "asks_date", 0.1) > 0.5:
        parts.append("moves things toward meeting up")
    elif v(c, "escalation") < desired["escalation"] - 0.1 or slot == "CHILLER":
        parts.append("keeps the escalation low-key")
    if v(c, "style_fit") > 0.7 and len(parts) < 3:
        parts.append("sounds like you")
    if not parts:
        parts.append("keeps things easy and natural")
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    text = text[0].upper() + text[1:]
    prefix = {"BOLDER": "Higher-escalation option. ", "CHILLER": "Lower-risk option. "}.get(slot, "")
    return prefix + text + "."


def labels(c: Candidate, desired: dict) -> list[str]:
    out = []
    if c.components.get("context_fit", 0) > 0.72:
        out.append("Strong fit for this moment")
    if v(c, "replyability") > 0.68:
        out.append("High replyability")
    if abs(v(c, "escalation") - desired["escalation"]) < 0.15:
        out.append("Appropriate escalation")
    if v(c, "style_fit") > 0.7:
        out.append("Matches your style")
    return out


def display_scores(c: Candidate) -> dict:
    pct = lambda d, n=0.5: round(100 * v(c, d, n))  # noqa: E731
    return {
        "Relevance": pct("relevance"),
        "Confidence": pct("confidence"),
        "Flirt": pct("flirt"),
        "Replyability": pct("replyability"),
        "Escalation": pct("escalation"),
        "Style fit": pct("style_fit"),
        "Cringe risk": pct("cringe", 0.2),
        "Neediness": pct("neediness", 0.2),
        "Genericness": pct("generic", 0.2),
    }
