"""The Jev evaluation schema: many narrow questions instead of "is this a good message?".

This catalogue is a core product asset - tune wording here, not in the ranking code.
"""

from .jev.types import choice, noul, score
from .schemas import STAGES, STRATEGIES

# ---------------------------------------------------------------------------
# Conversation-level questions (state extraction + strategy selection)
# ---------------------------------------------------------------------------

TONES = ["playful", "teasing", "flirty", "warm", "curious", "serious", "dry", "low_effort"]
MATCH_BEHAVIORS = [
    "teasing_challenge",
    "asking_question",
    "sharing",
    "flirting",
    "short_reply",
    "logistics",
    "compliment",
    "no_message",
]

STATE_QUESTIONS = [
    choice("stage", "Which stage best describes this conversation right now?", STAGES),
    choice("tone", "What is the dominant tone of the match's recent messages?", TONES),
    choice("match_behavior", "What is the match doing in their most recent message?", MATCH_BEHAVIORS),
    choice("momentum", "How is the conversation's momentum?", ["positive", "neutral", "fading"]),
    score("match_engagement", "How engaged is the match in this conversation?"),
    score("user_engagement", "How engaged is the user in this conversation?"),
    score("flirt_level", "How flirtatious has the conversation been so far?"),
    score("escalation_readiness", "How ready is this conversation to move toward exchanging numbers or meeting?"),
    score("need_question", "How much does the conversation need the user to ask a question right now?"),
    noul("number_appropriate", "Would asking for the match's phone number be appropriate at this point?"),
    noul("date_appropriate", "Would suggesting an in-person date be appropriate at this point?"),
]

STRATEGY_QUESTION = choice(
    "strategy",
    "Given the conversation state, which conversational move should the user make next?",
    STRATEGIES,
)

HOOK_QUESTION = score(
    "hook_strength",
    "How promising is this profile detail as a hook for a specific, fun first message?",
)
HOOK_APPEARANCE = noul("hook_is_appearance", "Is this detail only about the person's physical appearance?")

# ---------------------------------------------------------------------------
# Candidate-level questions. Each has a `polarity` used for display
# ("good" dims are shown as-is, "risk" dims are shown as risk scores).
# ---------------------------------------------------------------------------

CANDIDATE_QUESTIONS = [
    # message quality
    score("relevance", "How directly does the candidate respond to the match's last message (or profile, for an opener)?"),
    score("specificity", "How specific is the candidate to this person and conversation?"),
    score("naturalness", "How natural does the candidate sound, like something a real person would text?"),
    score("replyability", "How easy and inviting is the candidate for the match to reply to?"),
    # social dynamics
    score("confidence", "How confident does the sender come across in the candidate?"),
    score("playfulness", "How playful is the candidate?"),
    score("flirt", "How flirtatious is the candidate?"),
    score("warmth", "How warm and friendly is the candidate?"),
    score("humor", "How funny is the candidate?"),
    score("directness", "How direct and straightforward is the candidate about what the sender wants?"),
    score("escalation", "How much does the candidate push things forward (toward numbers, dates, or more intimacy)?"),
    # risks
    score("neediness", "Does the candidate sound overly eager or needy relative to the conversation state?"),
    score("pressure", "How much pressure does the candidate put on the match?"),
    score("cringe", "How cringeworthy or try-hard is the candidate?"),
    score("sexual", "How sexually suggestive is the candidate?"),
    # personalization
    score("style_fit", "How consistent is the candidate with the user's own communication style and example messages?"),
    # yes / no checks
    noul("generic", "Could this candidate reasonably have been sent to almost anyone?"),
    noul("repeats", "Does the candidate repeat a joke, question, or point the user already made in this conversation?"),
    noul("invented_info", "Does the candidate reference details about the match that are not in the profile or conversation?"),
    noul("manipulative", "Does the candidate use manipulative framing, guilt, or negging?"),
    noul("insulting", "Could the candidate reasonably be read as a genuine insult rather than playful teasing?"),
    noul("pickup_line", "Is the candidate an obviously canned pickup line?"),
    noul("asks_known_info", "Does the candidate ask for information the match already provided?"),
    noul("asks_question", "Does the candidate ask the match a question?"),
    noul("asks_number", "Does the candidate ask for the match's phone number or another way to contact them?"),
    noul("asks_date", "Does the candidate suggest meeting in person?"),
]

RISK_DIMS = {"neediness", "pressure", "cringe", "sexual", "generic", "repeats", "invented_info",
             "manipulative", "insulting", "pickup_line", "asks_known_info"}

PAIRWISE_QUESTION = choice(
    "pairwise",
    "Which of the two candidate replies is the stronger message for this exact moment, for this user?",
    ["A", "B"],
)

REASONING_DIMS = {q.id: q for q in CANDIDATE_QUESTIONS}
