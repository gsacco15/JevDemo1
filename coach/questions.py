"""The Jev evaluation schema: many narrow questions instead of "is this a good message?".

This file is the main thing to review and edit together (per TypeSafe's guidance):
every question and its criteria live here. Questions refer to named fields of the
state object built in pipeline.py:

    roles               who "user" and "match" are
    mode                "reply" or "opener"
    conversation        [{speaker: "user"|"match", text}], oldest first
    match_profile       the match's profile text (may be null)
    conversation_state  our structured read of the conversation (strategy + candidates)
    user_style          how the user normally texts (candidates)
    chosen_move         the conversational move we are aiming for (candidates)
    candidate_message   the message being judged (candidates)
    candidate_a / _b    the two messages being compared (pairwise)
    profile_detail      one line from the profile (hooks)
"""

from .jev.types import choice, noul, score
from .schemas import STAGES, STRATEGIES

# ---------------------------------------------------------------------------
# Conversation-level questions (state extraction)
# ---------------------------------------------------------------------------

STAGE_CRITERIA = {
    "PROFILE_OPENER": "No messages yet; the user is writing a first message based on `match_profile`",
    "OPENING_EXCHANGE": "Only the first few messages have been exchanged; still breaking the ice",
    "EARLY_CONVERSATION": "A real back-and-forth has started but there is little rapport yet",
    "BUILDING_RAPPORT": "Several friendly exchanges; they are getting to know each other, not much flirting",
    "ACTIVE_FLIRTING": "Both sides are clearly flirting or teasing each other with romantic interest",
    "READY_TO_ESCALATE": "Rapport and interest are strong enough that exchanging numbers or suggesting a date would feel natural",
    "NUMBER_EXCHANGE": "They are in the middle of exchanging phone numbers or moving off the app",
    "DATE_PLANNING": "They are actively working out a date: time, place, or plans",
    "POST_DATE": "They have already met in person and are talking afterwards",
    "LOW_MOMENTUM": "The conversation is stalling: short or slow replies, little interest shown",
    "RECOVERY": "Something went awkward or wrong and the conversation needs to be repaired",
}
assert set(STAGE_CRITERIA) == set(STAGES)

TONE_CRITERIA = {
    "playful": "Light, jokey, having fun",
    "teasing": "Poking fun at the user or challenging them in a friendly way",
    "flirty": "Openly romantic or suggestive interest",
    "warm": "Kind, friendly, supportive",
    "curious": "Asking questions, wanting to learn about the user",
    "serious": "Earnest or thoughtful, not joking",
    "dry": "Deadpan or understated humor",
    "low_effort": "Minimal effort: very short, generic, or disengaged replies",
}

MATCH_BEHAVIOR_CRITERIA = {
    "teasing_challenge": "Teasing the user or issuing a playful challenge or doubt",
    "asking_question": "Asking the user a question that expects an answer",
    "sharing": "Telling the user something about themselves",
    "flirting": "Flirting or expressing romantic interest",
    "short_reply": "A very short reply that gives little to work with",
    "logistics": "Talking about plans, meeting up, times, places, or contact details",
    "compliment": "Complimenting the user",
    "no_message": "There is no message from the match yet",
}

LEVELS_ENGAGEMENT = [
    "Barely engaged: one-word or no replies, no questions, no effort",
    "Lukewarm: short replies, rarely asks anything back",
    "Engaged: normal-length replies, some questions or jokes back",
    "Very engaged: long or enthusiastic replies, asks questions, builds on what the other person says",
]

STATE_QUESTIONS = [
    choice("stage", "Which stage best describes this dating-app `conversation` right now?", STAGE_CRITERIA),
    choice("tone", "What is the dominant tone of the match's most recent messages in `conversation`?", TONE_CRITERIA),
    choice("match_behavior", "What is the match doing in their most recent message in `conversation`?", MATCH_BEHAVIOR_CRITERIA),
    choice("momentum", "How is the momentum of the `conversation` trending?", {
        "positive": "Picking up: replies are getting longer, warmer, or more frequent",
        "neutral": "Steady: no clear change",
        "fading": "Dying down: replies are getting shorter, slower, or less interested",
    }),
    score("match_engagement", "How engaged is the match (speaker \"match\") in the `conversation`?", LEVELS_ENGAGEMENT),
    score("user_engagement", "How engaged is the user (speaker \"user\") in the `conversation`?", LEVELS_ENGAGEMENT),
    score("flirt_level", "How flirtatious has the `conversation` been so far, from both sides?", [
        "Not flirty at all: purely friendly or logistical",
        "Slightly flirty: light compliments or playful hints",
        "Clearly flirty: obvious mutual romantic interest",
        "Very flirty: intense, suggestive, or heavy flirting",
    ]),
    score("escalation_readiness", "How ready is the `conversation` to move toward exchanging numbers or meeting in person?", [
        "Not ready: barely started or the match seems uninterested",
        "Too early: some rapport but asking now would feel rushed",
        "Getting there: good rapport; asking soon would be natural",
        "Ready: strong mutual interest; asking now would feel natural or overdue",
    ]),
    score("need_question", "How much does the `conversation` need the user to ask the match a question in their next message?", [
        "Not at all: the match just asked something, or a question would feel like an interview",
        "Optional: a question could help but a statement or tease works just as well",
        "Needed: without a question the match has little to reply to",
    ]),
    noul("number_appropriate", "Given the `conversation`, it would feel natural and not pushy for the user to ask for the match's phone number in their next message."),
    noul("date_appropriate", "Given the `conversation`, it would feel natural and not pushy for the user to suggest meeting in person in their next message."),
]

STRATEGY_CRITERIA = {
    "CONTINUE_TOPIC": "Keep going on the current topic",
    "TEASE": "Tease back or playfully challenge the match",
    "ASK_QUESTION": "Ask the match a specific question",
    "ANSWER_AND_REDIRECT": "Answer the match's question, then turn it back to them",
    "FLIRT": "Add a bit of flirting",
    "ESCALATE_FLIRT": "Turn the flirting up noticeably",
    "CHANGE_TOPIC": "Move to a fresh topic because the current one has run out",
    "ASK_FOR_NUMBER": "Ask for the match's phone number",
    "ASK_FOR_DATE": "Suggest meeting up in person",
    "SUGGEST_SPECIFIC_DATE": "Propose a specific plan, place, or time",
    "PULL_BACK": "Ease off and send something low-pressure",
    "WAIT": "Send nothing yet; the user has already sent the last message(s)",
    "FOLLOW_UP": "The user already sent the last message, but should send a short follow-up to it (fix a mistake, add missing info)",
    "CLARIFY": "Ask what the match meant because their message was ambiguous",
    "END_CONVERSATION": "Politely wrap up the conversation",
}
assert set(STRATEGY_CRITERIA) == set(STRATEGIES)

# Asked only when the user sent the most recent message: is it their turn at all?
FOLLOWUP_REASONS = {
    "none": "No good reason: another message now would just be chasing or double-texting; the user should wait for a reply",
    "fix_mistake": "The user's own last message has a mistake, typo, or wrong detail (e.g. a day, time, name) worth correcting",
    "add_missing_info": "The user's last message left out information the match clearly needs, like a time, place, or how to meet",
    "answer_skipped": "The match asked the user something earlier that the user never answered",
}
FOLLOWUP_QUESTION = choice(
    "followup_reason",
    "The user sent the most recent message(s) in `conversation` and the match has not replied yet. Is there a good "
    "reason for the user to send another message before the match replies? Wanting to keep the conversation going "
    "is not by itself a good reason.",
    FOLLOWUP_REASONS,
)

STRATEGY_QUESTION = choice(
    "strategy",
    "Given the `conversation` and `conversation_state`, which move should the user make in their next message "
    "to keep the match interested without seeming needy or pushy?",
    STRATEGY_CRITERIA,
)

HOOK_QUESTION = score(
    "hook_strength",
    "How good is `profile_detail` as the basis for a specific, fun first message to this person?",
    [
        "Useless: nothing to talk about, or only about physical appearance",
        "Weak: generic detail most profiles share (likes travel, food, music)",
        "Decent: a specific interest or fact that invites a question",
        "Strong: specific, opinionated, funny, or unusual; easy to tease or riff on",
    ],
)
HOOK_APPEARANCE = noul("hook_is_appearance", "`profile_detail` is only about the person's physical appearance.")

# ---------------------------------------------------------------------------
# Candidate-level questions. All refer to `candidate_message`.
# ---------------------------------------------------------------------------

LEVELS_4 = lambda none, low, mid, high: [none, low, mid, high]  # noqa: E731

CANDIDATE_QUESTIONS = [
    # message quality
    score("relevance", "How well does `candidate_message` fit as the next message after the most recent message in "
          "`conversation`? If the match sent it, does the candidate respond to it; if the user sent it (a follow-up), "
          "does the candidate make sense on top of it? When `mode` is opener, judge fit to `match_profile`.", LEVELS_4(
              "Ignores it entirely; could follow any message",
              "Loosely connected",
              "Clearly responds to it",
              "Responds to it precisely and builds on it")),
    score("specificity", "How specific is `candidate_message` to this particular match and `conversation`?", LEVELS_4(
        "Completely generic; could be sent to anyone",
        "Slightly tailored",
        "Clearly about this person or conversation",
        "Only makes sense for this exact person and moment")),
    score("naturalness", "How natural does `candidate_message` sound as a text a real person would send on a dating app?", LEVELS_4(
        "Robotic, formal, or obviously AI-written",
        "A bit stiff or scripted",
        "Sounds like a normal person texting",
        "Effortless; sounds exactly like a real text")),
    score("replyability", "How easy and inviting is `candidate_message` for the match to reply to?", LEVELS_4(
        "Conversation dead end; nothing to reply to",
        "Possible to reply but takes effort",
        "Easy to reply to",
        "Almost impossible not to reply; obvious fun opening")),
    # social dynamics
    score("confidence", "How confident does the user come across in `candidate_message`?", LEVELS_4(
        "Insecure, apologetic, or hesitant",
        "Somewhat unsure",
        "Comfortable and self-assured",
        "Very confident and bold")),
    score("playfulness", "How playful or teasing is `candidate_message`?", LEVELS_4(
        "Completely serious", "Slightly light-hearted", "Playful", "Very playful or teasing")),
    score("flirt", "How flirtatious is `candidate_message`?", LEVELS_4(
        "Not flirty; purely friendly", "Hint of flirting", "Clearly flirty", "Very flirty or suggestive")),
    score("warmth", "How warm and kind does `candidate_message` feel toward the match?", LEVELS_4(
        "Cold or dismissive", "Neutral", "Friendly", "Very warm and caring")),
    score("humor", "How funny is `candidate_message` likely to be to the match?", LEVELS_4(
        "Not funny or not trying to be", "Mildly amusing", "Funny", "Very funny")),
    score("directness", "How direct and straightforward is `candidate_message` about what the user wants?", LEVELS_4(
        "Vague or indirect", "Somewhat hints at intent", "Clear about intent", "Very direct and explicit about intent")),
    score("escalation", "How much does `candidate_message` push the relationship forward (toward numbers, dates, or more intimacy)?", LEVELS_4(
        "Not at all; stays where things are",
        "A small step forward",
        "A clear step forward, like suggesting a date",
        "A big leap forward")),
    # risks
    score("neediness", "Given the `conversation`, how needy or overly eager does `candidate_message` sound?", LEVELS_4(
        "Not needy at all; relaxed", "Slightly eager", "Noticeably needy or seeking validation", "Desperate")),
    score("pressure", "How much pressure does `candidate_message` put on the match to respond or agree?", LEVELS_4(
        "No pressure", "Mild pressure", "Noticeable pressure", "Pushy or demanding")),
    score("cringe", "How cringeworthy or try-hard would `candidate_message` feel to the match?", LEVELS_4(
        "Not at all cringe", "Slightly try-hard", "Cringe", "Very cringe or embarrassing")),
    score("sexual", "How sexually suggestive is `candidate_message`?", LEVELS_4(
        "Not sexual at all", "Mild innuendo", "Clearly sexual", "Explicit")),
    # personalization
    score("style_fit", "How well does `candidate_message` match the user's own texting style in `user_style` "
          "(length, tone, humor, emoji use, capitalization, and the example messages)?", LEVELS_4(
              "Nothing like how the user texts",
              "Somewhat off from their style",
              "Close to their style",
              "Sounds exactly like the user wrote it")),
    # yes / no checks
    noul("generic", "`candidate_message` could reasonably have been sent to almost anyone on a dating app."),
    noul("repeats", "`candidate_message` repeats a joke, question, or point the user already made in `conversation`."),
    noul("invented_info", "`candidate_message` mentions facts about the match that do not appear in `match_profile` or `conversation`."),
    noul("manipulative", "`candidate_message` tries to manipulate the match: guilt-tripping them, negging them "
         "(backhanded compliments meant to lower their confidence), or pressuring them emotionally. "
         "Self-deprecating humor about the user's own earlier message is not manipulation."),
    noul("insulting", "The match could reasonably feel genuinely insulted or belittled by `candidate_message` "
         "(not playful teasing, and not the user joking about themselves)."),
    noul("pickup_line", "`candidate_message` is an obviously canned pickup line."),
    noul("asks_known_info", "`candidate_message` asks the match for information they already gave in `match_profile` or `conversation`."),
    noul("asks_question", "`candidate_message` asks the match a question."),
    noul("asks_number", "`candidate_message` asks for the match's phone number or another way to contact them off the app."),
    noul("asks_date", "`candidate_message` suggests meeting in person."),
]

RISK_DIMS = {"neediness", "pressure", "cringe", "sexual", "generic", "repeats", "invented_info",
             "manipulative", "insulting", "pickup_line", "asks_known_info"}

PAIRWISE_QUESTION = choice(
    "pairwise",
    "Both messages are possible next messages from the user in `conversation`. Which one is the stronger message "
    "for this exact moment, given `conversation_state` and the user's own style in `user_style`?",
    {"A": "`candidate_a` is the stronger message", "B": "`candidate_b` is the stronger message"},
)

REASONING_DIMS = {q.id: q for q in CANDIDATE_QUESTIONS}
