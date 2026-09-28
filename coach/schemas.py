"""Core data model. Platform-agnostic: Hinge is just the first `platform` value."""

from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["user", "match"]

STAGES = [
    "PROFILE_OPENER",
    "OPENING_EXCHANGE",
    "EARLY_CONVERSATION",
    "BUILDING_RAPPORT",
    "ACTIVE_FLIRTING",
    "READY_TO_ESCALATE",
    "NUMBER_EXCHANGE",
    "DATE_PLANNING",
    "POST_DATE",
    "LOW_MOMENTUM",
    "RECOVERY",
]

STRATEGIES = [
    "CONTINUE_TOPIC",
    "TEASE",
    "ASK_QUESTION",
    "ANSWER_AND_REDIRECT",
    "FLIRT",
    "ESCALATE_FLIRT",
    "CHANGE_TOPIC",
    "ASK_FOR_NUMBER",
    "ASK_FOR_DATE",
    "SUGGEST_SPECIFIC_DATE",
    "PULL_BACK",
    "WAIT",
    "CLARIFY",
    "END_CONVERSATION",
]


class Message(BaseModel):
    speaker: Speaker
    text: str


class Sliders(BaseModel):
    """All sliders are 0..1 with 0.5 = neutral."""

    bold: float = Field(0.5, ge=0, le=1, description="Chill (0) <-> Bold (1)")
    flirty: float = Field(0.5, ge=0, le=1, description="Friendly (0) <-> Flirty (1)")
    serious: float = Field(0.5, ge=0, le=1, description="Playful (0) <-> Serious (1)")
    direct: float = Field(0.5, ge=0, le=1, description="Safe (0) <-> Direct (1)")
    expressive: float = Field(0.5, ge=0, le=1, description="Short (0) <-> Expressive (1)")


class StyleProfile(BaseModel):
    average_message_length: Literal["short", "medium", "long"] = "short"
    preferred_directness: float = 0.6
    preferred_flirt_level: float = 0.5
    emoji_frequency: float = 0.15
    question_frequency: float = 0.35
    humor_style: list[str] = Field(default_factory=lambda: ["playful", "teasing"])
    avoids: list[str] = Field(default_factory=lambda: ["pickup_line_style", "long_explanations", "excessive_emojis"])
    sample_messages: list[str] = Field(default_factory=list, description="A few messages the user actually sent")


class Hook(BaseModel):
    source: str
    subject: str
    text: str
    strength: float = 0.0
    is_appearance: bool = False


class ConversationState(BaseModel):
    platform: str = "hinge"
    stage: str = "EARLY_CONVERSATION"
    message_count: int = 0
    match_engagement: float = 0.5
    user_engagement: float = 0.5
    tone: list[str] = Field(default_factory=list)
    match_behavior: str = "sharing"
    topics: list[str] = Field(default_factory=list)
    questions_recently_asked_by_user: int = 0
    questions_recently_asked_by_match: int = 0
    flirt_level: float = 0.3
    escalation_readiness: float = 0.3
    need_question: float = 0.5
    number_requested: bool = False
    date_requested: bool = False
    number_appropriate: float = 0.2
    date_appropriate: float = 0.2
    conversation_momentum: str = "neutral"
    hooks: list[Hook] = Field(default_factory=list)
    last_match_message: str | None = None
    last_speaker: str | None = None


class Judgment(BaseModel):
    question_id: str
    kind: Literal["noul", "score", "choice"]
    value: float = 0.0  # noul: P(true); score: expected level in 0..1; choice: prob of top option
    choice: str | None = None
    probs: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    source: Literal["jev", "heuristic", "reasoning"] = "heuristic"


class Candidate(BaseModel):
    id: str
    text: str
    intended_strategy: str | None = None
    intended_boldness: str | None = None  # chill / balanced / bold (generator's own tag)
    judgments: dict[str, Judgment] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)  # deterministic: words, emojis, questions
    rejected: bool = False
    reject_reasons: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0
    pairwise_winrate: float | None = None
    final: float = 0.0


class CoachRequest(BaseModel):
    mode: Literal["reply", "opener"] = "reply"
    conversation_text: str | None = None
    messages: list[Message] | None = None
    profile_text: str | None = None
    sliders: Sliders = Field(default_factory=Sliders)
    num_candidates: int | None = None
    match_pronoun: Literal["she", "he", "they"] = "they"
    platform: str = "hinge"


class RerankRequest(BaseModel):
    session_id: str
    sliders: Sliders


class FeedbackRequest(BaseModel):
    session_id: str
    candidate_id: str
    kind: Literal[
        "copy",
        "select",
        "like",
        "dislike",
        "too_much",
        "too_boring",
        "not_me",
        "too_long",
        "too_cheesy",
        "more_direct",
        "less_direct",
        "edited",
    ]
    edited_text: str | None = None
