"""Claude: the creative layer (candidate generation), the slow brain (low-confidence
escalation), and the eyes (screenshot parsing). Only used when ANTHROPIC_API_KEY is set."""

import base64
import json
import logging

import anthropic

from .config import settings
from .schemas import STRATEGIES

log = logging.getLogger(__name__)

VOICE = """You are the writing engine inside a dating message coach. You write candidate text messages
the USER could send to their MATCH on a dating app.

Voice: socially perceptive, confident, fun, direct, modern, concise, nonjudgmental.
Never write pickup-artist material: no negging, no manipulation, no guilt, no "make them chase",
no canned pickup lines, nothing sexual unless the conversation is already clearly there.
Write like a real person texting: short, lowercase is fine, minimal emojis unless the user's style uses them.
Never invent facts about the match that are not in the profile or conversation.
The goal is to help the user sound like themselves, only sharper - not to invent a persona."""

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def _json_call(model: str, system: str, content, schema: dict, max_tokens: int = 8000, effort: str = "low") -> dict:
    resp = await client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "strategy": {"type": "string", "enum": STRATEGIES},
                    "boldness": {"type": "string", "enum": ["chill", "balanced", "bold"]},
                },
                "required": ["text", "strategy", "boldness"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


async def generate_candidates(ctx: dict) -> list[dict]:
    style = ctx["style"]
    samples = "\n".join(f"- {s}" for s in style.get("sample_messages", [])[:8]) or "(none provided)"
    alt = ", ".join(ctx["alt_strategies"]) or "none"
    avoid = "\n".join(f"- {t}" for t in ctx.get("avoid_texts", [])[:40]) or "(none)"
    hooks = "\n".join(f"- [{h['strength']:.2f}] {h['text']}" for h in ctx.get("hooks", [])[:4]) or "(n/a)"
    if ctx.get("focus"):
        spread = (f"Write {ctx['n']} distinct candidate messages. THIS BATCH'S FOCUS: {ctx['focus']}\n"
                  "(Other batches are covering the other angles in parallel, so stay on your focus.)")
    else:
        spread = (f"Write {ctx['n']} distinct candidate messages. Spread them out on purpose:\n"
                  "- about 60% use the recommended move, the rest use the alternatives\n"
                  "- roughly a third each chill / balanced / bold")
    turn = ""
    if ctx.get("last_speaker") == "user":
        reason = {"fix_mistake": "fix the mistake/typo in the user's own last message",
                  "add_missing_info": "add the information the user's last message left out",
                  "answer_skipped": "answer the question from the match that the user skipped"}.get(ctx.get("followup_reason") or "")
        turn = ("\nTURN: The USER sent the last message and the match has not replied yet. These are FOLLOW-UPS sent on "
                "top of the user's own message, not replies to the match's older message. "
                + (f"Each one should {reason}, quickly and lightly." if reason else
                   "There is no strong reason to send more, so every option must be very low-pressure and optional.") + "\n")
    prompt = f"""MODE: {ctx['mode']}  (reply = respond to the conversation; opener = first message on a profile)
PLATFORM: {ctx['platform']}
MATCH PRONOUN: {ctx['pronoun']}

MATCH PROFILE:
{ctx.get('profile') or '(not provided)'}

BEST PROFILE HOOKS (strength-ranked):
{hooks}

CONVERSATION (oldest first):
{ctx.get('conversation') or '(no messages yet)'}

CONVERSATION STATE (from our judgment layer):
{json.dumps(ctx['state_summary'], indent=1)}

{turn}RECOMMENDED MOVE: {ctx['strategy']}
ALSO WORTH EXPLORING: {alt}

USER STYLE:
{json.dumps({k: v for k, v in style.items() if k != 'sample_messages'})}
Messages the user actually sent before:
{samples}

TONE SLIDERS (0..1, 0.5 neutral): {json.dumps(ctx['sliders'])}

DO NOT REPEAT OR CLOSELY PARAPHRASE THESE (already considered):
{avoid}

{spread}
- vary length, structure and angle; do not write near-duplicates
- each candidate is a single message the user could send as-is
Tag each with the move it uses and its boldness."""
    data = await _json_call(settings.generator_model, VOICE, prompt, GEN_SCHEMA, max_tokens=8000, effort="low")
    return data.get("candidates", [])


REASON_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "value": {"type": "number"}},
                "required": ["id", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}


async def resolve_low_confidence(context: dict, asks: list[dict]) -> dict[str, float]:
    """System Two. `asks` = [{id, candidate, question, kind, levels}] -> {id: value in 0..1}."""
    lines = []
    for a in asks:
        scale = f" | levels (low->high): {a['levels']}" if a.get("levels") else ""
        lines.append(f"{a['id']} | {a['kind']} | candidate_message: {a['candidate']!r} | question: {a['question']}{scale}")
    ctx = {k: v for k, v in context.items() if k != "candidate_message"}
    prompt = f"""A fast judgment model was unsure about the following narrow judgments. Decide each carefully.

STATE (JSON):
{json.dumps(ctx, ensure_ascii=False, indent=1)}

For kind=noul answer the probability (0..1) that the statement is true.
For kind=score answer the position on the listed levels as 0..1 (0 = first level, 1 = last level).

JUDGMENTS:
""" + "\n".join(lines)
    data = await _json_call(settings.reasoning_model, "You are a careful social-dynamics analyst.", prompt,
                            REASON_SCHEMA, max_tokens=4000, effort="low")
    return {a["id"]: max(0.0, min(1.0, float(a["value"]))) for a in data.get("answers", [])}


PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["conversation", "profile", "both"]},
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string", "enum": ["user", "match"]}, "text": {"type": "string"}},
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        },
        "profile_text": {"type": "string"},
    },
    "required": ["kind", "messages", "profile_text"],
    "additionalProperties": False,
}


async def parse_screenshot(image: bytes, media_type: str) -> dict:
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                     "data": base64.standard_b64encode(image).decode()}},
        {"type": "text", "text": """This is a screenshot from a dating app (usually Hinge).
If it shows a chat: extract every message in order. Bubbles on the RIGHT (usually colored) are the app
user ("user"); bubbles on the LEFT are the other person ("match"). Skip timestamps and UI chrome.
If it shows a profile: put all prompt answers, captions, job, location and notable photo details
(describe photos briefly, e.g. "photo: hiking in Patagonia") into profile_text, one per line.
Leave fields empty when not applicable."""},
    ]
    return await _json_call(settings.generator_model, "You extract structured data from screenshots.", content,
                            PARSE_SCHEMA, max_tokens=4000, effort="low")


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "string"},
        "basics": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "value": {"type": "string"}}, "required": ["label", "value"], "additionalProperties": False}},
        "prompts": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "answer": {"type": "string"}}, "required": ["title", "answer"], "additionalProperties": False}},
        "photos": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "basics", "prompts", "photos"],
    "additionalProperties": False,
}


async def structure_profile(text: str) -> dict:
    """Messy pasted dating profile (any app, any order) -> structured fields. Used only when the rule parser
    can't make sense of the paste."""
    system = """You turn pasted dating-app profiles (Hinge, Tinder, Bumble, OkCupid, notes, anything) into fields.
- name/age: empty string if not given.
- basics: short facts with a Title Case label: Location, Hometown, Job, Education, Height, Interests, Pets,
  Drinking, Smoking, Kids, Zodiac, Religion, Politics, Dating intention, Languages, Exercise, ...
- prompts: prompt/question titles with her answers, and bio text (title "About me"). Keep her exact words.
- photos: one short description per photo.
Ignore meta chatter from whoever pasted it ("here's this girl's profile", "lol"). Never invent anything."""
    return await _json_call(settings.generator_model, system, [{"type": "text", "text": text[:8000]}],
                            PROFILE_SCHEMA, max_tokens=3000, effort="low")


async def profile_from_screenshots(images: list[tuple[bytes, str]], existing: str = "") -> dict:
    """Several screenshots of ONE dating profile (scrolled, overlapping, any order) -> structured profile.
    Anything already captured in `existing` is kept and merged. Images are only held in memory."""
    content = [{"type": "image", "source": {"type": "base64", "media_type": mt,
                                            "data": base64.standard_b64encode(data).decode()}} for data, mt in images]
    note = f"\n\nAlready captured from earlier screenshots (keep all of it, merge new details in):\n{existing[:6000]}" \
        if existing.strip() else ""
    content.append({"type": "text", "text": f"""These {len(images)} screenshots are all from ONE person's dating profile
(Hinge, Tinder, Bumble or similar), taken while scrolling, so they overlap and may be out of order.
Capture everything on the profile exactly once:
- name and age
- basics, using these labels where they fit: Location, Hometown, Job, Education, Height, Dating intention,
  Relationship type, Drinking, Smoking, Weed, Kids, Pets, Religion, Politics, Zodiac, Languages, Exercise,
  Gender, Sexuality, Pronouns, Interests (comma-separated)
- every prompt: its title and her answer word for word; a bio goes under the title "About me";
  a voice prompt without visible text becomes answer "(voice prompt)"
- every photo: one line on what she's doing, where, and notable objects, pets or places
  (e.g. "holding a golden retriever at Zilker Park"). Include visible captions. Don't rate looks.
Ignore app buttons, likes, comments, timestamps and ads. Never invent anything you can't see.{note}"""})
    return await _json_call(settings.generator_model, "You transcribe dating profiles from screenshots into fields.",
                            content, PROFILE_SCHEMA, max_tokens=4000, effort="low")
