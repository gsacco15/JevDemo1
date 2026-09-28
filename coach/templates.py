"""Offline candidate generator (used when no ANTHROPIC_API_KEY is set).

Produces a deliberately mixed pool - good, mediocre and bad candidates - so the
rejection and ranking layers have something real to do.
"""

import random
import re

from .jev.heuristics import STOPWORDS, content_words, proper_nouns

TEASE = [
    ("Big claim for someone who hasn't tested it yet", "balanced"),
    ("Guess there's only one way to find out", "balanced"),
    ("You sound very confident for someone operating on zero evidence", "balanced"),
    ("Dangerous amount of confidence over there", "chill"),
    ("That's a suspicious amount of confidence 😂", "chill"),
    ("Bold words. I'm going to need a demonstration", "bold"),
    ("Respectfully, I've been waiting for someone to say that to me", "balanced"),
    ("I'm writing this down so I can bring it up when you lose", "bold"),
    ("Noted. Filing this under famous last words", "chill"),
]
TOPIC_TEASE = [
    ("Okay but your {topic} take is a little suspicious", "chill"),
    ("I have concerns about the {topic} thing and I'm prepared to defend them", "balanced"),
    ("Hot take incoming about {topic}. Brace yourself", "balanced"),
    ("{Topic}? Bold. Explain yourself", "chill"),
]
QUESTIONS = [
    ("Important question before this goes any further: how serious are we talking about {topic}?", "balanced"),
    ("Okay I need the full {topic} story", "chill"),
    ("What's the most unhinged thing {topic} has made you do?", "balanced"),
    ("Scale of casual to friendship ending, where does your {topic} energy sit?", "balanced"),
    ("So what got you into {topic}?", "chill"),
]
ANSWER_REDIRECT = [
    ("Honestly? Depends who's asking. Your turn though - {topic}?", "balanced"),
    ("Short answer yes, long answer I'll tell you over a drink. What about you?", "bold"),
    ("Guilty. But I feel like you're hiding a worse answer", "balanced"),
    ("Ha, fair question. I'll answer if you tell me yours first", "chill"),
]
FLIRT = [
    ("You're making it very hard to play it cool here", "balanced"),
    ("Careful, I'm starting to like you", "bold"),
    ("Okay you're kind of fun, I'll allow it", "chill"),
]
CONTINUE = [
    ("Wait, go back to the {topic} part", "chill"),
    ("Ok the {topic} thing is actually impressive", "chill"),
    ("Now I'm curious - is {topic} a lifelong thing or a recent obsession?", "balanced"),
]
CHANGE = [
    ("Completely unrelated but I need your most controversial food opinion", "balanced"),
    ("New topic: what's the best thing you've done this month?", "chill"),
    ("Settle something for me - best breakfast spot around here?", "chill"),
]
NUMBER = [
    ("Send me your number and we'll settle this properly", "bold"),
    ("This is too good for the app. What's your number?", "bold"),
    ("I feel like this deserves a real text thread - can I get your number?", "balanced"),
]
DATE = [
    ("Okay you have to show me this {topic} situation in person", "bold"),
    ("Prove it. {Topic}, this week?", "bold"),
    ("Let's settle this over a drink this week", "bold"),
    ("I think we need to take this debate somewhere with drinks. Thursday?", "bold"),
    ("We should grab a coffee and continue this in person", "balanced"),
]
BAD = [
    ("hey beautiful 😍", "chill"),
    ("haha yeah", "chill"),
    ("Are you from Tennessee? Because you're the only 10 I see", "bold"),
    ("Why aren't you replying faster? I thought we had something", "bold"),
    ("You're so hot, come over to my place tonight", "bold"),
    ("Sorry if this is weird, I'm not sure what to say lol, maybe we could talk more if you want?", "chill"),
    ("Most girls wouldn't get my humor so you'd be lucky to keep up", "bold"),
    ("Hey how's your day going?", "chill"),
    ("That is truly a delightful sentiment and furthermore I must say I find your perspective utterly fascinating and would love to hear more about everything", "balanced"),
]

ALIASES = {"TEASE": TEASE + TOPIC_TEASE, "ASK_QUESTION": QUESTIONS, "ANSWER_AND_REDIRECT": ANSWER_REDIRECT,
           "FLIRT": FLIRT, "ESCALATE_FLIRT": FLIRT + DATE, "CONTINUE_TOPIC": CONTINUE, "CHANGE_TOPIC": CHANGE,
           "ASK_FOR_NUMBER": NUMBER, "ASK_FOR_DATE": DATE, "SUGGEST_SPECIFIC_DATE": DATE,
           "PULL_BACK": CHANGE, "WAIT": CHANGE, "CLARIFY": QUESTIONS, "END_CONVERSATION": CHANGE}
STRAT_OF = {id(t): s for s, pool in [("TEASE", TEASE + TOPIC_TEASE), ("ASK_QUESTION", QUESTIONS),
                                        ("ANSWER_AND_REDIRECT", ANSWER_REDIRECT), ("FLIRT", FLIRT),
                                        ("CONTINUE_TOPIC", CONTINUE), ("CHANGE_TOPIC", CHANGE),
                                        ("ASK_FOR_NUMBER", NUMBER), ("ASK_FOR_DATE", DATE)] for t in pool}


NOT_TOPICS = {"definitely", "really", "actually", "literally", "totally", "honestly", "probably", "couldn't",
              "couldnt", "keep", "think", "know", "want", "going", "thing", "things", "something", "anything",
              "never", "always", "sure", "maybe", "good", "great", "best", "worst", "better", "pretty", "kind",
              "sounds", "said", "tell", "make", "made", "time", "people", "someone", "guess", "right", "wrong"}


def _ok(w: str) -> bool:
    return w not in NOT_TOPICS and w not in STOPWORDS and not w.endswith("ly") and len(w) > 2


def extract_topic(*texts: str) -> str | None:
    """Best-effort noun-ish topic from the given texts, in priority order."""
    texts = [t for t in texts if t]
    for text in texts:
        m = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b", text)
        if m:
            return m[0]
        after = re.search(r"\b(?:at|about|love|into|obsessed with|addicted to|playing|play|of)\s+([a-z][a-z' ]{2,24})", text.lower())
        if after:
            phrase = " ".join(w for w in after.group(1).split()[:2] if _ok(w) or w in ("go",))
            if phrase and any(_ok(w) for w in phrase.split()):
                return phrase
        nouns = [n for n in proper_nouns(" " + text) if _ok(n)]
        if nouns:
            return sorted(nouns, key=len, reverse=True)[0].title()
    for text in texts:
        words = [w for w in content_words(text) if _ok(w) and len(w) > 3]
        if words:
            return max(words, key=len)
    return None


def generate(ctx: dict, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    n = ctx["n"]
    if ctx["mode"] == "opener":
        topic = extract_topic((ctx.get("hooks") or [{}])[0].get("text"), ctx.get("profile"))
    else:
        recent = "\n".join(reversed((ctx.get("conversation") or "").splitlines()[-4:]))
        topic = extract_topic(ctx.get("last_match"), recent)
    topic = topic or "that"
    avoid = {t.lower() for t in ctx.get("avoid_texts", [])}

    def fill(tpl: str) -> str:
        return tpl.replace("{topic}", topic).replace("{Topic}", topic[:1].upper() + topic[1:])

    main = ALIASES.get(ctx["strategy"], CONTINUE)
    alts = [t for s in ctx["alt_strategies"] for t in ALIASES.get(s, [])]
    if ctx["mode"] == "opener":
        main, alts = QUESTIONS + TOPIC_TEASE, CONTINUE + FLIRT + [BAD[0], BAD[2], BAD[7]]
    pool: list[tuple[str, str, str]] = []
    for tpl_list, share in ((main, 0.55), (alts, 0.25), (BAD, 0.2)):
        picks = list(tpl_list)
        rng.shuffle(picks)
        for tpl in picks[: max(1, round(n * share))]:
            strat = STRAT_OF.get(id(tpl), ctx["strategy"])
            pool.append((fill(tpl[0]), strat, tpl[1]))
    # top up from everything if short
    everything = [t for lst in ALIASES.values() for t in lst]
    rng.shuffle(everything)
    for tpl in everything:
        if len(pool) >= n:
            break
        pool.append((fill(tpl[0]), STRAT_OF.get(id(tpl), ctx["strategy"]), tpl[1]))

    # always seed a few bold options so the Bold slider has something to find
    if ctx["mode"] == "reply":
        wild = list(NUMBER + DATE + FLIRT)
        rng.shuffle(wild)
        for tpl in wild[:3]:
            pool.insert(len(pool) // 2, (fill(tpl[0]), STRAT_OF.get(id(tpl), ctx["strategy"]), tpl[1]))

    seen, out = set(), []
    for text, strat, bold in pool:
        key = text.lower()
        if key in seen or key in avoid:
            continue
        seen.add(key)
        out.append({"text": text, "strategy": strat, "boldness": bold})
    return out[:n]
