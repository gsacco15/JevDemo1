"""Turn whatever profile text gets pasted into a structured profile.

Handles 'Key: value' pastes, prompt titles with the answer on the next line, unlabeled basics in any order
('5\\'4"', 'Gemini', 'Drinks sometimes', 'Austin, Texas'), photo sections and tab-separated photo tables,
and ignores meta lines like "Here's a fictional Hinge profile...". Pure text rules, no model calls, so the
UI preview and the hook ranker share one parser.
"""

import re

BASIC_KEYS = {"distance", "name", "age", "location", "job", "work", "job title", "education", "school", "height",
              "dating intention", "dating intentions", "politics", "drinking", "drinks", "smoking", "smokes",
              "exercise", "hometown", "religion", "religious beliefs", "kids", "children", "family plans", "pets",
              "zodiac", "star sign", "languages", "language", "gender", "pronouns", "sexuality", "relationship type",
              "ethnicity", "weed", "marijuana", "drugs", "covid vaccine"}
KEY_LABEL = {"work": "Job", "job title": "Job", "school": "Education", "drinks": "Drinking", "smokes": "Smoking",
             "star sign": "Zodiac", "children": "Kids", "family plans": "Kids", "language": "Languages",
             "dating intentions": "Dating intention", "religious beliefs": "Religion", "marijuana": "Weed"}
# section headers seen across apps (Hinge, Tinder, Bumble, ...) -> what the lines under them are
SECTIONS = {
    "photos": {"photo set", "photos", "pictures", "pics", "photo captions", "captions", "images", "my photos"},
    "bio": {"about me", "about", "my bio", "bio", "a little about me", "summary", "self-summary", "description",
            "my self-summary", "who i am", "self summary"},
    "interests": {"interests", "my interests", "into", "likes", "loves", "i'm into it", "passions", "hobbies", "things i like", "i'm into", "my passions",
                  "i'm interested in"},
    "basics": {"basics", "my basics", "vitals", "virtues", "lifestyle", "my lifestyle", "more about me",
               "essentials", "details", "info", "profile info", "my details", "the basics"},
    "prompts": {"hinge prompts", "prompts", "my prompts", "questions", "profile prompts"},
    "skip": {"profile", "image description", "generation prompt", "#", "my profile", "profile details"},
}
EMOJI_KEY = {"📍": "Location", "🏠": "Hometown", "🏡": "Hometown", "🎓": "Education", "💼": "Job", "📏": "Height",
             "🍷": "Drinking", "🍸": "Drinking", "🍺": "Drinking", "🚬": "Smoking", "🐶": "Pets", "🐱": "Pets",
             "🐾": "Pets", "👶": "Kids", "🙏": "Religion", "🗳": "Politics", "💬": "Languages", "🔍": "Looking for",
             "💘": "Looking for", "♀": "Gender", "♂": "Gender", "🏋": "Exercise", "💪": "Exercise"}

# Hinge prompt titles (lowercase, no trailing punctuation). Unknown titles are caught by the shape rule below.
PROMPTS = """typical sunday|my ideal sunday|my simple pleasures|dating me is like|i go crazy for|unusual skills
my most irrational fear|a shower thought i recently had|i'm looking for|green flags i look out for
the way to win me over is|i geek out on|a life goal of mine|my love language is|we'll get along if
i won't shut up about|two truths and a lie|best travel story|i recently discovered that|my greatest strength
the one thing i'd love to know about you is|i'm convinced that|together we could|the key to my heart is
don't hate me if i|this year, i really want to|i bet you can't|my happy place|worst idea i've ever had
first round is on me if|let's debate this topic|believe it or not, i|the hallmark of a good relationship is
i want someone who|change my mind about|biggest risk i've taken|i'm weirdly attracted to
we're the same type of weird if|a boundary of mine is|you should not go out with me if|i'll fall for you if
i'd fall for you if|all i ask is that you|my cry-in-the-car song is|give me travel tips for|my biggest date fail
what i order for the table|the dorkiest thing about me is|i'm overly competitive about
most spontaneous thing i've done|i take pride in|i'm a regular at|i know the best spot in town for
teach me something about|my therapist would say i|a random fact i love is|the last time i cried happy tears was
a perfect first date|i get along best with people who|i'm known for|my friends ask me for advice about
i'll know it's time to delete hinge when|i want someone who|i'm looking for someone who|let's make sure we're on the same page about
a non-negotiable|my golden rule|i feel most supported when|what if i told you that|our first date|best way to ask me out is by
you should leave a comment if|i'm the type of texter who|the most spontaneous thing i've done|my most useless skill
i hype myself up by|my mom would describe me as|my friends would describe me as|i beat my blues by
a review by a friend|i'll brag about you to my friends if|sunday are for|sundays are for|swipe right if|the best way to ask me out is by
my zombie apocalypse plan|what i'm doing with my life|i'm really good at|i spend a lot of time thinking about
favorite books, movies, shows, music, and food|six things i could never do without|on a typical friday night i am
you should message me if|the most private thing i'm willing to admit|i value|my favorite things|looking for|wants
my ideal date|ideal first date|perfect date|fun fact|fun facts|random facts|my love language|love language
if loving this is wrong, i don't want to be right|i'm secretly really good at""".replace("\n", "|").split("|")
PROMPTS = {p.strip() for p in PROMPTS if p.strip()}
# a title with an unknown wording still looks like a sentence stem: short, no end punctuation, dangling word
STEM_END = re.compile(r"\b(is|are|was|could|would|if|when|that|for|about|with|to|of|by|at|on|like|me|my|who|i|can't|can|should|in)$", re.I)

ZODIAC = {"aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn",
          "aquarius", "pisces"}
LANGS = {"english", "spanish", "french", "german", "italian", "portuguese", "mandarin", "cantonese", "chinese",
         "japanese", "korean", "hindi", "arabic", "russian", "dutch", "greek", "hebrew", "vietnamese", "tagalog",
         "polish", "turkish", "swedish", "norwegian", "danish", "urdu", "bengali", "punjabi", "farsi", "persian"}
BASIC_PATTERNS = [
    ("Height", re.compile(r"^\d\s*['’]\s*\d{1,2}\s*(\"|”|''|’’)?$|^\d{3}\s*cm$", re.I)),
    ("Hometown", re.compile(r"^(originally from|from|hometown|grew up in)\b", re.I)),
    ("Location", re.compile(r"^(lives in|living in|based in|located in)\b", re.I)),
    ("Distance", re.compile(r"^\d+(\.\d+)?\s*(miles?|mi|km|kilometers?)\s+away$", re.I)),
    ("Drinking", re.compile(r"\b(drinks?|drinker|drinking|sober)\b", re.I)),
    ("Smoking", re.compile(r"\b(smokes?|smoker|smoking)\b", re.I)),
    ("Dating intention", re.compile(r"^(something serious|something casual|marriage|relationship|nothing serious|new friends|still figuring (it|that) out)$", re.I)),
    ("Weed", re.compile(r"\b(weed|marijuana|cannabis|420)\b", re.I)),
    ("Drugs", re.compile(r"\bdrugs\b", re.I)),
    ("Kids", re.compile(r"\b(children|kids)\b|^not sure yet$", re.I)),
    ("Pets", re.compile(r"^(dog|cat) lover$|^(has|have|dog|cat)\b.*\b(dogs?|cats?|pets?|puppy|kitten|golden|retriever|lab)\b|^(dog|cat) (mom|dad|parent|person)$|^(dogs?|cats?)$", re.I)),
    ("Gender", re.compile(r"^(woman|man|non-?binary|trans woman|trans man|female|male)$", re.I)),
    ("Sexuality", re.compile(r"^(straight|gay|lesbian|bisexual|bi|queer|pansexual|asexual|heterosexual|homosexual)$", re.I)),
    ("Pronouns", re.compile(r"^(she|he|they)\s*/\s*(her|him|them)", re.I)),
    ("Dating intention", re.compile(r"^(long[- ]term|short[- ]term|life partner|figuring out|casual|long term relationship|short term relationship)", re.I)),
    ("Relationship type", re.compile(r"^(monogamy|non-?monogamy|monogamous|ethical non-?monogamy)", re.I)),
    ("Exercise", re.compile(r"^(active|very active|sometimes active|almost never|gym rat|works? out \w+|exercises? \w+)$", re.I)),
    ("Religion", re.compile(r"^(christian|catholic|jewish|muslim|hindu|buddhist|agnostic|atheist|spiritual|mormon|sikh)$", re.I)),
    ("Politics", re.compile(r"^(liberal|moderate|conservative|not political|progressive|apolitical|politically \w+)$", re.I)),
    ("Education", re.compile(r"\b(university|college|school|institute|academy|polytechnic|mba|phd|bachelor|masters?|grad school)\b", re.I)),
]
# basics worth opening on; the rest are shown on the card but not judged as hooks
HOOKABLE_BASICS = ["Interests", "Pets", "Hometown", "Job", "Education", "Location"]


LEAD_SYMBOLS = re.compile(r"^[^\w'\"(]+")


def _clean(line: str) -> tuple[str, str | None]:
    """Strip bullets and leading emoji; an emoji that labels a field ('📍 Brooklyn') comes back as its key."""
    line = line.strip()
    lead = LEAD_SYMBOLS.match(line)
    key = None
    if lead:
        key = next((k for e, k in EMOJI_KEY.items() if e in lead.group(0)), None)
        line = line[lead.end():]
    return line.strip(), key


def _is_meta(line: str) -> bool:
    low = line.lower()
    return bool(re.match(r"^(here[’']?s|here is|below is|this is|the following|ok|okay)\b", low)) or \
        ("fictional" in low and "profile" in low) or "raw information" in low or \
        (len(low.split()) <= 12 and ":" not in low and bool(re.search(r"\b(this|her|his|their) (girl|guy|woman|man|person|profile|match)\b|\bon (hinge|bumble|tinder|raya|okcupid|feeld)\b", low)))


def _norm(line: str) -> str:
    return line.lower().replace("’", "'").rstrip(" .…:?").strip()


def _is_prompt_title(line: str) -> bool:
    low = _norm(line)
    if low in PROMPTS or low in {"looking for", "i'm looking for", "relationship goals", "anthem", "my anthem"}:
        return True
    words = low.split()
    return 2 <= len(words) <= 9 and not re.search(r"[.!?,]$", line) and bool(STEM_END.search(low))


def _header_kind(line: str) -> str | None:
    """'Photo captions/details visible on profile:' -> 'photos', 'My interests' -> 'interests'."""
    low = _norm(line)
    for kind, names in SECTIONS.items():
        if low in names:
            return kind
    if line.startswith("#") or "image description" in low:
        return "skip"
    if line.rstrip().endswith(":") and len(low.split()) <= 8:
        if re.search(r"\b(photos?|pics?|pictures?|captions?|images?)\b", low):
            return "photos"
        if re.search(r"\bprompts?\b", low):
            return "prompts"
        if re.search(r"\b(basics|vitals|details|lifestyle)\b", low):
            return "basics"
        if re.search(r"\binterests?\b|\bhobbies\b", low):
            return "interests"
    return None


def _basic_of(line: str) -> tuple[str, str] | None:
    low = line.lower().strip(" .")
    if low in ZODIAC:
        return "Zodiac", line
    for key, pat in BASIC_PATTERNS:
        if pat.search(line) and len(line.split()) <= 8:
            return key, line
    parts = [p.strip().lower() for p in re.split(r"[,/&]|\band\b", line) if p.strip()]
    if parts and all(p in LANGS for p in parts):
        return "Languages", line
    # 'Austin, Texas' / 'Brooklyn, NY'
    if re.match(r"^[A-Z][\w.' -]{1,30},\s*[A-Z][\w.' -]{1,30}$", line) and len(line.split()) <= 5:
        return "Location", line
    # 'Nurse at Mount Sinai', 'Works in marketing', 'Product Designer @ Figma', 'software engineer, google'
    if re.match(r"^(works? (in|at|as)|job:)\b", low) or (re.search(r"\s(at|@)\s+[A-Z]", line) and len(line.split()) <= 7) \
            or (JOB_WORDS.search(low) and len(line.split()) <= 6):
        return "Job", line
    # 'height 5'5', 'from chicago', 'school: UCLA' without a colon
    m = re.match(r"^(height|location|hometown|job|work|school|education|zodiac|star sign|religion|politics|pronouns|languages)\s+(.+)$", low)
    if m and len(line.split()) <= 6:
        key = KEY_LABEL.get(m.group(1), m.group(1).capitalize())
        return key, line[len(m.group(1)):].strip()
    return None


JOB_WORDS = re.compile(r"\b(engineer|developer|designer|nurse|teacher|doctor|physician|lawyer|attorney|consultant|analyst|manager|"
                       r"founder|student|accountant|marketing|sales|executive|scientist|researcher|artist|writer|chef|"
                       r"architect|therapist|dentist|pharmacist|paramedic|firefighter|officer|banker|trader|recruiter|"
                       r"realtor|photographer|producer|editor|professor|resident|surgeon|vet|veterinarian|pilot|"
                       r"bartender|barista|stylist|coach|trainer|entrepreneur|product|associate|director|specialist)\b")
SPLIT_ITEMS = re.compile(r"\s*[·•|]\s*")


def parse_profile(text: str) -> dict:
    """Any pasted dating profile -> {"name", "age", "basics": {label: value}, "prompts": [[title, answer]],
    "photos": [...], "other": [...]}. Order, labels and app don't matter."""
    out = {"name": None, "age": None, "basics": {}, "prompts": [], "photos": [], "other": []}
    raw = (text or "").splitlines()
    tabbed = any("\t" in r for r in raw)
    lines = []  # (text, emoji key)
    for r in raw:
        t, k = _clean(r)
        if t:
            lines.append((t, k))
    section, bio = None, []

    def add_basic(key, val):
        val = val.strip(" .")
        if not val:
            return
        if key == "Location" and "Location" in out["basics"]:
            key = "Hometown"
        if key in ("Hometown", "Location"):
            val = re.sub(r"^(originally from|from|hometown:?|grew up in|lives in|living in|based in|located in)\s*", "", val, flags=re.I)
        if key == "Interests" and "Interests" in out["basics"]:
            out["basics"]["Interests"] += ", " + val
        elif key not in out["basics"]:
            out["basics"][key] = val

    def flush_bio():
        if bio:
            out["prompts"].append(["About me", " ".join(bio)]); bio.clear()

    def name_age(t):
        """'Maddie, 29' / 'Jess 26' / 'Emma 27 London. Works in...' -> True if consumed (the rest is parsed too)."""
        m = re.match(r"^([A-Z][a-zA-Z'-]{1,20})\s*[,·|•-]?\s*(\d{2})\b\s*[,·|•-]?\s*(.*)$", t)
        if not m or out["name"] or not (18 <= int(m.group(2)) <= 99):
            return False
        out["name"], out["age"] = m.group(1), m.group(2)
        rest = m.group(3).strip()
        first, more = re.split(r"(?<=[.!?])\s+", rest, 1)[0], rest
        head = first.strip(" .,")
        if head and len(head.split()) <= 4 and head[:1].isupper() and (not _basic_of(head) or _basic_of(head)[0] == "Location"):
            add_basic("Location", head); more = rest[len(first):].strip()
        if more:
            sentence_pass(more)
        return True

    def item(t, hint=None) -> bool:
        """Classify one short item (a line or a '·'-separated piece). True if it was a basic."""
        if hint and hint != "Looking for":
            add_basic(hint, t); return True
        b = _basic_of(t)
        if b:
            add_basic(*b); return True
        return False

    def sentence_pass(t):
        """Free-form paragraph: pull structured bits out of each sentence, keep the rest as the bio."""
        rest = []
        for sent in re.split(r"(?<=[.!?])\s+", t):
            s2 = sent.strip()
            ph = re.match(r"^(photos?|pics?|pictures?)\s*[:\-–]\s*(.+)$", s2, re.I)
            if ph:
                out["photos"].extend(p.strip(" .") for p in re.split(r",\s*(?:and\s+)?|;\s*", ph.group(2)) if p.strip(" ."))
                continue
            core = s2.rstrip(".!")
            if name_age(core):
                continue
            if len(core.split()) <= 8 and item(core):
                continue
            rest.append(s2)
        if rest:
            out["other"].append(" ".join(rest))

    i = 0
    while i < len(lines):
        line, emoji = lines[i]; i += 1
        nxt = lines[i][0] if i < len(lines) else ""
        low = _norm(line)
        if _is_meta(line):
            if re.search(r"\b(photo|caption|picture)s?\b", low) and line.endswith(":"):
                section = "photos"
            continue
        kind = _header_kind(line)
        if kind:
            flush_bio(); section = None if kind in ("skip", "prompts") else kind
            continue
        # numbered / tab-separated photo tables: '1\tBarton Springs photo\tSame fictional woman...'
        row = re.match(r"^(\d+)[.)]?\s+(.+)$", line)
        if row and (tabbed or re.search(r"\bphoto\b", line, re.I)) and not re.match(r"^\d+\s*(miles?|km|mi)\b", row.group(2)):
            cols = [c.strip() for c in re.split(r"\t+|\s{3,}", row.group(2)) if c.strip()]
            out["photos"].append(_photo_line(cols[0], cols[1] if len(cols) > 1 else ""))
            continue
        if emoji == "Looking for":
            out["prompts"].append(["Looking for", line]); continue
        if emoji and len(line.split()) <= 10:
            add_basic(emoji, line); continue
        if name_age(line):
            continue
        if re.fullmatch(r"\d{2}", line) and not out["age"] and 18 <= int(line) <= 99:
            out["age"] = line; continue
        # 'Key: value' / 'Prompt title: answer' / 'We'll get along if... you have opinions'
        kv = re.match(r"^([^:]{1,70}):\s*(.*)$", line) or re.match(r"^(.{3,70}?)(?:\.\.\.|…)\s*(.+)$", line) or \
            re.match(r"^([A-Za-z' ]{2,20}?)\s+[-–—=]\s+(.+)$", line)
        if kv and (not re.match(r"^\d", line) or re.match(r"^\d+\s+(photos?|pics?|pictures?)\b", line, re.I)):
            k, v = kv.group(1).strip(), kv.group(2).strip()
            kl = _norm(k)
            if kl in BASIC_KEYS:
                if kl == "name" and v: out["name"] = v
                elif kl == "age" and v: out["age"] = v
                else: add_basic(KEY_LABEL.get(kl, k[:1].upper() + k[1:]), v)
                continue
            if kl in SECTIONS["interests"]:
                for p in re.split(r"\s*[·•|,]\s*", v): add_basic("Interests", p)
                continue
            if re.match(r"^(\d+\s+)?(photos?|pics?|pictures?)( captions?)?$", kl):
                parts = re.split(r"\s*\b\d+[.)]\s+", v) if re.search(r"\b\d[.)]\s", v) else re.split(r",\s*(?:and\s+)?|;\s*", v)
                out["photos"].extend(p.strip(" .,") for p in parts if p.strip(" .,"))
                continue
            if kl in ("wants", "seeking", "looking for", "intentions", "relationship goals"):
                out["prompts"].append(["Looking for", v]); continue
            if section == "photos":
                out["photos"].append(f"{k}: {v}" if v else k); continue
            if kl in SECTIONS["bio"]:
                k = "About me"
            flush_bio()
            if not v:
                if nxt and not _is_prompt_title(nxt) and not _header_kind(nxt):
                    out["prompts"].append([k, nxt]); i += 1
                continue
            out["prompts"].append([k, v]); continue
        # 'Dating intention' on one line, value on the next
        if low in BASIC_KEYS and nxt:
            if low == "name": out["name"] = nxt
            elif low == "age": out["age"] = nxt
            else: add_basic(KEY_LABEL.get(low, line[:1].upper() + line[1:]), nxt)
            i += 1; continue
        if _is_prompt_title(line) and nxt and not _header_kind(nxt) and not _is_prompt_title(nxt):
            flush_bio(); section = None
            out["prompts"].append([line.rstrip(" :…"), nxt]); i += 1; continue
        if section == "photos":
            out["photos"].append(line); continue
        if section == "interests":
            for p in re.split(r"\s*[·•|,]\s*", line): add_basic("Interests", p)
            continue
        pieces = SPLIT_ITEMS.split(line)
        if len(pieces) > 1 and section != "bio":  # 'Active · Sometimes drinks · Never smokes · Virgo'
            for p in pieces:
                if not item(p) and section == "basics":
                    out["basics"].setdefault(p, p)
            continue
        if section == "bio":
            bio.append(line); continue
        commas = [c.strip() for c in line.split(",") if c.strip()]
        if len(commas) >= 3 and sum(bool(_basic_of(c)) for c in commas) >= 2:
            for c in commas:
                if not item(c): out["basics"].setdefault(c, c)
            continue
        if re.search(r"\s+but\s+(originally\s+)?from\s+", line, re.I) and len(line.split()) <= 10:
            here, there = re.split(r"\s+but\s+(?:originally\s+)?from\s+", line, 1, flags=re.I)
            item(here); add_basic("Hometown", there); continue
        if item(line):
            continue
        words = line.split()
        if section == "basics" and len(words) <= 6:
            out["basics"].setdefault(line, line); continue
        short_title = len(words) <= 4 and not re.search(r"[.!?]$", line) and line[:1].isupper()
        if short_title and "Job" not in out["basics"] and not any(w[:1].islower() for w in words[1:] if w not in ("at", "of", "in", "&", "and")):
            add_basic("Job", line); continue  # 'Account Executive'
        if short_title and len(words) <= 3 and "Location" not in out["basics"] and "Job" in out["basics"]:
            add_basic("Location", line); continue  # 'San Francisco'
        if len(words) >= 3:
            sentence_pass(line)
        else:  # never drop anything: unrecognised short bits still show on the card
            out["basics"].setdefault(line, line)
    flush_bio()
    return out


PHOTO_BOILERPLATE = re.compile(r"^.*?\bfictional\b.*?\b(woman|man|person|guy|girl)\b(\s+(living|based)\s+in\s+[^,]+)?[,\s]*", re.I)


def _photo_line(desc: str, detail: str) -> str:
    """'Barton Springs photo' + 'Same fictional woman swimming at Barton Springs in Austin, standing...' ->
    'Barton Springs photo - swimming at Barton Springs in Austin, standing waist deep'."""
    detail = PHOTO_BOILERPLATE.sub("", detail or "").strip(" ,")
    clauses = [c.strip() for c in detail.split(",") if c.strip()][:2]
    words = " ".join(", ".join(clauses).split()[:18])
    return desc + (f" - {words}" if words else "")


def looks_weak(p: dict, text: str) -> bool:
    """True when the rules clearly didn't understand the paste, so a model should structure it instead."""
    unknown = [k for k, v in p["basics"].items() if k == v]
    words = len((text or "").split())
    return bool(p["other"]) or len(unknown) >= 2 or (words > 40 and not p["prompts"] and not p["photos"]) or \
        (words > 15 and not p["name"] and not p["prompts"])


def from_model(d: dict) -> dict:
    """Claude's structured profile -> the same shape parse_profile returns."""
    return {"name": d.get("name") or None, "age": str(d["age"]) if d.get("age") else None,
            "basics": {b["label"]: b["value"] for b in d.get("basics", []) if b.get("label") and b.get("value")},
            "prompts": [[x["title"] or "About me", x["answer"]] for x in d.get("prompts", []) if x.get("answer")],
            "photos": [ph for ph in d.get("photos", []) if ph], "other": [], "by": "claude"}


def hook_pieces(text: str, parsed: dict | None = None) -> list[tuple[str, str]]:
    """Parsed profile -> (source, text) hook candidates: prompts and photos first, then the few basics worth
    opening on."""
    p = parsed or parse_profile(text)
    prompts = [("prompt", f"{q}: {a}") for q, a in p["prompts"]]
    other = [("prompt", s) for line in p["other"] for s in re.split(r"(?<=[.!?])\s+", line) if len(s.split()) >= 3]
    photos = [("photo", f"Photo: {ph}") for ph in p["photos"]]
    basics = [("basic", f"{k}: {p['basics'][k]}") for k in HOOKABLE_BASICS if k in p["basics"]]
    return (prompts + other)[:12] + photos[:8] + basics[:3]


def profile_to_text(p: dict) -> str:
    """Structured profile -> clean text for the profile box. parse_profile() reads it back the same way."""
    lines = []
    head = ", ".join(x for x in (p.get("name"), p.get("age")) if x)
    if head:
        lines.append(head)
    for k, v in p.get("basics", {}).items():
        lines.append(v if k == v else f"{k}: {v}")
    for title, answer in p.get("prompts", []):
        lines += ["", f"{title or 'About me'}:", answer]
    if p.get("photos"):
        lines.append("")
        lines += [f"Photo: {ph}" for ph in p["photos"]]
    return "\n".join(lines).strip()
