"""Turn pasted conversation text into structured messages.

Accepted formats (one message per line, continuation lines are appended):
    Me: hey
    Her: Lol you definitely couldn't keep up
    Them / Him / Match / <any name>: ...
"""

import re

from .schemas import Message

USER_LABELS = {"me", "i", "you", "user", "self", "mine"}
LINE_RE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,24})\s*[:：]\s*(.*)$")


def parse_conversation(text: str) -> list[Message]:
    messages: list[Message] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            label, body = m.group(1).strip().lower(), m.group(2).strip()
            speaker = "user" if label in USER_LABELS else "match"
            if body:
                messages.append(Message(speaker=speaker, text=body))
            else:
                messages.append(Message(speaker=speaker, text=""))
        elif messages:
            sep = " " if messages[-1].text else ""
            messages[-1].text += sep + line
        else:
            # Unlabelled single message: assume it came from the match.
            messages.append(Message(speaker="match", text=line))
    return [m for m in messages if m.text]


def render_conversation(messages: list[Message], pronoun_label: str = "MATCH") -> str:
    return "\n".join(f"{'ME' if m.speaker == 'user' else pronoun_label}: {m.text}" for m in messages)
