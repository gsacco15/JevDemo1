"""Jev's three typed question styles: Noul (yes/no), Choice (pick one), Score (ordered levels)."""

from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_LEVELS = ["none", "low", "medium", "high", "very_high"]


@dataclass(frozen=True)
class Question:
    id: str
    kind: Literal["noul", "score", "choice"]
    prompt: str
    options: tuple[str, ...] = ()  # choice options, or score levels (ordered low -> high)

    @property
    def levels(self) -> tuple[str, ...]:
        return self.options or tuple(DEFAULT_LEVELS)


def noul(id: str, prompt: str) -> Question:
    return Question(id, "noul", prompt)


def score(id: str, prompt: str, levels: tuple[str, ...] = tuple(DEFAULT_LEVELS)) -> Question:
    return Question(id, "score", prompt, levels)


def choice(id: str, prompt: str, options: list[str] | tuple[str, ...]) -> Question:
    return Question(id, "choice", prompt, tuple(options))


@dataclass
class EvalItem:
    """One thing to judge.

    `text` is what a real Jev deployment sees (rendered app state).
    `data` is the same state in structured form, used by the local heuristic judge.
    """

    key: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
