"""Jev's three typed question styles, matching the TypeSafe System One API.

    noul   - instructions only                      -> answer: {"noul": P(yes)}
    choice - instructions + {option: description}   -> answer: {"choice", "confidence", "probabilities"}
    score  - instructions + [level descriptions]    -> answer: {"score" (level index), "confidence", "probabilities"}
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Question:
    id: str
    kind: Literal["noul", "score", "choice"]
    instructions: str
    # choice: ((option, description), ...); score: (level description, ...) ordered low -> high
    criteria: tuple = ()

    @property
    def options(self) -> tuple[str, ...]:
        """Choice option keys, or score level descriptions."""
        if self.kind == "choice":
            return tuple(k for k, _ in self.criteria)
        return tuple(self.criteria)

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(self.criteria)

    def to_api(self) -> dict:
        spec: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.kind == "choice":
            spec["criteria"] = dict(self.criteria)
        elif self.kind == "score":
            spec["criteria"] = list(self.criteria)
        return spec


def noul(id: str, instructions: str) -> Question:
    return Question(id, "noul", instructions)


def score(id: str, instructions: str, levels: list[str]) -> Question:
    assert 2 <= len(levels) <= 10, "Score needs 2-10 levels"
    return Question(id, "score", instructions, tuple(levels))


def choice(id: str, instructions: str, options: dict[str, str]) -> Question:
    return Question(id, "choice", instructions, tuple(options.items()))


@dataclass
class EvalItem:
    """One thing to judge.

    `state` is what Jev sees: named JSON fields the questions refer to by backticked path.
    `data` is the same situation in the shape the local heuristic judge expects.
    """

    key: str
    state: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
