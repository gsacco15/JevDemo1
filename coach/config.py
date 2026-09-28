"""Runtime configuration, read from environment variables.

Everything has a working default so the app runs with zero keys:
  - no TYPESAFE_API_KEY   -> local heuristic judge stands in for Jev (JEV_API_KEY also accepted)
  - no ANTHROPIC_API_KEY  -> template generator stands in for the LLM
"""

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # Jev / TypeSafe
    jev_api_key: str | None = field(default_factory=lambda: os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY"))
    jev_base_url: str = field(default_factory=lambda: os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai"))
    jev_model: str = field(default_factory=lambda: os.environ.get("JEV_MODEL", "jev-latest"))
    jev_concurrency: int = field(default_factory=lambda: int(os.environ.get("JEV_CONCURRENCY", "16")))

    # Generative + reasoning model (Claude)
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    generator_model: str = field(default_factory=lambda: os.environ.get("GENERATOR_MODEL", "claude-opus-5"))
    reasoning_model: str = field(default_factory=lambda: os.environ.get("REASONING_MODEL", "claude-opus-5"))
    # Force the local stand-ins even when keys exist (useful for cheap testing).
    force_mock: bool = field(default_factory=lambda: os.environ.get("COACH_FORCE_MOCK", "") == "1")

    # Pipeline sizes
    num_candidates: int = field(default_factory=lambda: int(os.environ.get("NUM_CANDIDATES", "30")))
    tournament_size: int = 12
    # how much Jev's head-to-head win rate moves the final score (+/- half this at 100%/0% win rate)
    tournament_weight: float = field(default_factory=lambda: _env_float("TOURNAMENT_WEIGHT", 30))

    # Confidence policy (System One -> System Two)
    conf_auto: float = field(default_factory=lambda: _env_float("CONF_AUTO", 0.90))
    conf_escalate: float = field(default_factory=lambda: _env_float("CONF_ESCALATE", 0.5))
    escalation_top: int = field(default_factory=lambda: int(os.environ.get("ESCALATION_TOP", "6")))
    escalation_cap: int = field(default_factory=lambda: int(os.environ.get("ESCALATION_CAP", "24")))

    # Access control for public deploys: set APP_PASSWORD to require it; searches are rate limited per IP
    app_password: str | None = field(default_factory=lambda: os.environ.get("APP_PASSWORD") or None)
    rate_limit_per_hour: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_HOUR", "60")))

    # Privacy
    # Serverless hosts (Vercel) only allow writes under /tmp
    db_path: str = field(default_factory=lambda: os.environ.get(
        "COACH_DB", "/tmp/coach.db" if os.environ.get("VERCEL") else "data/coach.db"))
    retention_hours: float = field(default_factory=lambda: _env_float("RETENTION_HOURS", 24))

    @property
    def use_real_jev(self) -> bool:
        return bool(self.jev_api_key) and not self.force_mock

    @property
    def use_claude(self) -> bool:
        return bool(self.anthropic_api_key) and not self.force_mock


settings = Settings()
