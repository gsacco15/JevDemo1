from ..config import settings
from .client import HeuristicJudge, TypeSafeJevJudge
from .types import EvalItem, Question


def make_judge():
    if settings.use_real_jev:
        return TypeSafeJevJudge(settings.jev_api_key, settings.jev_base_url, settings.jev_model, settings.jev_concurrency)
    return HeuristicJudge()


__all__ = ["make_judge", "EvalItem", "Question", "HeuristicJudge", "TypeSafeJevJudge"]
