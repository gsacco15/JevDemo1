"""Learn what this user actually likes from lightweight feedback.

Preferences are small, inspectable numbers attached to the user's account:
  offsets      - shifts to the desired tone (flirt, escalation, directness, ...)
  weights      - multipliers on ranking components (style_fit, brevity_fit, ...)
  length_mult  - preferred message length relative to default
Nothing here is a hidden model; it is easy to show the user and easy to reset.
"""

from .ranking import TONE_DIMS

LR = {"copy": 0.2, "select": 0.15, "like": 0.15, "edited": 0.2, "dislike": -0.08, "not_me": -0.08}
FIXED = {
    "too_much": {"escalation": -0.06, "flirt": -0.05, "directness": -0.03},
    "too_boring": {"escalation": 0.05, "playfulness": 0.05, "flirt": 0.03},
    "too_cheesy": {"playfulness": -0.03, "flirt": -0.03},
    "more_direct": {"directness": 0.07, "escalation": 0.03},
    "less_direct": {"directness": -0.07, "escalation": -0.03},
}
WEIGHT_NUDGES = {
    "not_me": {"style_fit": 1.12},
    "too_long": {"brevity_fit": 1.1},
    "too_cheesy": {"naturalness": 1.08, "originality": 1.05},
    "too_boring": {"originality": 1.06},
}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def empty() -> dict:
    return {"offsets": {k: 0.0 for k in TONE_DIMS}, "weights": {}, "length_mult": 1.0, "events": 0}


def update(prefs: dict, kind: str, feats: dict, desired: dict, components: dict,
           edited_text: str | None = None, original_text: str | None = None) -> dict:
    p = empty() | (prefs or {})
    off, w = dict(p["offsets"]), dict(p["weights"])

    if kind in LR:
        lr = LR[kind]
        for k in TONE_DIMS:
            if k in feats and k in desired:
                off[k] = off.get(k, 0.0) + lr * (feats[k] - desired[k])
        if lr > 0 and feats.get("words") and desired.get("words"):
            p["length_mult"] *= (feats["words"] / desired["words"]) ** (0.15 if kind != "edited" else 0.05)
        if lr > 0:
            for comp, val in components.items():
                w[comp] = w.get(comp, 1.0) * (1 + 0.05 * (val - 0.6))
    for k, d in FIXED.get(kind, {}).items():
        off[k] = off.get(k, 0.0) + d
    for k, m in WEIGHT_NUDGES.get(kind, {}).items():
        w[k] = w.get(k, 1.0) * m
    if kind == "too_long":
        p["length_mult"] *= 0.85
    if kind == "edited" and edited_text and original_text:
        ratio = max(1, len(edited_text.split())) / max(1, len(original_text.split()))
        p["length_mult"] *= ratio ** 0.3

    p["offsets"] = {k: round(_clamp(x, -0.35, 0.35), 4) for k, x in off.items()}
    p["weights"] = {k: round(_clamp(x, 0.5, 2.0), 4) for k, x in w.items()}
    p["length_mult"] = round(_clamp(p["length_mult"], 0.4, 2.5), 4)
    p["events"] = p.get("events", 0) + 1
    return p


def describe(prefs: dict) -> list[str]:
    """Plain-language summary for the UI."""
    if not prefs or not prefs.get("events"):
        return ["No feedback yet - using your style settings."]
    out = []
    names = {"flirt": "flirty", "escalation": "forward", "directness": "direct", "playfulness": "playful",
             "confidence": "confident", "warmth": "warm"}
    for k, x in sorted(prefs.get("offsets", {}).items(), key=lambda kv: -abs(kv[1])):
        if abs(x) >= 0.03:
            out.append(f"{'More' if x > 0 else 'Less'} {names.get(k, k)} ({x:+.2f})")
    lm = prefs.get("length_mult", 1.0)
    if abs(lm - 1) >= 0.05:
        out.append(f"{'Longer' if lm > 1 else 'Shorter'} messages (x{lm:.2f})")
    for k, m in sorted(prefs.get("weights", {}).items(), key=lambda kv: -abs(kv[1] - 1)):
        if abs(m - 1) >= 0.05:
            out.append(f"{'Values' if m > 1 else 'Cares less about'} {k.replace('_', ' ')} (x{m:.2f})")
    return out[:8] or ["Learning... not enough signal yet."]
