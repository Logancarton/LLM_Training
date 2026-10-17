"""State brief: the only path from the layers to the LLM, about 200 tokens.

Filled in priority order; lines are dropped from the bottom at the cap.
Corrections are never dropped.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config
from action import INSTRUCTIONS


def _caution_line(ht: float, stakes: float) -> Optional[str]:
    if ht >= config.SAFETY_HT:
        why = "stakes" if stakes >= 0.5 else "uncertainty"
        return f"caution: high ({why}) - rely on verified context only"
    if ht >= 0.5:
        return "caution: moderate - flag anything you are unsure of"
    return None


def _affect_line(affect: Optional[dict]) -> Optional[str]:
    if not affect:
        return None
    dv = affect["valence"] - affect["valence_base"]
    da = affect["arousal"] - affect["arousal_base"]
    if abs(dv) < 0.15 and abs(da) < 0.15:
        return "user_now: near their usual"
    mood = "above" if dv > 0 else "below"
    energy = "up" if da > 0.15 else "down" if da < -0.15 else "steady"
    return f"user_now: valence {mood} their usual ({dv:+.2f}), arousal {energy}"


def _location_line(locations: List[dict]) -> str:
    if not locations or locations[0]["kind"] == "llm_general":
        return "location: likely general knowledge (no stored answer)"
    top = locations[0]
    return f"location: past {top['kind']} - {top['summary'][:140]}"


def build(action: str, mods: Dict[str, float], stakes: float, affect: Optional[dict],
          corrections: List[dict], hot: List[dict], locations: List[dict],
          evidence: Optional[List[dict]] = None) -> str:
    must = ["[STATE]", f"action: {action} - {INSTRUCTIONS.get(action, action.lower())}"]
    caution = _caution_line(mods["ht"], stakes)
    if caution:
        must.append(caution)
    for c in corrections[:config.BRIEF_MAX_CORRECTIONS]:
        wrong = f' (not "{c["wrong_text"][:80]}")' if c.get("wrong_text") else ""
        must.append(f'correction: "{c["right_text"][:160]}"{wrong}, '
                    f'confirmed {c["confirmations"]}x')
    optional = []
    for e in (evidence or []):   # the user's own earlier words, quoted with their age
        optional.append(f'evidence: user said "{e["content"][:140]}" ({e["when"]})')
    affect_line = _affect_line(affect)
    if affect_line:
        optional.append(affect_line)
    if hot:
        optional.append("hot: " + "; ".join(h["text"][:60] for h in hot[:config.BRIEF_MAX_HOT]))
    optional.append(_location_line(locations))

    lines = list(must)
    for line in optional:
        if len("\n".join(lines + [line])) > config.BRIEF_MAX_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)
