"""Action selection (basal ganglia): actions compete; the highest score wins.

HEDGE is the floor: it wins whenever nothing else clears its bar.
"""
from __future__ import annotations

from typing import Dict, Tuple

import config

ANSWER, CLARIFY, RETRIEVE_MORE, HEDGE = "ANSWER", "CLARIFY", "RETRIEVE_MORE", "HEDGE"

INSTRUCTIONS = {
    ANSWER: "answer directly",
    CLARIFY: "ask one short clarifying question before answering",
    HEDGE: "answer, and state plainly what is uncertain",
}


def answer_bar(ht: float) -> float:
    return config.ANSWER_BAR_BASE + config.ANSWER_BAR_HT * ht


def select(confidence: float, ambiguity: float, mods: Dict[str, float],
           retried: bool) -> Tuple[str, Dict[str, float]]:
    bar = answer_bar(mods["ht"])
    scores = {
        ANSWER: confidence - bar,
        CLARIFY: ambiguity - config.CLARIFY_OFFSET,
        HEDGE: 0.0,
    }
    if not retried and mods["ht"] < config.SAFETY_HT:
        scores[RETRIEVE_MORE] = (bar - confidence) * mods["ne"]
    best = max(scores, key=lambda a: scores[a])
    return best, {k: round(v, 3) for k, v in scores.items()}
