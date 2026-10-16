"""Feedback-calibrated action gating for the existing Sonara session.

Adapted from the uploaded sonara_loop pre/post-loop and modulators.
SonaraWorldModel owns prediction; SonaraMemorySystem owns episodic memory.
This gate learns a *decision threshold*, never facts. Only explicitly rated,
user-sourced neural predictions can teach it. All durable feedback uses the
existing append-only events.jsonl, not a second memory graph or runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Optional

BASE_CONFIDENCE = 0.25
MAX_UNCERTAINTY = 2.2
SIMILARITY_FLOOR = 0.30
MAX_CAUTION = 0.50
MAX_CONFIDENCE_BAR = 0.70
FEEDBACK_DECAY = 0.80
MAX_EVIDENCE = 20

STOP = frozenset("a an and are as at be before by do does for from how i in is it of on or the to was were what when where who why with you".split())


def cue_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9_]+", text.lower())) - STOP


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Decision:
    action: str  # NEURAL_PREDICTION, RECALL_EVIDENCE, or ABSTAIN
    confidence_bar: float
    uncertainty_ceiling: float
    caution: float
    supporting_ratings: int
    reason: str


class AdaptiveLoop:
    """Conservative cue-scoped decision calibration from explicit user feedback."""

    def __init__(self, past_events: Iterable[dict] = ()):
        self._ratings: list[dict] = []
        for event in past_events:
            if event.get("type") != "adaptive_feedback" or event.get("source") != "user":
                continue
            question = event.get("question")
            score = event.get("rating")
            if (not isinstance(question, str) or not question.strip()
                    or isinstance(score, bool) or not isinstance(score, (float, int))
                    or not math.isfinite(score) or not 0.0 <= score <= 1.0
                    or event.get("target_type") != "neural_prediction"):
                continue
            self._ratings.append({"tokens": cue_tokens(question), "rating": float(score)})
        self._ratings = self._ratings[-MAX_EVIDENCE:]

    def caution_for(self, question: str) -> tuple[float, int]:
        """Negative ratings raise caution only for similar question cues."""
        current = cue_tokens(question)
        negative, mass, count = 0.0, 0.0, 0
        for i, record in enumerate(reversed(self._ratings)):
            similarity = overlap(current, record["tokens"])
            if similarity < SIMILARITY_FLOOR:
                continue
            weight = similarity * (FEEDBACK_DECAY ** i)
            negative += (1.0 - record["rating"]) * weight
            mass += weight
            count += 1
        if mass == 0:
            return 0.0, 0
        return min(MAX_CAUTION, MAX_CAUTION * negative / mass), count

    def decide(self, question: str, prediction: Optional[object], evidence_count: int = 0) -> Decision:
        caution, support = self.caution_for(question)
        threshold = min(MAX_CONFIDENCE_BAR, BASE_CONFIDENCE + caution)
        ceiling = MAX_UNCERTAINTY - caution
        fallback = "RECALL_EVIDENCE" if evidence_count > 0 else "ABSTAIN"
        if prediction is None:
            return Decision(fallback, threshold, ceiling, caution, support, "world model disabled")
        try:
            answer = prediction.answer
            confidence = float(prediction.confidence)
            uncertainty = float(prediction.uncertainty)
        except (AttributeError, TypeError, ValueError):
            return Decision(fallback, threshold, ceiling, caution, support, "invalid prediction")
        if not isinstance(answer, str) or not answer.strip() or answer.lower() == "unknown":
            return Decision(fallback, threshold, ceiling, caution, support, "world model abstained")
        if not (math.isfinite(confidence) and math.isfinite(uncertainty)):
            return Decision(fallback, threshold, ceiling, caution, support, "non-finite prediction score")
        if confidence < threshold or uncertainty >= ceiling:
            return Decision(fallback, threshold, ceiling, caution, support,
                            "insufficient confidence or excessive uncertainty")
        return Decision("NEURAL_PREDICTION", threshold, ceiling, caution, support,
                        "neural prediction passed feedback-calibrated gate")

    def feedback(self, question: str, score: float) -> dict:
        """Update the in-memory gate; caller persists the explicit user event."""
        if (not question.strip() or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError("Rate a neural answer from 0 to 1")
        self._ratings.append({"tokens": cue_tokens(question), "rating": float(score)})
        self._ratings = self._ratings[-MAX_EVIDENCE:]
        caution, count = self.caution_for(question)
        return {"caution": round(caution, 3), "matching_ratings": count}
