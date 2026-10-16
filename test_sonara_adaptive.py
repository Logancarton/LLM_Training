"""Focused tests for Sonara's opt-in neural decision calibration."""
import unittest
from types import SimpleNamespace
from sonara_adaptive import AdaptiveLoop


def prediction(confidence=0.4, uncertainty=1.0, answer="shed"):
    return SimpleNamespace(answer=answer, confidence=confidence, uncertainty=uncertainty)


class AdaptiveLoopTests(unittest.TestCase):
    def test_default_matches_neural_gate_and_safe_abstention(self):
        loop = AdaptiveLoop()
        self.assertEqual(loop.decide("Where is the ball?", prediction()).action, "NEURAL_PREDICTION")
        self.assertEqual(loop.decide("Where is the ball?", None).action, "ABSTAIN")
        self.assertEqual(loop.decide("Where is the ball?", prediction(answer="unknown")).action, "ABSTAIN")
        self.assertEqual(loop.decide("Where is the ball?", prediction(confidence=float("nan"))).action, "ABSTAIN")

    def test_negative_rating_changes_related_decisions_not_unrelated(self):
        loop = AdaptiveLoop()
        loop.feedback("Where is the ball?", 0.0)
        self.assertEqual(loop.decide("Where is the ball?", prediction()).action, "ABSTAIN")
        self.assertEqual(loop.decide("Where is the ball?", prediction()).supporting_ratings, 1)
        self.assertEqual(loop.decide("Why does the moon glow?", prediction()).action, "NEURAL_PREDICTION")

    def test_restart_accepts_only_explicit_neural_feedback(self):
        history = [
            {"type": "adaptive_feedback", "source": "model", "target_type": "neural_prediction", "question": "Where is the ball?", "rating": 0},
            {"type": "adaptive_feedback", "source": "user", "target_type": "abstention", "question": "Where is the ball?", "rating": 0},
            {"type": "adaptive_feedback", "source": "user", "target_type": "neural_prediction", "question": "Where is the ball?", "rating": 0},
        ]
        self.assertEqual(AdaptiveLoop(history).decide("Where is the ball?", prediction()).action, "ABSTAIN")
        self.assertEqual(AdaptiveLoop(history).decide("Where is the ball?", prediction()).supporting_ratings, 1)
        self.assertEqual(AdaptiveLoop(history[:2]).decide("Where is the ball?", prediction()).action, "NEURAL_PREDICTION")

    def test_retrieval_is_evidence_not_inferred_answer(self):
        loop = AdaptiveLoop()
        self.assertEqual(loop.decide("Where is the ball?", None, evidence_count=2).action, "RECALL_EVIDENCE")
        self.assertEqual(loop.decide("Where is the ball?", prediction(confidence=0.1), evidence_count=1).action, "RECALL_EVIDENCE")
        self.assertEqual(loop.decide("Where is the ball?", prediction(), evidence_count=1).action, "NEURAL_PREDICTION")

    def test_rating_validation(self):
        loop = AdaptiveLoop()
        for score in (-1, 2, float("nan"), True):
            with self.assertRaises(ValueError):
                loop.feedback("where is it", score)


if __name__ == "__main__":
    unittest.main()
