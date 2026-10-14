"""Behavioral checks for the two-model Sonara vertical slice; no network or secrets."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sonara_memory import SonaraMemorySystem
from sonara_world import WorldCognition, parse_state
from sonara_research import ResearchCognition, ResearchModel


class TwoModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.memories = SonaraMemorySystem(self.root / "sonara_memories")
        self.world = WorldCognition(self.memories, self.root)

    def test_observation_state_correction_restart_and_isolation(self):
        first = self.world.observe("Jake moved the ball into the pond.", "s1")
        self.assertEqual(first.semantic_frame["entity"], "ball")
        self.assertEqual(first.semantic_frame["location"], "pond")
        self.world.observe("Logan looked at his phone.", "s1")
        second = self.world.observe("The ball is in the shed.", "s1")
        answer = self.world.answer_where("Where is the ball?")
        self.assertIn("shed", answer)
        self.assertIn(second.id, answer)
        self.assertNotIn("pond (", answer)
        self.assertIn("Unknown", self.world.answer_where("Where is the cat?"))
        self.assertIsNone(self.world.answer_where("Who is Jake?"))
        reloaded = SonaraMemorySystem(self.root / "sonara_memories")
        restarted = WorldCognition(reloaded, self.root)
        self.assertIn("shed", restarted.answer_where("Where is the ball?"))
        self.assertEqual(len(reloaded.memories), 3)
        self.assertIn(second.id, reloaded.entity_index["ball"])
        self.assertIn("time-qualified", restarted.answer_where("Where was the ball at 12:15?"))

    def test_time_order_across_timezone_offsets(self):
        # 19:15 UTC is later than 12:00 -07:00, regardless of string order.
        self.world.observe("2026-10-13T12:00:00-07:00 The ball is in the yard.", "s")
        self.world.observe("2026-10-13T19:15:00+00:00 The ball is in the pond.", "s")
        self.assertIn("pond", self.world.answer_where("Where is the ball?"))

    def test_event_time_separate_from_recorded_time(self):
        frame = parse_state("2026-10-13T12:15:00-07:00 Jake moved the ball to yard.",
                            "2026-10-13T23:00:00+00:00")
        self.assertEqual(frame["observed_at"], "2026-10-13T12:15:00-07:00")
        self.assertIsNone(parse_state("A ball might be in some place.", "2026-10-13T23:00:00+00:00"))

    def test_neural_learning_is_explicit_bounded_and_guarded(self):
        torch.manual_seed(8)
        for i in range(5):
            self.world.observe(f"Jake moved the ball to yard {i}.", "s")
            self.world.observe(f"Logan read a book on page {i}.", "s")
        self.assertIsNone(self.world.change_probability("The ball moved to yard."))
        result = self.world.learn_bounded(max_examples=8, steps=25)
        self.assertEqual(result["examples"], 6)
        self.assertEqual(result["holdout"], 2)
        self.assertIn(result["accepted"], (True, False))
        if result["accepted"]:
            self.assertTrue((self.root / "world_model.pt").exists())
            self.assertIsNotNone(WorldCognition(self.memories, self.root).change_probability(
                "The ball moved to yard."))
        # Other events remain intact regardless of gradient acceptance.
        self.assertEqual(len(self.memories.memories), 10)

    def test_research_feedback_updates_only_research_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-token",
                                     "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}):
            calls = []
            def fake_api(provider, question):
                calls.append((provider, question))
                return "Hypothesis, not verified."
            research = ResearchCognition(self.root, transport=fake_api)
            result = research.research("Why does an object move?")
            self.assertEqual(result["provider"], "openai")
            self.assertEqual(result["status"], "unverified_external_evidence")
            self.assertEqual(len(calls), 1)
            trained = research.feedback(result["question"], result["provider"], 1.0)
            self.assertTrue(trained["trained"])
            self.assertLess(trained["utility_error_after"], trained["utility_error_before"])
            self.assertTrue((self.root / "research_model.pt").exists())
            self.assertIsInstance(ResearchCognition(self.root, transport=fake_api).model, ResearchModel)
            self.assertEqual(len(self.memories.memories), 0)
            self.assertFalse((self.root / "world_model.pt").exists())

    def test_research_cannot_send_without_explicit_configured_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
                                     "GEMINI_API_KEY": ""}):
            research = ResearchCognition(self.root, transport=lambda *_: self.fail("called API"))
            with self.assertRaisesRegex(ValueError, "No AI provider"):
                research.research("Explain something")
            with self.assertRaises(ValueError):
                research.feedback("x", "openai", 2.0)


if __name__ == "__main__":
    unittest.main()
