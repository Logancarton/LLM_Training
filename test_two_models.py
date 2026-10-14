"""Behavioral tests for Sonara's world owner and fully local research assistant."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import torch

from LLM_Training import read_records, run_session
from sonara_memory import SonaraMemorySystem
from sonara_world import WorldCognition, parse_state
from sonara_research import (
    DEFAULT_LOCAL_MODEL, LOCAL_ENDPOINT, ResearchCognition, ResearchModel,
    ollama_local_generate,
)


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
        self.assertIn("Unknown", restarted.answer_where("Where was the ball at 12:15?"))

    def test_time_order_across_timezone_offsets(self):
        self.world.observe("2026-10-13T12:00:00-07:00 The ball is in the yard.", "s")
        self.world.observe("2026-10-13T19:15:00+00:00 The ball is in the pond.", "s")
        self.assertIn("pond", self.world.answer_where("Where is the ball?"))

    def test_historical_before_query_uses_correct_prior_event(self):
        pond = self.world.observe("Jake moved the ball into the pond.", "session")
        shed = self.world.observe("Jake moved the ball into the shed.", "session")
        yard = self.world.observe("Jake moved the ball into the yard.", "session")
        self.world.observe("The cat is in the attic.", "session")
        before_shed = self.world.answer_where("Where was the ball before it was at the shed?")
        self.assertIn("Previously reported location: ball was at pond", before_shed)
        self.assertIn(pond.id, before_shed)
        self.assertIn(shed.id, before_shed)
        self.assertNotIn(yard.id, before_shed)
        self.assertIn("shed", self.world.answer_where("Where was the ball before it was in the yard?"))
        self.assertIn("shed", self.world.answer_where("Where was the ball previously?"))
        self.assertIn("yard", self.world.answer_where("Where is the ball?"))
        self.assertEqual(len(self.memories.memories), 4)

        restarted = WorldCognition(SonaraMemorySystem(self.root / "sonara_memories"), self.root)
        self.assertIn("pond", restarted.answer_where("Where was the ball before it was at the shed?"))
        self.assertIn("yard", restarted.answer_where("Where is the ball?"))

    def test_historical_unknown_and_ambiguous_references_abstain(self):
        self.world.observe("Jake moved the ball into the pond.", "session")
        self.assertIn("Unknown", self.world.answer_where("Where was the ball before it was at the pond?"))
        self.assertIn("Unknown", self.world.answer_where("Where was the ball before it was at the shed?"))
        self.assertIn("Unknown", self.world.answer_where("Where was the ball after it was at the pond?"))
        self.assertIn("Unknown", self.world.answer_where("Where was the ball before the cat was at the pond?"))
        self.assertIn("Unknown", self.world.answer_where("Where was the ball before it disappeared?"))
        self.world.observe("Jake moved the ball into the shed.", "session")
        self.world.observe("Jake moved the ball into the yard.", "session")
        self.world.observe("Jake moved the ball into the shed.", "session")
        self.assertIn("multiple recorded", self.world.answer_where(
            "Where was the ball before it was at the shed?"))
        self.assertIn("yard", self.world.answer_where("Where was the ball before?"))
        self.assertIn("shed", self.world.answer_where("Where is the ball?"))

    def test_live_session_historical_query_survives_restart(self):
        with patch("builtins.input", side_effect=[
            "Jake moved the ball into the pond.",
            "Jake moved the ball into the shed.",
            "/ask Where was the ball before it was at the shed?",
            "/quit",
        ]):
            run_session(self.root, None, None, None, None, False)
        with patch("builtins.input", side_effect=[
            "/ask Where was the ball before it was at the shed?", "/quit",
        ]):
            run_session(self.root, None, None, None, None, False)
        rows = [row for row in read_records(self.root / "events.jsonl")
                if row.get("type") == "prediction"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn("Previously reported location: ball was at pond", row["text"])
            self.assertIn("before the recorded shed observation", row["text"])
        memory = SonaraMemorySystem(self.root / "sonara_memories")
        self.assertEqual(len(memory.memories), 2)
        self.assertFalse((self.root / "world_model.pt").exists())
        self.assertFalse((self.root / "research_utility_model.pt").exists())

    def test_event_time_separate_from_recorded_time(self):
        frame = parse_state("2026-10-13T12:15:00-07:00 Jake moved the ball to yard.",
                            "2026-10-13T23:00:00+00:00")
        self.assertEqual(frame["observed_at"], "2026-10-13T12:15:00-07:00")
        self.assertIsNone(parse_state("A ball might be in some place.",
                                       "2026-10-13T23:00:00+00:00"))
        moved = parse_state("Jake moved the ball from the yard to the pond.",
                            "2026-10-13T23:00:00+00:00")
        self.assertEqual(moved["location"], "pond")
        self.assertIsNone(parse_state("Jake took the ball from the yard.",
                                       "2026-10-13T23:00:00+00:00"))

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
        self.assertEqual(len(self.memories.memories), 10)

    def test_research_is_local_and_research_feedback_isolated(self):
        calls = []

        def fake_local(question, model):
            calls.append((question, model))
            return "One possible explanation; not verified."

        research = ResearchCognition(self.root, transport=fake_local)
        result = research.research("Why does an object move?")
        self.assertEqual(result["model"], DEFAULT_LOCAL_MODEL)
        self.assertEqual(result["status"], "unverified_local_model_output")
        self.assertAlmostEqual(result["estimated_utility"], research.expected_utility(result["question"]), places=3)
        self.assertEqual(calls, [("Why does an object move?", DEFAULT_LOCAL_MODEL)])
        trained = research.feedback(result["question"], 1.0)
        self.assertTrue(trained["trained"])
        self.assertLess(trained["utility_error_after"], trained["utility_error_before"])
        self.assertTrue((self.root / "research_utility_model.pt").exists())
        reloaded = ResearchCognition(self.root, transport=fake_local)
        self.assertIsInstance(reloaded.model, ResearchModel)
        self.assertAlmostEqual(reloaded.expected_utility(result["question"]),
                               research.expected_utility(result["question"]), places=6)
        self.assertEqual(len(self.memories.memories), 0)
        self.assertFalse((self.root / "world_model.pt").exists())

    def test_ollama_transport_cannot_contact_external_providers(self):
        with patch("sonara_research.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(b'{"response":"Local hypothesis"}')
            answer = ollama_local_generate("Why does the ball move?", DEFAULT_LOCAL_MODEL)
            self.assertEqual(answer, "Local hypothesis")
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, LOCAL_ENDPOINT)
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], DEFAULT_LOCAL_MODEL)
            self.assertEqual(payload["prompt"], "Why does the ball move?")
            self.assertFalse(payload["stream"])
            self.assertTrue(opener.called)
        with patch("sonara_research.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = urllib.error.URLError("Connection refused")
            with self.assertRaisesRegex(ValueError, "ollama serve"):
                ollama_local_generate("Question?", DEFAULT_LOCAL_MODEL)
        with self.assertRaises(ValueError):
            ollama_local_generate("Question?", "https://example.com/injected")
        with self.assertRaises(ValueError):
            ResearchCognition(self.root).feedback("Question?", 2.0)

    def test_live_session_local_research_is_unverified_and_never_promoted(self):
        with patch("sonara_research.ollama_local_generate",
                   return_value="A tentative explanation, not proof.") as generated:
            with patch("builtins.input",
                       side_effect=["/research Why does it float?", "/rate-research 0.8", "/quit"]):
                run_session(self.root, None, None, None, None, False, enable_research=True)
        generated.assert_called_once_with("Why does it float?", DEFAULT_LOCAL_MODEL)
        research_rows = read_records(self.root / "research_events.jsonl")
        rating_rows = read_records(self.root / "research_feedback.jsonl")
        self.assertEqual(len(research_rows), 1)
        self.assertEqual(len(rating_rows), 1)
        self.assertEqual(research_rows[0]["source"], "local_model")
        self.assertEqual(research_rows[0]["status"], "unverified_local_model_output")
        self.assertEqual(len(SonaraMemorySystem(self.root / "sonara_memories").memories), 0)
        self.assertFalse((self.root / "world_model.pt").exists())

    def test_local_research_opt_in_and_no_server_needed_for_disabled_mode(self):
        with patch("sonara_research.ollama_local_generate") as generated:
            with patch("builtins.input", side_effect=["/research Why?", "/quit"]):
                run_session(self.root, None, None, None, None, False, enable_research=False)
            generated.assert_not_called()
        with self.assertRaisesRegex(ValueError, "1-4000"):
            ResearchCognition(self.root, transport=lambda *_: self.fail("unexpected")).research("")


if __name__ == "__main__":
    unittest.main()
