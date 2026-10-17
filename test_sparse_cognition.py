"""Research gates for verified, learned sparse activation in Sonara's world owner."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sonara_memory import SonaraMemorySystem
from sonara_world_model import SonaraWorldModel


class LearnedSparseCognitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        torch.set_num_threads(1)
        self.owner = SonaraWorldModel(device="cpu", dim=24)
        self.events = (
            ["Mara carried the blue ball into the courtyard."]
            + [f"Neighbor_{i} counted stone_{i} inside room_{i}." for i in range(7)]
            + ["Mara moved the blue ball into the greenhouse.",
               "Correction: the ball never reached the greenhouse, it remained in the courtyard."]
        )
        self.question = "where is the blue ball?"

    def test_untrained_selector_never_silently_prunes_evidence(self):
        for event in self.events:
            self.owner.observe(event)
        self.owner._learned_sparse_activation = True
        self.owner._max_active_events = 2
        implicit = self.owner.predict(self.question)
        full = self.owner.predict(self.question, events=list(self.events))
        self.assertEqual(implicit.distribution, full.distribution)
        self.assertEqual(self.owner.last_sparse_trace, {})

    def test_verified_learning_reaches_live_ranker_and_preserves_originals(self):
        self.owner._learned_sparse_activation = True
        verified = self.owner.learn_step(
            self.events[:3] + ["Verified: Mara put the blue ball in the observatory."],
            self.question, "observatory"
        )
        self.assertTrue(verified["sparse_attention"]["trained"])
        self.assertGreater(verified["sparse_attention"]["parameter_delta"], 0)
        self.assertGreater(self.owner._sparse_training_updates, 0)
        for event in self.events:
            self.owner.observe(event)
        original = list(self.owner.session_events)
        self.owner._max_active_events = 2
        self.owner._sparse_min_confidence = 1.01
        pred = self.owner.predict(self.question)
        trace = self.owner.last_sparse_trace
        self.assertTrue(trace["sparse"])
        self.assertEqual(trace["total_events"], len(original))
        self.assertEqual(trace["activated_events"], 4)
        self.assertTrue(trace["recovered"])
        self.assertTrue(trace["unresolved"])
        self.assertEqual(pred.answer, "unknown")
        self.assertEqual(self.owner.session_events, original)
        self.assertEqual(trace["indices"], sorted(set(trace["indices"])))

    def test_missing_verified_answer_never_trains_fake_grounding(self):
        before = [p.detach().clone() for p in self.owner.model.sparse_selector.parameters()]
        status = self.owner.train_sparse_attention(
            self.events, self.question, "unobserved_destination"
        )
        self.assertFalse(status["trained"])
        self.assertEqual(status["reason"], "answer_not_grounded_in_evidence")
        self.assertEqual(self.owner._sparse_training_updates, 0)
        for b, p in zip(before, self.owner.model.sparse_selector.parameters()):
            self.assertTrue(torch.equal(b, p))

    def test_restart_recovers_neural_attention_and_authoritative_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp) / "memories"
            memory = SonaraMemorySystem(str(evidence_dir))
            for event in self.events:
                memory.remember(event, source="user_observation")
            self.owner.train_sparse_attention(self.events, self.question, "courtyard", steps=2)
            checkpoint = Path(tmp) / "world.pt"
            self.owner.save_checkpoint(checkpoint)
            restarted = SonaraWorldModel(checkpoint_path=checkpoint, device="cpu", dim=24)
            self.assertFalse(restarted._learned_sparse_activation)
            self.assertEqual(restarted._sparse_training_updates, self.owner._sparse_training_updates)
            reloaded = SonaraMemorySystem(str(evidence_dir))
            for observed in reloaded.memories.values():
                if observed.source == "user_observation":
                    restarted.observe(observed.content)
            self.assertEqual(restarted.session_events, self.events)
            self.assertEqual(
                self.owner._rank_learned_events(self.events, self.question, 3),
                restarted._rank_learned_events(self.events, self.question, 3)
            )
            # A compatible pre-selector world checkpoint preserves all original
            # world weights while initializing only the absent optional head.
            legacy = torch.load(checkpoint, weights_only=False)
            legacy["version"] = 2
            legacy.pop("sparse_training_updates")
            legacy["state_dict"] = {
                k: v for k, v in legacy["state_dict"].items()
                if not k.startswith("sparse_selector.")
            }
            torch.save(legacy, checkpoint)
            migrated = SonaraWorldModel(checkpoint_path=checkpoint, device="cpu", dim=24)
            self.assertEqual(migrated._sparse_training_updates, 0)
            for k, v in legacy["state_dict"].items():
                self.assertTrue(torch.equal(v, migrated.model.state_dict()[k]))

    def test_no_training_from_unverified_query_or_session_reset(self):
        self.owner._learned_sparse_activation = True
        for event in self.events:
            self.owner.observe(event)
        self.owner.predict("What did Mara do?")
        self.assertEqual(self.owner._sparse_training_updates, 0)
        self.owner.clear_session()
        self.assertEqual(self.owner.session_events, [])
        self.assertEqual(self.owner.last_sparse_trace, {})


if __name__ == "__main__":
    unittest.main()
