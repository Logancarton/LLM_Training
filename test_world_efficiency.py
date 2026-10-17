"""Behavioral verification of Sonara's bounded processing optimizations."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sonara_world_model import SonaraWorldModel


class ProcessingEfficiencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)
        self.owner = SonaraWorldModel(device="cpu", dim=24, lr=0.002)
        self.events = [
            "12:15 jake carried the ball into the yard.",
            "12:16 jake moved the ball into the shed.",
            "12:17 lucy placed the stick into the pond.",
            "12:18 correction: the ball never reached the shed; it stayed in the yard.",
        ]
        self.queries = [
            "where is the ball?",
            "where was the ball before 12:16?",
            "what did lucy place into the pond?",
            "did 12:15 happen before 12:17?",
        ]

    def _set_optimized(self, enabled):
        model = self.owner.model
        model._cache_event_plans = enabled
        model.embedding._vectorized_char_bags = enabled
        model.head._vectorized_token_scoring = enabled
        model._event_plan_cache.clear()

    def test_equivalent_predictions_and_loss_across_ablations(self):
        self.owner.model.eval()
        self._set_optimized(False)
        originals = [self.owner.predict(q, events=self.events) for q in self.queries]
        baseline_loss = self.owner.model.compute_loss(
            self.events, self.queries[0], "yard", self.owner.word_to_id, self.owner.device
        ).detach().item()
        for ablation in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ):
            self.owner.model._cache_event_plans = ablation[0]
            self.owner.model.embedding._vectorized_char_bags = ablation[1]
            self.owner.model.head._vectorized_token_scoring = ablation[2]
            for query, original in zip(self.queries, originals):
                candidate = self.owner.predict(query, events=self.events)
                self.assertEqual(set(original.distribution), set(candidate.distribution))
                for token, prob in original.distribution.items():
                    self.assertAlmostEqual(prob, candidate.distribution[token], delta=2e-5)
            loss = self.owner.model.compute_loss(
                self.events, self.queries[0], "yard", self.owner.word_to_id, self.owner.device
            ).detach().item()
            self.assertAlmostEqual(loss, baseline_loss, delta=2e-5)

    def test_gradients_and_cache_never_reuse_neural_tensors(self):
        self._set_optimized(False)
        self.owner.model.train()
        raw = self.owner.model.compute_loss(
            self.events, self.queries[0], "yard", self.owner.word_to_id, self.owner.device
        )
        self.owner.optimizer.zero_grad(set_to_none=True)
        raw.backward()
        gradient_original = self.owner.model.embedding.hash_embeddings.weight.grad.detach().clone()
        self.owner.optimizer.zero_grad(set_to_none=True)
        self._set_optimized(True)
        optimized = self.owner.model.compute_loss(
            self.events, self.queries[0], "yard", self.owner.word_to_id, self.owner.device
        )
        optimized.backward()
        gradient_optimized = self.owner.model.embedding.hash_embeddings.weight.grad
        self.assertTrue(torch.allclose(gradient_original, gradient_optimized, atol=2e-5, rtol=2e-4))
        self.assertEqual(len(self.owner.model._event_plan_cache), 1)
        first = self.owner.model._prepare_events(self.events)
        self.assertIs(first, self.owner.model._prepare_events(self.events))
        before = self.owner.predict(self.queries[0], events=self.events).distribution
        update = self.owner.learn_step(self.events, self.queries[0], "yard")
        after = self.owner.predict(self.queries[0], events=self.events).distribution
        self.assertGreater(update["parameter_delta"], 0)
        self.assertNotEqual(before, after)

    def test_long_session_selects_bounded_relevant_evidence(self):
        self.owner.observe("Jake placed the ball in the yard.")
        for i in range(50):
            self.owner.observe(f"Person_{i} watched bird_{i} at garden_{i}.")
        self.owner.observe("Jake moved the ball into the shed.")
        selected = self.owner._activate_session_events("where is the ball?")
        self.assertLessEqual(len(selected), self.owner._max_active_events)
        self.assertIn("Jake placed the ball in the yard.", selected)
        self.assertIn("Jake moved the ball into the shed.", selected)
        self.assertEqual(len(self.owner.session_events), 52)
        implicit = self.owner.predict("where is the ball?").distribution
        explicit = self.owner.predict("where is the ball?", events=selected).distribution
        for token in implicit:
            self.assertAlmostEqual(implicit[token], explicit[token], delta=1e-7)
        self.assertEqual(len(self.owner.session_events), 52)

    def test_crowded_causal_chain_falls_back_to_full_evidence(self):
        for i in range(40):
            self.owner.observe(f"Jake moved the ball into location_{i}.")
        self.assertEqual(
            self.owner._activate_session_events("where is the ball?"),
            self.owner.session_events,
        )
        self.owner.clear_session()
        self.assertEqual(self.owner._session_event_terms, [])

    def test_replay_persists_and_old_checkpoint_is_rejected_before_state_mutation(self):
        self.owner.learn_step(self.events, "where is the ball?", "yard")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world.pt"
            self.owner.save_checkpoint(path)
            restored = SonaraWorldModel(checkpoint_path=path, device="cpu", dim=24)
            self.assertEqual(restored.trained_examples_count, self.owner.trained_examples_count)
            self.assertEqual(dict(restored.replay_buckets), dict(self.owner.replay_buckets))
            a = self.owner.predict("where is the ball?", events=self.events)
            b = restored.predict("where is the ball?", events=self.events)
            self.assertEqual(a.distribution, b.distribution)
            original_payload = torch.load(path, weights_only=False)
            incompatible = copy.deepcopy(original_payload)
            incompatible["state_dict"] = {"obsolete.weight": torch.zeros(2, 3)}
            torch.save(incompatible, path)
            fresh = SonaraWorldModel(device="cpu", dim=24)
            prior_vocab = copy.deepcopy(fresh.word_to_id)
            prior_weights = {k: v.clone() for k, v in fresh.model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "Incompatible world-model checkpoint"):
                fresh.load_checkpoint(path)
            self.assertEqual(fresh.word_to_id, prior_vocab)
            self.assertTrue(all(
                torch.equal(v, prior_weights[k]) for k, v in fresh.model.state_dict().items()
            ))


if __name__ == "__main__":
    unittest.main()
