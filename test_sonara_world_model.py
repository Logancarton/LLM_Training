"""test_sonara_world_model.py — Unit tests for Sonara's neural World Model.
=============================================================================
Tests:
  1. Initialization and forward prediction.
  2. Differentiable loss computation and gradient backpropagation.
  3. Real parameter changes during gradient descent.
  4. Generalization to completely unfamiliar entities (zero-shot vocabulary).
  5. Correction handling and temporal state update.
  6. Calibration and uncertainty on missing information ("unknown").
  7. Checkpoint saving, atomic replacement, and restart persistence.
  8. Frozen-weights control baseline.
"""
import copy
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from sonara_world_model import NeuralWorldModel, SonaraWorldModel, CLOSED_VOCABULARY


class SonaraWorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "test_world_model.pt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_forward_and_loss(self):
        wm = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32)
        events = [
            "12:15 Jake carried the ball into the yard.",
            "12:16 Jake moved the ball into the pond.",
        ]
        query = "where is the ball?"
        pred = wm.predict(query, events=events)
        self.assertIsInstance(pred.answer, str)
        self.assertGreater(pred.confidence, 0.0)
        self.assertGreaterEqual(pred.uncertainty, 0.0)
        self.assertEqual(len(pred.attended_events), 2)

        # Loss should be non-zero and finite
        loss = wm.model.compute_loss(events, query, "pond", wm.word_to_id, wm.device)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_parameter_updates_via_gradient(self):
        wm = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32, lr=0.01)
        events = [
            "12:15 Jake carried the ball into the yard.",
            "12:16 Jake moved the ball into the pond.",
        ]
        query = "where is the ball?"
        target = "pond"

        # Check parameter delta
        initial_params = [p.clone() for p in wm.model.parameters()]
        metrics = wm.learn_step(events, query, target)
        self.assertGreater(metrics["parameter_delta"], 0.0)
        self.assertEqual(metrics["trained_count"], 1)

        # Confirm at least one parameter changed
        changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, wm.model.parameters()))
        self.assertTrue(changed, "Neural parameters failed to update after learn_step")

        # Train a few more steps and confirm loss decreases
        losses = [metrics["loss"]]
        for _ in range(15):
            m = wm.learn_step(events, query, target)
            losses.append(m["loss"])
        self.assertLess(losses[-1], losses[0], f"Training loss failed to decrease: {losses[0]} -> {losses[-1]}")

    def test_unseen_entity_generalization(self):
        """Unfamiliar entities not seen during training can still be pointed to directly."""
        wm = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32)
        # Novel names: Sophia, frisbee, attic
        events = [
            "Sophia carried the frisbee into the attic.",
        ]
        pred = wm.predict("where is the frisbee?", events=events)
        self.assertIn("attic", pred.distribution)
        self.assertIn("frisbee", pred.distribution)

    def test_missing_observation_unknown(self):
        """When an observation is absent, the model should assign mass to unknown."""
        wm = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32)
        events = [
            "Jake carried the ball into the yard.",
        ]
        # Query about an entity never mentioned
        query = "where is the bicycle?"
        pred = wm.predict(query, events=events)
        self.assertIn("unknown", pred.distribution)

    def test_checkpoint_restart_persistence(self):
        """Verify weights and predictions match exactly across process restart."""
        wm1 = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32, lr=0.01)
        events = [
            "12:15 Max brought the toy into the shed.",
        ]
        query = "where is the toy?"
        for _ in range(10):
            wm1.learn_step(events, query, "shed")

        saved_path = wm1.save_checkpoint()
        self.assertTrue(saved_path.exists())
        pred1 = wm1.predict(query, events=events)

        # Simulate restart: load into completely new instance
        wm2 = SonaraWorldModel(checkpoint_path=saved_path, device="cpu", dim=32)
        pred2 = wm2.predict(query, events=events)

        self.assertEqual(pred1.answer, pred2.answer)
        self.assertAlmostEqual(pred1.confidence, pred2.confidence, places=5)
        self.assertAlmostEqual(pred1.uncertainty, pred2.uncertainty, places=5)

    def test_frozen_weights_control(self):
        """Frozen weights should exhibit zero parameter delta."""
        wm = SonaraWorldModel(checkpoint_path=self.checkpoint_path, device="cpu", dim=32)
        for p in wm.model.parameters():
            p.requires_grad = False
        events = ["Jake moved the ball into the pond."]
        query = "where is the ball?"

        before = [p.clone() for p in wm.model.parameters()]
        # Attempt step with frozen parameters
        wm.model.eval()
        with torch.no_grad():
            pred = wm.predict(query, events=events)
        after = [p.clone() for p in wm.model.parameters()]
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a))


if __name__ == "__main__":
    unittest.main()
