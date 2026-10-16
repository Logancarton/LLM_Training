"""Behavioral checks for Sonara's real pre-action/outcome learning route."""
from pathlib import Path
import tempfile
import unittest

from sonara_memory import SonaraMemorySystem
from experiment_causal_loop import (
    ACTIONS, DESTINATIONS, HELDOUT, TRAIN_ACTOR, TRAIN_OBJECT,
    TwoRouteWorld, context, question, run_seed,
)


class CausalExperienceTests(unittest.TestCase):
    def test_intervention_is_hidden_until_action_and_changes_across_worlds(self):
        even, odd = TwoRouteWorld(100), TwoRouteWorld(101)
        self.assertNotEqual(even.mapping["tap"], odd.mapping["tap"])
        self.assertNotEqual(even.act("tap"), even.act("hold"))
        before = context(TRAIN_ACTOR, TRAIN_OBJECT)
        self.assertTrue(all(even.mapping[a] not in question(TRAIN_ACTOR, TRAIN_OBJECT, a)
                            for a in ACTIONS if even.mapping[a] not in ("north", "south")))
        self.assertNotIn("moved", " ".join(before))
        self.assertNotIn("verified", " ".join(before).lower())
        with self.assertRaises(ValueError):
            even.act("unsafe")

    def test_real_feedback_updates_owner_and_persists_without_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_seed(100, dim=16, heldout_count=1, data_dir=tmp)
            trial, checks = result["trial"], result["verification"]
            self.assertTrue(all(checks.values()), checks)
            self.assertEqual(trial["verified_learning_steps"], 1)
            self.assertGreater(trial["neural_update"]["parameter_delta"], 0)
            self.assertEqual(trial["neural_update"]["trained_count"], 1)
            self.assertIn(trial["verified_observed_outcome"], DESTINATIONS)
            self.assertEqual(result["conditions"]["before_feedback"],
                             result["conditions"]["no_gradient_control"])
            self.assertEqual(result["conditions"]["after_verified_feedback"],
                             result["conditions"]["after_restart"])
            self.assertNotIn(TRAIN_ACTOR, [x[0] for x in HELDOUT])
            self.assertNotIn(TRAIN_OBJECT, [x[1] for x in HELDOUT])
            memory = SonaraMemorySystem(Path(result["artifacts_directory"]) / "sonara_memories")
            self.assertEqual(len(memory.memories), 2)
            self.assertEqual({m.source for m in memory.memories.values()},
                             {"agent_action", "environment_observation"})
            self.assertEqual(len(memory.recall("verified simulator observation", min_score=0.01)), 1)

    def test_lie_control_is_isolated_from_verified_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_seed(101, dim=16, heldout_count=1, data_dir=tmp)
            observed = report["trial"]["verified_observed_outcome"]
            wrong = next(x for x in DESTINATIONS if x != observed)
            memory = SonaraMemorySystem(Path(report["artifacts_directory"]) / "sonara_memories")
            outcomes = [m.content for m in memory.memories.values()
                        if m.source == "environment_observation"]
            self.assertEqual(len(outcomes), 1)
            self.assertIn(f"to {observed}.", outcomes[0])
            self.assertNotIn(f"to {wrong}.", outcomes[0])
            self.assertEqual(report["trial"]["false_label_control_update"]["trained_count"], 1)


if __name__ == "__main__":
    unittest.main()
