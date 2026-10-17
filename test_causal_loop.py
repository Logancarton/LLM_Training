"""Behavioral checks for Sonara's real pre-action/outcome learning route."""
from pathlib import Path
import tempfile
import unittest

from sonara_memory import SonaraMemorySystem
from experiment_causal_loop import (
    ACTIONS, DESTINATIONS, HELDOUT, TRAIN_ACTOR, TRAIN_OBJECT,
    TwoRouteWorld, context, question, run_seed, run_pair_seed,
)


class CausalExperienceTests(unittest.TestCase):
    def test_intervention_is_hidden_until_action_and_changes_across_worlds(self):
        even, odd = TwoRouteWorld(100), TwoRouteWorld(101)
        self.assertNotEqual(even.mapping["tap"], odd.mapping["tap"])
        self.assertNotEqual(even.act("tap"), even.act("hold"))
        before = context(TRAIN_ACTOR, TRAIN_OBJECT)
        self.assertTrue(all(question(TRAIN_ACTOR, TRAIN_OBJECT, a).endswith(
                            f"uses {a}?") for a in ACTIONS))
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
            self.assertTrue(any(m.id == trial["outcome_memory_id"] for m, _ in
                                memory.recall("verified observation using", min_score=0.01)))

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


    def test_two_verified_actions_train_same_world_model_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pair_seed(100, dim=16, heldout_count=1,
                                   steps=8, data_dir=tmp)
            self.assertEqual(result["verified_experience_count"], 2)
            self.assertEqual(result["contrast_gradient_steps"], 8)
            self.assertTrue(all(result["verification"].values()),
                            result["verification"])
            self.assertGreater(result["updates"]["action_contrast"]["parameter_delta"], 0)
            self.assertEqual(result["updates"]["action_contrast"]["trained_count"], 9)
            self.assertEqual(result["conditions"]["after_two_with_contrast"],
                             result["conditions"]["after_restart"])
            self.assertEqual(result["conditions"]["no_gradient"],
                             result["one_experience"]["conditions"]["before_feedback"])
            self.assertNotEqual(result["second_trial"]["action"],
                                result["one_experience"]["trial"]["selected_action"])
            self.assertNotEqual(result["second_trial"]["observed_outcome"],
                                result["one_experience"]["trial"]["verified_observed_outcome"])
            memory = SonaraMemorySystem(
                Path(result["one_experience"]["artifacts_directory"]) / "sonara_memories")
            self.assertEqual(len(memory.memories), 4)

    def test_contrast_rejects_unobserved_or_unrepresentable_counterfactual(self):
        from sonara_world_model import SonaraWorldModel
        model = SonaraWorldModel(device="cpu", dim=16)
        events = context(TRAIN_ACTOR, TRAIN_OBJECT)
        query_tap = question(TRAIN_ACTOR, TRAIN_OBJECT, "tap")
        query_hold = question(TRAIN_ACTOR, TRAIN_OBJECT, "hold")
        before = [weight.detach().clone() for weight in model.model.parameters()]
        for pairs in (
            [(query_tap, "north")],
            [(query_tap, "north"), (query_tap, "south")],
            [(query_tap, "north"), (query_hold, "north")],
            [(query_tap, "north"), (query_hold, "secret_destination")],
        ):
            with self.assertRaises(ValueError):
                model.learn_action_contrast(events, pairs)
        self.assertEqual(model.trained_examples_count, 0)
        self.assertTrue(all(
            (weight == previous).all().item()
            for weight, previous in zip(model.model.parameters(), before)))




if __name__ == "__main__":
    unittest.main()
