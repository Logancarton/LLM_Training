"""Controlled Sonara experience -> prediction -> action -> outcome -> learning gate.

This is an *experiment harness*, not an autonomous production runtime or
another world model. The existing SonaraWorldModel owns prediction/gradients;
SonaraMemorySystem owns the evidence record. A hidden deterministic simulator
supplies ground truth only AFTER the chosen intervention is executed.

Primary question: does ONE verified experience change model predictions AND
goal-directed actions on new objects/actors? Compare identical starting weights
with no-feedback and deliberately false-label controls. Report failures.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from sonara_memory import SonaraMemorySystem
from sonara_world_model import SonaraWorldModel

ACTIONS = ("tap", "hold")
DESTINATIONS = ("north", "south")
TRAIN_ACTOR, TRAIN_OBJECT = "amara", "puck"
HELDOUT = (
    ("brielle", "marble"),
    ("cyrus", "beacon"),
    ("dalia", "parcel"),
    ("elio", "token"),
    ("faye", "capsule"),
    ("gunnar", "ribbon"),
)


@dataclass(frozen=True)
class TwoRouteWorld:
    """Environment dynamics, inaccessible to the neural predictor before action."""
    seed: int

    @property
    def mapping(self) -> Dict[str, str]:
        # Counterbalanced latent world; same words/actions can have opposite
        # effects across seeds. This rule is simulator truth, NOT model logic.
        return dict(zip(ACTIONS, DESTINATIONS if self.seed % 2 == 0
                        else reversed(DESTINATIONS)))

    def act(self, action: str) -> str:
        if action not in ACTIONS:
            raise ValueError("Unknown simulator action; no observation can be verified")
        return self.mapping[action]


def context(actor: str, obj: str) -> List[str]:
    """Observation prior to intervention. Neither control's result is revealed."""
    return [
        f"{actor} found the {obj} at the hub.",
        "The destinations are north and south.",
        "The available controls are tap and hold.",
    ]


def question(actor: str, obj: str, action: str) -> str:
    if action not in ACTIONS:
        raise ValueError("Unknown simulator action")
    return f"where will the {obj} go if {actor} uses {action}?"


def choose_action(model: SonaraWorldModel, events: Sequence[str],
                  actor: str, obj: str, goal: str) -> Tuple[str, Dict[str, float]]:
    """Fixed experimental scoring rule, NOT a new production planner.

    Calls only Sonara's existing neural predictor; never sees simulator truth.
    """
    if goal not in DESTINATIONS:
        raise ValueError("Goal must name an allowed destination")
    scores: Dict[str, float] = {}
    for action in ACTIONS:
        pred = model.predict(question(actor, obj, action), events=list(events))
        scores[action] = float(pred.distribution.get(goal, 0.0))
    selected = max(ACTIONS, key=lambda a: scores[a])  # deterministic tie
    return selected, scores


def evaluate(model: SonaraWorldModel, world: TwoRouteWorld,
             heldout: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """No gradients or held-out feedback; simulator used only for scoring.

    Each independent probe resets the visible context. Observed outcomes from
    training and evaluation never enter the held-out model inputs.
    """
    successes = 0
    conditional_correct = 0
    target_probability = 0.0
    action_count = 0
    prediction_count = 0
    probes = []
    for actor, obj in heldout:
        events = context(actor, obj)
        for action in ACTIONS:
            prediction = model.predict(question(actor, obj, action), events=events)
            truth = world.act(action)  # evaluator-only; not a model input
            opposite = next(x for x in DESTINATIONS if x != truth)
            p_truth = prediction.distribution.get(truth, 0.0)
            p_wrong = prediction.distribution.get(opposite, 0.0)
            conditional_correct += int(p_truth > p_wrong)
            target_probability += p_truth
            prediction_count += 1
        for goal in DESTINATIONS:
            selected, scores = choose_action(model, events, actor, obj, goal)
            actual = world.act(selected)
            successes += int(actual == goal)
            action_count += 1
            if len(probes) < 5:
                probes.append({
                    "actor": actor, "object": obj, "goal": goal,
                    "selected_action": selected, "actual_result": actual,
                    "success": actual == goal, "neural_action_scores": scores,
                })
    return {
        "actions": action_count,
        "goal_successes": successes,
        "goal_success_rate": round(successes / max(1, action_count), 4),
        "predictions": prediction_count,
        "correct_destination_rankings": conditional_correct,
        "destination_ranking_accuracy": round(conditional_correct / max(1, prediction_count), 4),
        "mean_probability_of_true_destination": round(target_probability / max(1, prediction_count), 6),
        "probe_examples": probes,
    }


def run_seed(seed: int, *, dim: int = 24, lr: float = 0.004,
             heldout_count: int = 3, data_dir: Path | str | None = None) -> Dict[str, Any]:
    """One real interaction and exactly one verified neural learning step."""
    if not 1 <= heldout_count <= len(HELDOUT) or dim < 8 or lr <= 0:
        raise ValueError("Invalid dim, learning rate, or held-out count")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    if data_dir is None:
        with tempfile.TemporaryDirectory(prefix="sonara-causal-") as folder:
            return _run_seed(seed, Path(folder), dim, lr, heldout_count)

    root = Path(data_dir) / "causal_runs" / f"seed_{seed}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    return _run_seed(seed, root, dim, lr, heldout_count)


def _run_seed(seed: int, root: Path, dim: int, lr: float,
              heldout_count: int) -> Dict[str, Any]:
    world = TwoRouteWorld(seed)
    memory = SonaraMemorySystem(root / "sonara_memories")
    initial_path = root / "initial_world.pt"
    model = SonaraWorldModel(device="cpu", dim=dim, lr=lr)
    model.save_checkpoint(initial_path)

    # Both controls share EXACTLY the initial neural weights and input format.
    frozen = SonaraWorldModel(checkpoint_path=initial_path, device="cpu", dim=dim, lr=lr)
    false_feedback = SonaraWorldModel(checkpoint_path=initial_path, device="cpu", dim=dim, lr=lr)
    unseen = HELDOUT[:heldout_count]
    assert all(a != TRAIN_ACTOR and o != TRAIN_OBJECT for a, o in unseen)
    before = evaluate(model, world, unseen)

    # The world is unknown while the model selects a single exploratory action.
    prior_events = context(TRAIN_ACTOR, TRAIN_OBJECT)
    desired = DESTINATIONS[seed % 2]
    action, scores = choose_action(model, prior_events, TRAIN_ACTOR, TRAIN_OBJECT, desired)
    ask = question(TRAIN_ACTOR, TRAIN_OBJECT, action)
    pre = model.predict(ask, events=prior_events)

    action_memory = memory.remember(
        f"Simulator trial: {TRAIN_ACTOR} used {action} on {TRAIN_OBJECT}; target {desired}.",
        source="agent_action", session_id=f"causal_{seed}")
    observed = world.act(action)  # hidden effect revealed only here
    outcome_memory = memory.remember(
        f"Verified simulator observation: using {action} moved {TRAIN_OBJECT} to {observed}.",
        source="environment_observation", session_id=f"causal_{seed}")
    # Existing world owner ingests observation; training uses PRE-OUTCOME context.
    model.observe(outcome_memory.content)
    true_metrics = model.learn_step(prior_events, ask, observed)

    # Explicit invalid-feedback control: not an observation, never put in
    # memory or the real learner. Exposes label-driven shortcuts in results.
    opposite = next(dest for dest in DESTINATIONS if dest != observed)
    false_metrics = false_feedback.learn_step(prior_events, ask, opposite)

    after = evaluate(model, world, unseen)
    frozen_eval = evaluate(frozen, world, unseen)
    wrong_eval = evaluate(false_feedback, world, unseen)

    trained_path = root / "trained_world.pt"
    model.save_checkpoint(trained_path)
    restored = SonaraWorldModel(checkpoint_path=trained_path, device="cpu", dim=dim, lr=lr)
    after_restart = evaluate(restored, world, unseen)
    reread = SonaraMemorySystem(root / "sonara_memories")
    memory_ok = (len(reread.memories) == 2
                 and {m.source for m in reread.memories.values()}
                 == {"agent_action", "environment_observation"}
                 and any(m.id == outcome_memory.id and m.content == outcome_memory.content
                         for m in reread.memories.values()))
    prediction_ok = after_restart == after
    unchanged_ok = frozen_eval == before

    report: Dict[str, Any] = {
        "seed": seed,
        "synthetic_environment": True,
        "latent_world_rule_for_audit_only": world.mapping,
        "trial": {
            "prior_events": prior_events, "question": ask,
            "goal": desired, "selected_action": action,
            "pre_outcome_prediction": {
                "answer": pre.answer, "confidence": pre.confidence,
                "distribution": {dest: pre.distribution.get(dest, 0.0) for dest in DESTINATIONS},
            },
            "action_scores": scores,
            "action_memory_id": action_memory.id,
            "outcome_memory_id": outcome_memory.id,
            "verified_observed_outcome": observed,
            "prediction_error": int(pre.answer != observed),
            "verified_learning_steps": 1,
            "neural_update": true_metrics,
            "false_label_control_update": false_metrics,
        },
        "conditions": {
            "before_feedback": before,
            "after_verified_feedback": after,
            "no_gradient_control": frozen_eval,
            "incorrect_label_control": wrong_eval,
            "after_restart": after_restart,
        },
        "verification": {
            "same_initial_weights_control": unchanged_ok,
            "parameter_update_occurred": true_metrics["parameter_delta"] > 0.0,
            "learning_count_exactly_one": model.trained_examples_count == 1,
            "restart_predictions_identical": prediction_ok,
            "memory_restart_verified": memory_ok,
            "isolated_heldout_actors_objects": True,
        },
        "artifacts_directory": str(root),
    }
    report["improvement"] = {
        "heldout_goal_success_delta": round(
            after["goal_success_rate"] - before["goal_success_rate"], 4),
        "heldout_target_probability_delta": round(
            after["mean_probability_of_true_destination"]
            - before["mean_probability_of_true_destination"], 6),
        "heldout_ranking_accuracy_delta": round(
            after["destination_ranking_accuracy"]
            - before["destination_ranking_accuracy"], 4),
        "outperformed_incorrect_feedback_on_goal_success": (
            after["goal_success_rate"] > wrong_eval["goal_success_rate"]),
    }
    return report


def run_experiment(*, seeds: int = 3, dim: int = 24, lr: float = 0.004,
                   heldout_count: int = 3, data_dir: str | Path = "learning_data",
                   output: str | Path = "artifacts/causal_learning_report.json") -> Dict[str, Any]:
    if not 1 <= seeds <= 12:
        raise ValueError("Seeds must be between 1 and 12 (bounded laptop workload)")
    runs = [run_seed(100 + i, dim=dim, lr=lr,
                     heldout_count=heldout_count, data_dir=data_dir)
            for i in range(seeds)]
    metrics = [r["improvement"] for r in runs]
    summary = {
        "seeds": seeds,
        "one_verified_experience_per_seed": True,
        "mean_goal_success_delta": round(sum(x["heldout_goal_success_delta"] for x in metrics) / seeds, 4),
        "mean_target_probability_delta": round(
            sum(x["heldout_target_probability_delta"] for x in metrics) / seeds, 6),
        "mean_ranking_accuracy_delta": round(
            sum(x["heldout_ranking_accuracy_delta"] for x in metrics) / seeds, 4),
        "seeds_improved_goal_success": sum(x["heldout_goal_success_delta"] > 0 for x in metrics),
        "seeds_improved_target_probability": sum(x["heldout_target_probability_delta"] > 0 for x in metrics),
        "all_verifications_passed": all(all(r["verification"].values()) for r in runs),
    }
    # Improvement is empirical, not implied by one weight update.
    report = {
        "experiment": "single verified experience, novel-entity causal action transfer",
        "limits": ("Tiny synthetic two-action world; trained on one action and one result. "
                   "Unseen objects are not unseen dynamics. Neural gradients alone do not "
                   "prove causal understanding or general-world intelligence."),
        "summary": summary,
        "runs": runs,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sonara single-experience causal transfer test")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--heldout", type=int, default=3)
    parser.add_argument("--data-dir", default="learning_data")
    parser.add_argument("--output", default="artifacts/causal_learning_report.json")
    args = parser.parse_args()
    report = run_experiment(seeds=args.seeds, dim=args.dim, lr=args.lr,
                            heldout_count=args.heldout, data_dir=args.data_dir,
                            output=args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
