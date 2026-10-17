"""Controlled ablation: how much *verified* replay is needed for temporal learning?

Identical architecture/seed/scenarios/optimizer/40 gradient steps for every cap.
Only the number of already-observed verified historical examples per update
changes. No inference of counterfactual labels and no new memory owner.
"""
import argparse
import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path

import torch

from experiment_world_learning import build_experiment_datasets, evaluate_model
from sonara_world_model import SonaraWorldModel


CAPS = (None, 1, 2, 4)
LABELS = ("full_replay", "replay_1", "replay_2", "replay_4")


def run_seed(seed: int, steps: int):
    train, unseen = build_experiment_datasets(seed)
    pairs = [(sc.events, q.question, q.target) for sc in train for q in sc.questions]
    runs = {}
    for cap, label in zip(CAPS, LABELS):
        torch.manual_seed(seed)
        owner = SonaraWorldModel(device="cpu", dim=24, lr=0.003)
        owner._max_replay_examples_per_update = cap
        rng = random.Random(seed)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        tracemalloc.start()
        replay_used = 0
        for _ in range(steps):
            events, question, target = rng.choice(pairs)
            result = owner.learn_step(events, question, target)
            replay_used += result["replay_examples"]
        train_wall = time.perf_counter() - start_wall
        train_cpu = time.process_time() - start_cpu
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        evaluation = evaluate_model(owner, unseen, "heldout")
        runs[label] = {
            "gradient_steps": steps,
            "mean_replayed_examples_per_step": round(replay_used / steps, 3),
            "cpu_train_seconds": round(train_cpu, 3),
            "wall_train_seconds": round(train_wall, 3),
            "peak_python_kib": round(peak / 1024, 2),
            "heldout_accuracy": evaluation["accuracy"],
            "previous_state_accuracy": evaluation["by_category"]["previous_state"]["accuracy"],
            "multi_object_accuracy": evaluation["by_category"]["multi_object"]["accuracy"],
        }
    return {"seed": seed, "variants": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="artifacts/replay_budget_report.json")
    args = parser.parse_args()
    if not (1 <= args.seeds <= 10 and 1 <= args.steps <= 100):
        parser.error("bounded seeds and steps required")
    torch.set_num_threads(1)
    runs = [run_seed(42 + i, args.steps) for i in range(args.seeds)]
    summary = {
        label: {
            metric: round(statistics.mean(run["variants"][label][metric] for run in runs), 4)
            for metric in (
                "mean_replayed_examples_per_step", "cpu_train_seconds",
                "wall_train_seconds", "peak_python_kib",
                "heldout_accuracy", "previous_state_accuracy",
                "multi_object_accuracy",
            )
        } for label in LABELS
    }
    report = {
        "seeds": args.seeds,
        "gradient_steps_per_condition_per_seed": args.steps,
        "meaning": "Same verified training observations, architecture, and held-out scenarios; only replay examples per gradient update vary.",
        "limitation": "Synthetic recurring language templates; Python peak excludes native tensor allocations.",
        "summary": summary, "runs": runs,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    # The reference should preserve PR #9's previous-state capability; if not,
    # the benchmark is not comparable and must not be used to choose a cap.
    if any(run["variants"]["full_replay"]["previous_state_accuracy"] < 1.0 for run in runs):
        raise SystemExit("Replay benchmark reference lost expected temporal tracking")


if __name__ == "__main__":
    main()
