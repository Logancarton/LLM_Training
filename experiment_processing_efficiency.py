"""Reproducible CPU ablations of Sonara's existing temporal world-model owner.

Compares PR #9-equivalent scalar preprocessing/scoring against single-component
and combined optimizations using identical weights and verified input events.
This benchmark DOES NOT by itself prove held-out world reasoning or action choice;
the existing world/causal experiment CI gates remain authoritative for that.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import torch

from sonara_world_model import SonaraWorldModel


ABLATIONS = {
    "pr9_reference": (False, False, False),
    "event_cache_only": (True, False, False),
    "char_bags_only": (False, True, False),
    "pointer_batch_only": (False, False, True),
    "all_optimized": (True, True, True),
}
EVENTS = [
    "12:15 jake carried the ball into the yard.",
    "12:16 jake moved the ball into the shed.",
    "12:17 lucy placed the stick into the pond.",
    "12:18 correction: the ball never reached the shed; it stayed in the yard.",
]
QUESTIONS = [
    "where is the ball?",
    "where was the ball before 12:16?",
    "what did lucy place into the pond?",
    "did 12:15 happen before 12:17?",
]
TARGETS = ["yard", "yard", "stick", "yes"]


def run_seed(seed: int, repetitions: int = 8) -> dict:
    torch.manual_seed(seed)
    owner = SonaraWorldModel(device="cpu", dim=24)
    results = {}
    reference = None
    for name, (cache, bags, batch) in ABLATIONS.items():
        owner.model._cache_event_plans = cache
        owner.model._event_plan_cache.clear()
        owner.model.embedding._vectorized_char_bags = bags
        owner.model.head._vectorized_token_scoring = batch
        # Warm up network kernels. No training; identical parameters for all
        # ablations, so behavior differences cannot be explained by seed drift.
        for question in QUESTIONS:
            owner.predict(question, events=EVENTS)
        predictions = [owner.predict(q, events=EVENTS).distribution for q in QUESTIONS]
        if reference is None:
            reference = predictions
        max_probability_delta = max(
            abs(a[token] - b[token])
            for a, b in zip(reference, predictions)
            for token in a
        )
        tracemalloc.start()
        wall = time.perf_counter()
        cpu = time.process_time()
        for _ in range(repetitions):
            for q in QUESTIONS:
                owner.predict(q, events=EVENTS)
        wall = time.perf_counter() - wall
        cpu = time.process_time() - cpu
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Real optimizer steps, with the same weights restored between
        # ablations, are verified separately in test_world_efficiency.py.
        results[name] = {
            "wall_ms_per_prediction": round(1000 * wall / (repetitions * len(QUESTIONS)), 4),
            "cpu_ms_per_prediction": round(1000 * cpu / (repetitions * len(QUESTIONS)), 4),
            "peak_python_kib": round(peak_python / 1024, 2),
            "max_probability_delta": round(max_probability_delta, 8),
            "event_cache_entries": len(owner.model._event_plan_cache),
        }
    return {"seed": seed, "variants": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--output", default="artifacts/processing_efficiency_report.json")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20 or not 1 <= args.repetitions <= 100:
        parser.error("seeds/repetitions must be bounded")
    torch.set_num_threads(1)
    runs = [run_seed(seed + 200, args.repetitions) for seed in range(args.seeds)]
    summary = {}
    for name in ABLATIONS:
        summary[name] = {
            "mean_wall_ms": round(statistics.mean(
                run["variants"][name]["wall_ms_per_prediction"] for run in runs
            ), 4),
            "mean_cpu_ms": round(statistics.mean(
                run["variants"][name]["cpu_ms_per_prediction"] for run in runs
            ), 4),
            "mean_peak_python_kib": round(statistics.mean(
                run["variants"][name]["peak_python_kib"] for run in runs
            ), 2),
            "max_probability_delta": max(
                run["variants"][name]["max_probability_delta"] for run in runs
            ),
        }
    baseline = summary["pr9_reference"]["mean_cpu_ms"]
    optimized = summary["all_optimized"]["mean_cpu_ms"]
    report = {
        "reference": "PR #9-equivalent behavior on the exact same weights",
        "limitations": "Synthetic repeated questions; Python peak does not measure complete tensor memory; CPU timings are environment-dependent.",
        "seeds": args.seeds,
        "repetitions": args.repetitions,
        "gradient_steps": 0,
        "summary": summary,
        "cpu_speedup_relative_to_pr9": round(baseline / max(optimized, 1e-8), 3),
        "runs": runs,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cpu_speedup_relative_to_pr9": report["cpu_speedup_relative_to_pr9"], "variants": summary}, indent=2))
    if any(summary[name]["max_probability_delta"] > 2e-5 for name in ABLATIONS):
        raise SystemExit("Optimization changed inference distributions beyond tolerance")


if __name__ == "__main__":
    main()
