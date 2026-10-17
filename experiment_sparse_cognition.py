"""Controlled opt-in sparse-memory benchmark. Synthetic evidence; not real-world cognition.

Uses the SAME SonaraWorldModel weights for every retrieval mode per seed.
Verified labels train the world and attention heads only on training episodes.
Unseen names, distractors, corrected observations and alternate wording are
reserved for held-out evaluation. No labels are passed to retrieval at test time.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import statistics
import time
from pathlib import Path

import torch

from sonara_world_model import (
    SonaraWorldModel, tokenize, STOPWORDS, STRUCTURAL_WORDS,
)


def episode(actor: str, obj: str, start: str, destination: str, count: int,
            *, corrected: bool = False, paraphrase: bool = False):
    if paraphrase:
        first = f"At 08:01 {actor} was seen carrying {obj} toward {start}."
        last = f"At 08:56 {actor} placed {obj} inside {destination}."
        correction = f"At 08:59 correction: {obj} remained in {start}, not {destination}."
        query = f"what is the current place of {obj}?"
    else:
        first = f"At 08:01 {actor} moved {obj} into {start}."
        last = f"At 08:56 {actor} moved {obj} from {start} into {destination}."
        correction = f"At 08:59 correction: {obj} never reached {destination}; it stayed in {start}."
        query = f"where is {obj}?"
    noise = [
        f"At 08:{i + 5:02d} witness_{i} noticed instrument_{i} by shelf_{i}."
        for i in range(count)
    ]
    history = [first] + noise + [last]
    if corrected:
        history.append(correction)
    answer = start if corrected else destination
    return history, query, answer


def fixed_indices(history, question, budget):
    cues = set(tokenize(question)) - STOPWORDS - STRUCTURAL_WORDS
    scores = [len(cues.intersection(tokenize(event))) for event in history]
    order = sorted(range(len(history)), key=lambda i: (-scores[i], i))
    return sorted(order[:budget])


def evaluate(owner, history, query, answer, mode, budget, rng):
    owner.clear_session()
    for event in history:
        owner.observe(event)
    t0 = time.process_time()
    if mode == "learned_recovery":
        owner._learned_sparse_activation = True
        owner._max_active_events = budget
        owner._sparse_recovery_limit = max(2 * budget, 16)
        owner._sparse_min_confidence = 0.35
        prediction = owner.predict(query)
        trace = dict(owner.last_sparse_trace)
        activated = trace.get("activated_events", len(history))
        recovered = trace.get("recovered", False)
        unresolved = trace.get("unresolved", False)
    else:
        owner._learned_sparse_activation = False
        if mode == "full":
            indices = list(range(len(history)))
        elif mode == "recency":
            indices = list(range(max(0, len(history) - budget), len(history)))
        elif mode == "random":
            indices = sorted(rng.sample(range(len(history)), k=min(budget, len(history))))
        elif mode == "fixed_overlap":
            indices = fixed_indices(history, query, budget)
        elif mode == "learned":
            indices = owner._rank_learned_events(history, query, budget)
        else:
            raise ValueError(mode)
        prediction = owner.predict(query, events=[history[i] for i in indices])
        activated = len(indices)
        recovered = False
        unresolved = False
    elapsed = (time.process_time() - t0) * 1000
    return {
        "answer": prediction.answer, "correct": prediction.answer == answer,
        "probability_verified": prediction.distribution.get(answer, 0.0),
        "activated": activated, "cpu_ms": elapsed,
        "recovered": recovered, "unresolved": unresolved,
    }


def run(seeds=3, budget=4, heldout=5):
    if seeds < 1 or heldout < 1 or not 1 <= budget <= 16:
        raise ValueError("seeds/heldout must be positive and budget 1..16")
    torch.set_num_threads(1)
    modes = ("full", "recency", "random", "fixed_overlap", "learned", "learned_recovery")
    records = []
    training = []
    for seed in range(seeds):
        torch.manual_seed(seed + 70)
        rng = random.Random(seed + 70)
        owner = SonaraWorldModel(device="cpu", dim=24, lr=0.003)
        owner._learned_sparse_activation = True
        # Train on three observed consequences. World/selector updates share
        # the SAME verified feedback, with no special evaluation labels.
        known = [
            ("Mara", "ball", "courtyard", "garage", 5, False),
            ("Lina", "book", "attic", "pantry", 7, True),
            ("Drew", "key", "porch", "cellar", 9, False),
        ]
        for actor, obj, start, dest, noise, corrected in known:
            events, query, answer = episode(actor, obj, start, dest, noise, corrected=corrected)
            result = owner.learn_step(events, query, answer)
            training.append({
                "seed": seed, "world_loss": result["loss"],
                "sparse_attention": result.get("sparse_attention"),
            })
        test_names = (
            ("Suri", "lantern", "studio", "workshop"),
            ("Nadia", "scarf", "hallway", "balcony"),
            ("Owen", "notebook", "lobby", "closet"),
            ("Priya", "compass", "barn", "library"),
            ("Jules", "violin", "market", "bedroom"),
            ("Rafi", "telescope", "garden", "office"),
        )
        for case_idx in range(heldout):
            actor, obj, start, dest = test_names[case_idx % len(test_names)]
            corrected = case_idx % 2 == 1
            paraphrase = case_idx % 3 == 2
            events, query, answer = episode(
                actor, obj, start, dest,
                count=12 + (case_idx % 4) * 4,
                corrected=corrected, paraphrase=paraphrase
            )
            baseline = evaluate(owner, events, query, answer, "full", budget, rng)
            for mode in modes:
                result = baseline if mode == "full" else evaluate(
                    owner, events, query, answer, mode, budget, rng
                )
                records.append({
                    "seed": seed, "case": case_idx, "mode": mode,
                    "ground_truth": answer, "teacher_answer": baseline["answer"],
                    "full_context_correct": baseline["correct"],
                    "matches_full": result["answer"] == baseline["answer"],
                    **result,
                })
    summary = {}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            "samples": len(rows),
            "verified_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "agreement_with_full": round(sum(r["matches_full"] for r in rows) / len(rows), 4),
            "mean_probability_verified": round(statistics.mean(r["probability_verified"] for r in rows), 4),
            "mean_cpu_ms": round(statistics.mean(r["cpu_ms"] for r in rows), 3),
            "mean_activated_events": round(statistics.mean(r["activated"] for r in rows), 2),
            "recoveries": sum(r["recovered"] for r in rows),
            "unresolved": sum(r["unresolved"] for r in rows),
        }
    return {
        "status": "EXPERIMENTAL; nondeployment, synthetic, no action selection benchmark",
        "design": {
            "seeds": seeds, "heldout_per_seed": heldout, "budget": budget,
            "training_episodes_per_seed": 3, "world_backbone": "PR10 temporal",
            "same_weights_within_seed": True,
            "teacher_not_authoritative": True,
            "native_memory_limit": "Peak process RSS is cumulative, not per variant",
        },
        "training": training,
        "world_gradient_updates": seeds * 3,
        "sparse_gradient_updates": sum(
            item.get("sparse_attention", {}).get("updates", 0)
            for item in training if item.get("sparse_attention")
        ),
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "summary": summary, "cases": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--heldout", type=int, default=5)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--output", default="artifacts/learned_sparse_report.json")
    args = parser.parse_args()
    report = run(args.seeds, args.budget, args.heldout)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
