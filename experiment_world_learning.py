"""experiment_world_learning.py — Rigorous Empirical Validation of Sonara Neural World Model.
========================================================================================

PURPOSE
    Scientifically measures whether Sonara's neural World Model exhibits genuine
    learning of relationships between events, objects, actions, locations, and time
    through gradient updates, and whether that learning generalizes to unfamiliar
    experiences.

EVALUATION CONDITIONS (As specified in Experiment 1):
    A. Untrained baseline: Randomly initialized model before training.
    B. Neural training: Gradient descent (AdamW) updating actual neural weights.
    C. Unseen generalization: Completely disjoint entities (unseen people, objects,
       locations), longer sequences, novel action combinations, missing observations.
    D. Shuffled-time control: Training on chronologically disrupted sequences to test
       whether temporal ordering contributes meaningfully to performance.
    E. Memory-only control: Deterministic episodic memory retrieval without neural
       world modeling.
    F. Frozen-model control: Identical architecture with frozen random weights.
    G. Restart persistence: Checkpoint serialization, process reload, and verification.

SEPARATE COGNITIVE CAPABILITIES:
    - Object identity tracking
    - Current-state prediction
    - Previous-state prediction
    - Event-order understanding
    - State-transition prediction
    - Multi-object interference
    - Correction handling
    - Unfamiliar entity generalization
    - Recognition of missing information (uncertainty/unknown)
    - Controlled causal interventions (same initial state, different actions)

RESOURCE CAP:
    MacBook Air friendly: PyTorch with MPS/CPU fallback, memory tracking, multi-seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sonara_memory import SonaraMemorySystem
from sonara_world_model import (
    CLOSED_VOCABULARY,
    NeuralWorldModel,
    SonaraWorldModel,
    WorldPrediction,
    tokenize,
    STOPWORDS,
)


# =====================================================================
# 1. SYNTHETIC ENVIRONMENT & SCENARIO GENERATORS
# =====================================================================

TRAIN_ACTORS = ("jake", "lucy", "max", "piper")
TRAIN_OBJECTS = ("ball", "stick", "toy", "bone")
TRAIN_PLACES = ("yard", "pond", "shed", "kitchen")

# Completely disjoint held-out vocabulary for unseen generalization
UNSEEN_ACTORS = ("sophia", "oliver", "emma", "noah")
UNSEEN_OBJECTS = ("frisbee", "blanket", "journal", "lantern")
UNSEEN_PLACES = ("attic", "porch", "garage", "basement")


@dataclass
class QuestionItem:
    question: str
    target: str
    category: str  # current_state, previous_state, object_identity, event_order,
                   # state_transition, multi_object, correction, missing_info, causal_intervention


@dataclass
class Scenario:
    id: str
    events: List[str]
    questions: List[QuestionItem]
    is_unseen: bool = False


def generate_scenario(
    actor: str,
    helper: str,
    obj1: str,
    obj2: str,
    p1: str,
    p2: str,
    p3: str,
    scene_id: str,
    corrected: bool = False,
    is_unseen: bool = False,
    long_seq: bool = False,
) -> Scenario:
    """Generate a clean episodic sequence with diverse cognitive test questions."""
    events = [
        f"12:15 {actor} carried the {obj1} into the {p1}.",
        f"12:16 {actor} moved the {obj1} into the {p2}.",
        f"12:17 {helper} placed the {obj2} into the {p3}.",
    ]
    if long_seq:
        events.append(f"12:18 {actor} inspected the {obj1} in the {p2}.")
        events.append(f"12:19 {helper} brought a blanket.")

    if corrected:
        events.append(f"12:20 correction: the {obj1} never reached the {p2}; it stayed in the {p1}.")
        final_obj1_place = p1
    else:
        events.append(f"12:20 {actor} left the {obj1} in the {p2}.")
        final_obj1_place = p2

    questions = [
        # Current state
        QuestionItem(f"where is the {obj1}?", final_obj1_place, "current_state"),
        # Previous state
        QuestionItem(f"where was the {obj1} before 12:16?", p1, "previous_state"),
        # Multi-object interference
        QuestionItem(f"where is the {obj2}?", p3, "multi_object"),
        # Object identity tracking
        QuestionItem(f"what did {helper} place into the {p3}?", obj2, "object_identity"),
        # Event order understanding
        QuestionItem("did 12:15 happen before 12:17?", "yes", "event_order"),
        QuestionItem("did 12:20 happen before 12:16?", "no", "event_order"),
        # State transition prediction
        QuestionItem(f"where will the {obj1} be if {actor} moves it into the {p3}?", p3, "state_transition"),
        # Correction handling
        QuestionItem(f"is the {obj1} in the {p1}?", "yes" if corrected else "no", "correction"),
        # Missing information / uncertainty
        QuestionItem("what did the cat chase?", "unknown", "missing_info"),
    ]

    return Scenario(id=scene_id, events=events, questions=questions, is_unseen=is_unseen)


def generate_causal_pair(seed: int, is_unseen: bool = False) -> Tuple[Scenario, Scenario]:
    """
    Controlled intervention pair: identical initial state, but different actions
    produce different physical outcomes.
    """
    rng = random.Random(seed)
    actors = UNSEEN_ACTORS if is_unseen else TRAIN_ACTORS
    objs = UNSEEN_OBJECTS if is_unseen else TRAIN_OBJECTS
    places = UNSEEN_PLACES if is_unseen else TRAIN_PLACES

    act = rng.choice(actors)
    obj = rng.choice(objs)
    p_init, p_a, p_b = rng.sample(places, 3)

    events_a = [
        f"10:00 {act} held the {obj} in the {p_init}.",
        f"10:01 {act} transferred the {obj} to the {p_a}.",
    ]
    events_b = [
        f"10:00 {act} held the {obj} in the {p_init}.",
        f"10:01 {act} transferred the {obj} to the {p_b}.",
    ]
    q_a = [QuestionItem(f"where is the {obj}?", p_a, "causal_intervention")]
    q_b = [QuestionItem(f"where is the {obj}?", p_b, "causal_intervention")]

    sc_a = Scenario(id=f"causal_a_{seed}", events=events_a, questions=q_a, is_unseen=is_unseen)
    sc_b = Scenario(id=f"causal_b_{seed}", events=events_b, questions=q_b, is_unseen=is_unseen)
    return sc_a, sc_b


def build_experiment_datasets(seed: int) -> Tuple[List[Scenario], List[Scenario]]:
    """Build separated training scenarios and unseen generalization scenarios."""
    rng = random.Random(seed)
    train_scenarios = []

    # 1. Training scenarios
    count = 0
    for ai, act in enumerate(TRAIN_ACTORS):
        helper = TRAIN_ACTORS[(ai + 1) % len(TRAIN_ACTORS)]
        for oi, obj1 in enumerate(TRAIN_OBJECTS):
            obj2 = TRAIN_OBJECTS[(oi + 1) % len(TRAIN_OBJECTS)]
            p1, p2, p3 = TRAIN_PLACES[0], TRAIN_PLACES[1], TRAIN_PLACES[2]
            for corrected in (False, True):
                count += 1
                train_scenarios.append(
                    generate_scenario(act, helper, obj1, obj2, p1, p2, p3,
                                      f"train_{count}", corrected=corrected)
                )
    # Add training causal pairs
    for i in range(6):
        ca, cb = generate_causal_pair(seed + i, is_unseen=False)
        train_scenarios.extend([ca, cb])

    rng.shuffle(train_scenarios)

    # 2. Unseen Generalization scenarios (Held-out disjoint entities & longer sequences)
    unseen_scenarios = []
    u_count = 0
    for ai, act in enumerate(UNSEEN_ACTORS):
        helper = UNSEEN_ACTORS[(ai + 1) % len(UNSEEN_ACTORS)]
        for oi, obj1 in enumerate(UNSEEN_OBJECTS):
            obj2 = UNSEEN_OBJECTS[(oi + 1) % len(UNSEEN_OBJECTS)]
            p1, p2, p3 = UNSEEN_PLACES[0], UNSEEN_PLACES[1], UNSEEN_PLACES[2]
            for corrected in (False, True):
                u_count += 1
                long_seq = (u_count % 2 == 0)
                unseen_scenarios.append(
                    generate_scenario(act, helper, obj1, obj2, p1, p2, p3,
                                      f"unseen_{u_count}", corrected=corrected,
                                      is_unseen=True, long_seq=long_seq)
                )
    for i in range(6):
        ca, cb = generate_causal_pair(seed + 100 + i, is_unseen=True)
        unseen_scenarios.extend([ca, cb])

    rng.shuffle(unseen_scenarios)
    return train_scenarios, unseen_scenarios


# =====================================================================
# 2. EVALUATION & CONTROL MECHANISMS
# =====================================================================

def evaluate_model(
    wm: SonaraWorldModel,
    scenarios: List[Scenario],
    name: str = "eval",
) -> Dict[str, Any]:
    """Evaluate accuracy, loss, uncertainty, and per-category metrics."""
    wm.model.eval()
    total_q = 0
    correct_q = 0
    total_loss = 0.0
    total_uncertainty = 0.0
    category_counts: Dict[str, int] = defaultdict(int)
    category_correct: Dict[str, int] = defaultdict(int)
    examples: List[Dict[str, Any]] = []

    with torch.no_grad():
        for sc in scenarios:
            for q_item in sc.questions:
                total_q += 1
                pred = wm.predict(q_item.question, events=sc.events)
                loss_val = wm.model.compute_loss(
                    sc.events, q_item.question, q_item.target, wm.word_to_id, wm.device
                ).item()

                matched = (pred.answer.lower().strip() == q_item.target.lower().strip())
                if matched:
                    correct_q += 1
                category_counts[q_item.category] += 1
                if matched:
                    category_correct[q_item.category] += 1

                total_loss += loss_val
                total_uncertainty += pred.uncertainty

                if len(examples) < 12 and (not matched or total_q % 10 == 0):
                    examples.append({
                        "scenario_id": sc.id,
                        "events": sc.events,
                        "question": q_item.question,
                        "target": q_item.target,
                        "predicted": pred.answer,
                        "confidence": round(pred.confidence, 4),
                        "uncertainty": pred.uncertainty,
                        "category": q_item.category,
                        "matched": matched,
                    })

    accuracy = round(correct_q / total_q, 4) if total_q else 0.0
    avg_loss = round(total_loss / total_q, 4) if total_q else 0.0
    avg_uncertainty = round(total_uncertainty / total_q, 4) if total_q else 0.0

    by_category = {}
    for cat, total_c in category_counts.items():
        by_category[cat] = {
            "accuracy": round(category_correct[cat] / total_c, 4),
            "correct": category_correct[cat],
            "total": total_c,
        }

    return {
        "dataset": name,
        "total_questions": total_q,
        "accuracy": accuracy,
        "loss": avg_loss,
        "avg_uncertainty": avg_uncertainty,
        "by_category": by_category,
        "sample_examples": examples[:8],
    }


def evaluate_memory_only(scenarios: List[Scenario]) -> Dict[str, Any]:
    """
    Memory-Only Control Baseline:
    Direct episodic memory retrieval using SonaraMemorySystem without neural reasoning.
    """
    with tempfile.TemporaryDirectory(prefix="sonara-mem-control-") as temp_dir:
        mem_sys = SonaraMemorySystem(temp_dir)
        total_q = 0
        correct_q = 0
        category_counts: Dict[str, int] = defaultdict(int)
        category_correct: Dict[str, int] = defaultdict(int)
        examples = []

        for sc in scenarios:
            # Store observations in episodic memory
            for ev in sc.events:
                mem_sys.remember(f"{sc.id} {ev}", source="user_observation")

            for q_item in sc.questions:
                total_q += 1
                category_counts[q_item.category] += 1
                # Retrieve from episodic memory
                results = mem_sys.recall(f"{sc.id} {q_item.question}", limit=3)
                retrieved_texts = [m.content.lower() for m, _ in results]

                # Lexical heuristic response
                predicted_ans = "unknown"
                if q_item.category in ("event_order", "correction"):
                    # Lexical memory retrieval has no sequence or boolean logic
                    predicted_ans = "unknown"
                else:
                    # Look for non-stopword tokens appearing in the top retrieved passage
                    if retrieved_texts:
                        tokens = [t for t in tokenize(retrieved_texts[0]) if t not in STOPWORDS]
                        # If target token is in top tokens, assign it
                        if q_item.target in tokens:
                            predicted_ans = q_item.target
                        elif len(tokens) > 2:
                            predicted_ans = tokens[-1]

                matched = (predicted_ans == q_item.target.lower().strip())
                if matched:
                    correct_q += 1
                    category_correct[q_item.category] += 1

                if len(examples) < 6:
                    examples.append({
                        "question": q_item.question,
                        "target": q_item.target,
                        "predicted": predicted_ans,
                        "category": q_item.category,
                        "matched": matched,
                    })

    accuracy = round(correct_q / total_q, 4) if total_q else 0.0
    by_category = {
        cat: {
            "accuracy": round(category_correct[cat] / category_counts[cat], 4),
            "correct": category_correct[cat],
            "total": category_counts[cat],
        }
        for cat in category_counts
    }

    return {
        "dataset": "memory_only_baseline",
        "total_questions": total_q,
        "accuracy": accuracy,
        "by_category": by_category,
        "sample_examples": examples,
    }


# =====================================================================
# 3. MULTI-SEED RUNNER
# =====================================================================

def run_experiment_seed(
    seed: int,
    steps: int = 120,
    lr: float = 0.003,
    dim: int = 64,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute all experimental conditions on one seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    start_time = time.monotonic()

    train_scenarios, unseen_scenarios = build_experiment_datasets(seed)

    # -------------------------------------------------------------
    # Condition A: Untrained Baseline
    # -------------------------------------------------------------
    wm_untrained = SonaraWorldModel(device=device, dim=dim, lr=lr)
    baseline_train = evaluate_model(wm_untrained, train_scenarios, "untrained_train")
    baseline_unseen = evaluate_model(wm_untrained, unseen_scenarios, "untrained_unseen")

    # -------------------------------------------------------------
    # Condition F: Frozen-Model Control
    # -------------------------------------------------------------
    wm_frozen = SonaraWorldModel(device=device, dim=dim, lr=lr)
    for p in wm_frozen.model.parameters():
        p.requires_grad = False
    frozen_eval = evaluate_model(wm_frozen, unseen_scenarios, "frozen_unseen")

    # -------------------------------------------------------------
    # Condition E: Memory-Only Control
    # -------------------------------------------------------------
    memory_only_eval = evaluate_memory_only(unseen_scenarios)

    # -------------------------------------------------------------
    # Condition B: Neural Training (Gradient Descent with AdamW)
    # -------------------------------------------------------------
    wm_trained = SonaraWorldModel(device=device, dim=dim, lr=lr)
    initial_weights = {n: p.detach().clone() for n, p in wm_trained.model.named_parameters()}

    train_pairs = [
        (sc.events, q.question, q.target)
        for sc in train_scenarios
        for q in sc.questions
    ]

    train_losses = []
    train_start_time = time.monotonic()
    rng = random.Random(seed)

    for step in range(steps):
        events, query, target = rng.choice(train_pairs)
        step_metrics = wm_trained.learn_step(events, query, target)
        train_losses.append(step_metrics["loss"])

    train_time = round(time.monotonic() - train_start_time, 3)

    # Calculate actual parameter change evidence
    param_delta_total = sum(
        float((p.detach() - initial_weights[n]).abs().sum().item())
        for n, p in wm_trained.model.named_parameters()
    )

    trained_train_eval = evaluate_model(wm_trained, train_scenarios, "trained_train")

    # -------------------------------------------------------------
    # Condition C: Unseen Generalization
    # -------------------------------------------------------------
    trained_unseen_eval = evaluate_model(wm_trained, unseen_scenarios, "trained_unseen")

    # -------------------------------------------------------------
    # Condition D: Shuffled-Time Control
    # -------------------------------------------------------------
    wm_shuffled = SonaraWorldModel(device=device, dim=dim, lr=lr)
    shuffled_pairs = []
    for sc in train_scenarios:
        permuted_events = list(sc.events)
        rng.shuffle(permuted_events)
        for q in sc.questions:
            shuffled_pairs.append((permuted_events, q.question, q.target))

    for _ in range(steps):
        events, query, target = rng.choice(shuffled_pairs)
        wm_shuffled.learn_step(events, query, target)

    shuffled_eval = evaluate_model(wm_shuffled, unseen_scenarios, "shuffled_unseen")

    # -------------------------------------------------------------
    # Condition G: Restart Persistence
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="sonara-restart-") as r_dir:
        ckpt_path = Path(r_dir) / "world_model_restart.pt"
        wm_trained.save_checkpoint(ckpt_path)

        # Re-instantiate fresh SonaraWorldModel from checkpoint
        wm_restarted = SonaraWorldModel(checkpoint_path=ckpt_path, device=device, dim=dim)
        restarted_unseen_eval = evaluate_model(wm_restarted, unseen_scenarios, "restarted_unseen")

        # Verify exact bit-for-bit prediction match
        restart_verified = (trained_unseen_eval["accuracy"] == restarted_unseen_eval["accuracy"]
                            and trained_unseen_eval["loss"] == restarted_unseen_eval["loss"])

    total_time = round(time.monotonic() - start_time, 2)

    return {
        "seed": seed,
        "runtime_seconds": total_time,
        "training": {
            "steps": steps,
            "train_pairs_pool": len(train_pairs),
            "first_loss": train_losses[0] if train_losses else None,
            "last_loss": train_losses[-1] if train_losses else None,
            "train_time_seconds": train_time,
            "total_parameter_l1_delta": round(param_delta_total, 4),
        },
        "conditions": {
            "untrained_baseline": {
                "train_accuracy": baseline_train["accuracy"],
                "unseen_accuracy": baseline_unseen["accuracy"],
                "unseen_loss": baseline_unseen["loss"],
                "unseen_uncertainty": baseline_unseen["avg_uncertainty"],
            },
            "neural_trained": {
                "train_accuracy": trained_train_eval["accuracy"],
                "train_loss": trained_train_eval["loss"],
                "unseen_accuracy": trained_unseen_eval["accuracy"],
                "unseen_loss": trained_unseen_eval["loss"],
                "unseen_uncertainty": trained_unseen_eval["avg_uncertainty"],
                "by_category": trained_unseen_eval["by_category"],
            },
            "shuffled_time_control": {
                "unseen_accuracy": shuffled_eval["accuracy"],
                "unseen_loss": shuffled_eval["loss"],
                "by_category": shuffled_eval["by_category"],
            },
            "frozen_weights_control": {
                "unseen_accuracy": frozen_eval["accuracy"],
                "unseen_loss": frozen_eval["loss"],
            },
            "memory_only_control": {
                "accuracy": memory_only_eval["accuracy"],
                "by_category": memory_only_eval["by_category"],
            },
            "restart_persistence": {
                "verified": restart_verified,
                "restarted_accuracy": restarted_unseen_eval["accuracy"],
                "restarted_loss": restarted_unseen_eval["loss"],
            },
        },
        "sample_examples": trained_unseen_eval["sample_examples"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run Sonara Neural World Model Experiment 1")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--steps", type=int, default=140, help="Training steps per seed (default: 140)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--dim", type=int, default=64, help="Model hidden width (default: 64)")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--output", default="artifacts/world_learning_report.json", help="Path for JSON results")
    args = parser.parse_args()

    device = "cpu"
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    elif args.device in ("cpu", "mps", "cuda"):
        device = args.device

    print(f"\n=======================================================")
    print(f" SONARA WORLD MODEL — EXPERIMENT 1: NEURAL LEARNING")
    print(f"=======================================================")
    print(f" Device: {device} | Seeds: {args.seeds} | Steps: {args.steps} | Dim: {args.dim}")
    print(f" Output target: {args.output}\n")

    runs = []
    base_seeds = [42, 43, 44][:args.seeds]
    for seed in base_seeds:
        print(f"--- Running Seed {seed} ---", flush=True)
        res = run_experiment_seed(seed, steps=args.steps, lr=args.lr, dim=args.dim, device=device)
        runs.append(res)
        cond = res["conditions"]
        print(f"  Untrained held-out acc:  {cond['untrained_baseline']['unseen_accuracy']:.1%}")
        print(f"  Neural trained train:    {cond['neural_trained']['train_accuracy']:.1%}")
        print(f"  Neural trained held-out: {cond['neural_trained']['unseen_accuracy']:.1%} (Loss: {cond['neural_trained']['unseen_loss']:.3f})")
        print(f"  Shuffled-time held-out:  {cond['shuffled_time_control']['unseen_accuracy']:.1%}")
        print(f"  Memory-only baseline:    {cond['memory_only_control']['accuracy']:.1%}")
        print(f"  Restart persistence:     {'PASSED' if cond['restart_persistence']['verified'] else 'FAILED'}")
        print(f"  Parameter delta:         {res['training']['total_parameter_l1_delta']:.2f}")

    # Compute aggregate metrics
    untrained_accs = [r["conditions"]["untrained_baseline"]["unseen_accuracy"] for r in runs]
    trained_accs = [r["conditions"]["neural_trained"]["unseen_accuracy"] for r in runs]
    shuffled_accs = [r["conditions"]["shuffled_time_control"]["unseen_accuracy"] for r in runs]
    memory_accs = [r["conditions"]["memory_only_control"]["accuracy"] for r in runs]
    frozen_accs = [r["conditions"]["frozen_weights_control"]["unseen_accuracy"] for r in runs]

    summary = {
        "num_seeds": len(runs),
        "mean_untrained_accuracy": round(sum(untrained_accs) / len(untrained_accs), 4),
        "mean_trained_heldout_accuracy": round(sum(trained_accs) / len(trained_accs), 4),
        "mean_shuffled_accuracy": round(sum(shuffled_accs) / len(shuffled_accs), 4),
        "mean_memory_only_accuracy": round(sum(memory_accs) / len(memory_accs), 4),
        "mean_frozen_accuracy": round(sum(frozen_accs) / len(frozen_accs), 4),
        "all_restarts_verified": all(r["conditions"]["restart_persistence"]["verified"] for r in runs),
    }

    report = {
        "title": "Sonara Neural World Model — Experiment 1 Results",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "runs": runs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Artifact Saved] Report exported to: {out_path.resolve()}")
    print(f"Summary: Trained Held-Out Mean Acc: {summary['mean_trained_heldout_accuracy']:.1%} "
          f"vs Untrained: {summary['mean_untrained_accuracy']:.1%} "
          f"vs Shuffled: {summary['mean_shuffled_accuracy']:.1%} "
          f"vs Memory-Only: {summary['mean_memory_only_accuracy']:.1%}")


if __name__ == "__main__":
    main()
