"""Frozen-world, pre/post training evaluation of Sonara's existing sparse selector.

Synthetic experiment. Verified labels never enter inference-time retrieval.
Report ALL held-out cases and the subset where frozen full context is correct.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiment_world_learning import build_experiment_datasets
from sonara_world_model import SonaraWorldModel, tokenize, STOPWORDS, STRUCTURAL_WORDS


def expand(events, distractors):
    """Insert irrelevant original experiences; preserve evidence chronology."""
    if not distractors:
        return list(events), list(range(len(events)))
    noise = [
        f"Visitor_{i} examined artifact_{i} beside storage_{i}."
        for i in range(distractors)
    ]
    return [events[0], *noise, *events[1:]], [
        0, *[i + distractors for i in range(1, len(events))]
    ]


def critical_indices(question, positions):
    """Synthetic audit-only causal labels; never sent to the ranker."""
    if question.category == "previous_state":
        return [positions[0]]
    if question.category in ("multi_object", "object_identity"):
        return [positions[2]] if len(positions) > 2 else []
    if question.category in ("current_state", "correction", "causal_intervention"):
        return [positions[-1]]
    return []


def fixed_indices(events, question, budget):
    cues = set(tokenize(question)) - STOPWORDS - STRUCTURAL_WORDS
    ranks = sorted(range(len(events)), key=lambda i: (
        -len(cues.intersection(tokenize(events[i]))), i
    ))
    return sorted(ranks[:budget])


def infer(owner, events, question, method, budget, seed):
    t0 = time.process_time()
    if method == "full":
        indices = list(range(len(events)))
    elif method == "fixed":
        indices = fixed_indices(events, question, budget)
    elif method == "recency":
        indices = list(range(max(0, len(events) - budget), len(events)))
    elif method == "random":
        indices = sorted(random.Random(seed).sample(
            range(len(events)), min(budget, len(events))
        ))
    elif method == "learned":
        indices = owner._rank_learned_events(events, question, budget)
    elif method == "recovery":
        owner.clear_session()
        for event in events:
            owner.observe(event)
        owner._learned_sparse_activation = True
        owner._max_active_events = budget
        owner._sparse_recovery_limit = min(32, max(budget + 1, 2 * budget))
        owner._sparse_min_confidence = 0.35
        pred = owner.predict(question)
        trace = dict(owner.last_sparse_trace)
        return pred, trace.get("indices", list(range(len(events)))), (
            time.process_time() - t0
        ) * 1000, bool(trace.get("recovered")), bool(trace.get("unresolved"))
    else:
        raise ValueError(method)
    pred = owner.predict(question, events=[events[i] for i in indices])
    return pred, indices, (time.process_time() - t0) * 1000, False, False


def train_world(owner, training, seed, steps):
    # Identical sampling strategy to the existing world benchmark.
    pairs = [(sc.events, q.question, q.target)
             for sc in training for q in sc.questions]
    rng = random.Random(seed)
    for _ in range(steps):
        owner.learn_step(*rng.choice(pairs))


def teacher_accuracy(owner, scenarios):
    hits = count = 0
    for sc in scenarios:
        for q in sc.questions:
            count += 1
            hits += owner.predict(q.question, events=sc.events).answer == q.target
    return hits / max(count, 1), count


def heldout_cases(scenarios, count):
    eligible = {"current_state", "previous_state", "multi_object",
                "object_identity", "correction", "causal_intervention"}
    cases = []
    for sc in scenarios[:count]:
        for q in (x for x in sc.questions if x.category in eligible):
            for noise in (0, 16):
                events, original_positions = expand(sc.events, noise)
                cases.append({
                    "id": f"{sc.id}_{q.category}_{noise}",
                    "events": events, "question": q.question,
                    "target": q.target, "category": q.category,
                    "distractors": noise,
                    "required": critical_indices(q, original_positions),
                })
    return cases


def evaluate(owner, cases, budget, seed, phase, known_full=None):
    modes = ("full", "fixed", "recency", "random", "learned", "recovery")
    results = []
    full_map = {}
    for case_no, c in enumerate(cases):
        predictions = {}
        for mode_no, mode in enumerate(modes):
            pred, indices, cpu, recovered, unresolved = infer(
                owner, c["events"], c["question"], mode, budget,
                seed * 100003 + case_no * 991 + mode_no
            )
            predictions[mode] = pred.answer
            if mode == "full":
                full_map[c["id"]] = pred.answer
                if known_full is not None and known_full[c["id"]] != pred.answer:
                    raise AssertionError("Frozen full-context output drifted")
            results.append({
                "seed": seed, "phase": phase, "case": c["id"],
                "category": c["category"], "distractors": c["distractors"],
                "mode": mode, "answer": pred.answer, "target": c["target"],
                "correct": pred.answer == c["target"],
                "activated": len(indices), "event_count": len(c["events"]),
                "cpu_ms": round(cpu, 3),
                "critical_present": (
                    all(i in indices for i in c["required"]) if c["required"] else None
                ),
                "recovered": recovered, "unresolved": unresolved,
            })
        for row in results[-len(modes):]:
            row["full_correct"] = predictions["full"] == c["target"]
            row["agrees_with_full"] = row["answer"] == predictions["full"]
    return results, full_map


def train_ranker_from_failures(owner, training, budget, limit):
    # Iterate ONLY the disjoint training scenarios; train after world is frozen.
    # Start with long events to confront previous 12-event training blind spot.
    competent, failures, updates, long_updates, delta = 0, 0, 0, 0, 0.0
    for sc in training:
        questions = [q for q in sc.questions if q.category in {
            "current_state", "previous_state", "multi_object",
            "object_identity", "correction"
        }]
        for q in questions[:3]:
            for noise in (16, 0):
                if updates >= limit:
                    break
                events, _ = expand(sc.events, noise)
                if owner.predict(q.question, events=events).answer != q.target:
                    continue
                competent += 1
                selected = owner._rank_learned_events(events, q.question, budget)
                sparse = owner.predict(
                    q.question, events=[events[i] for i in selected]
                ).answer
                if sparse == q.target:
                    continue
                failures += 1
                result = owner.train_sparse_attention(
                    events, q.question, q.target, steps=3
                )
                if result.get("trained"):
                    updates += 1
                    long_updates += len(events) > 12
                    delta += result["parameter_delta"]
            if updates >= limit:
                break
        if updates >= limit:
            break
    return {
        "verified_competent_training_candidates": competent,
        "sparse_errors_with_verified_outcome": failures,
        "sparse_training_examples": updates,
        "long_history_examples": long_updates,
        "selector_parameter_change_l1": round(delta, 5),
        "source": "train_split_only_verified_outcomes",
    }


def aggregate(rows):
    summary = {}
    for phase in ("before", "after"):
        for mode in ("full", "fixed", "recency", "random", "learned", "recovery"):
            subset = [r for r in rows if r["phase"] == phase and r["mode"] == mode]
            competent = [r for r in subset if r["full_correct"]]
            audited = [r for r in competent if r["critical_present"] is not None]
            rate = lambda s, key: (
                round(sum(bool(r[key]) for r in s) / len(s), 4) if s else None
            )
            summary[phase + "/" + mode] = {
                "all_cases": len(subset), "full_correct_cases": len(competent),
                "accuracy_all": rate(subset, "correct"),
                "accuracy_when_full_correct": rate(competent, "correct"),
                "causal_evidence_recall_when_full_correct": rate(
                    audited, "critical_present"
                ),
                "mean_cpu_ms": round(statistics.mean(
                    r["cpu_ms"] for r in subset
                ), 3),
                "mean_activated_events": round(statistics.mean(
                    r["activated"] for r in subset
                ), 2),
                "recoveries": sum(r["recovered"] for r in subset),
                "unresolved": sum(r["unresolved"] for r in subset),
            }
    return summary


def run(seeds=3, budget=4, heldout=5, world_steps=40,
        attention_examples=8, competence_floor=0.75):
    if not (1 <= seeds <= 3 and heldout > 0 and 1 <= budget <= 16
            and 1 <= world_steps <= 160 and 1 <= attention_examples <= 24):
        raise ValueError("Invalid bounded experiment configuration")
    torch.set_num_threads(1)
    all_rows, seed_reports = [], []
    for offset in range(seeds):
        seed = offset + 42
        random.seed(seed)
        torch.manual_seed(seed)
        owner = SonaraWorldModel(device="cpu", dim=64, lr=0.003)
        train_scenes, unseen = build_experiment_datasets(seed)
        train_world(owner, train_scenes, seed, world_steps)
        teacher_rate, teacher_count = teacher_accuracy(owner, unseen)
        cases = heldout_cases(unseen, heldout)
        before, reference = evaluate(owner, cases, budget, seed, "before")
        world_weights = {
            k: v.detach().clone() for k, v in owner.model.state_dict().items()
            if not k.startswith("sparse_selector.")
        }
        for name, weight in owner.model.named_parameters():
            if not name.startswith("sparse_selector."):
                weight.requires_grad_(False)
        if teacher_rate < competence_floor:
            status = "world_below_competence_floor; attention_training_skipped"
            training = {"sparse_training_examples": 0,
                        "reason": "world_incompetent"}
        else:
            training = train_ranker_from_failures(
                owner, train_scenes, budget, attention_examples
            )
            status = ("selector_trained_on_verified_errors"
                      if training["sparse_training_examples"] else
                      "no_trainable_verified_errors")
        unchanged = all(torch.equal(x, owner.model.state_dict()[k])
                        for k, x in world_weights.items())
        if not unchanged:
            raise AssertionError("World weights changed after freezing")
        after, _ = evaluate(owner, cases, budget, seed, "after", reference)
        all_rows.extend(before)
        all_rows.extend(after)
        seed_reports.append({
            "seed": seed, "teacher_short_accuracy": round(teacher_rate, 4),
            "teacher_question_count": teacher_count, "world_steps": world_steps,
            "world_weights_frozen_unchanged": unchanged,
            "attention_training": training, "status": status
        })
        print(json.dumps(seed_reports[-1]), flush=True)
    summary = aggregate(all_rows)
    pre = summary["before/learned"]["accuracy_when_full_correct"]
    post = summary["after/learned"]["accuracy_when_full_correct"]
    return {
        "research_status": "EXPERIMENTAL: no automatic production enablement",
        "design": {
            "training_separate_from_attention": True,
            "teacher_heldout_competence_floor": competence_floor,
            "attention_training_only_on_teacher_correct_train_examples": True,
            "same_frozen_world_weights_before_after": True,
            "heldout_disjoint_from_attention_training": True,
            "long_distractor_events": 16, "attention_budget": budget,
            "full_correct_subset_denominator_explicit": True,
            "synthetic_oracle_audit_only": True,
            "action_gate": "BLOCKED: temporal branch previously 77.78% vs main 100%"
        },
        "seeds": seed_reports,
        "learned_attention_conditional_accuracy_change": (
            round(post - pre, 4) if pre is not None and post is not None else None
        ),
        "summary": summary, "cases": all_rows
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--heldout", type=int, default=5)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--world-steps", type=int, default=40)
    p.add_argument("--attention-examples", type=int, default=8)
    p.add_argument("--output", default="artifacts/learned_sparse_report.json")
    args = p.parse_args()
    result = run(args.seeds, args.budget, args.heldout,
                 args.world_steps, args.attention_examples)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "seeds": result["seeds"],
        "summary": result["summary"],
        "delta": result["learned_attention_conditional_accuracy_change"]
    }, indent=2))


if __name__ == "__main__":
    main()
