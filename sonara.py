"""sonara.py — Sonara: a linear scorer that learns recall from verified feedback.

Each (question, memory) pair gets the baseline's inputs (cosine, age, source), day-
clock inputs, and three experience inputs. The clock is read when Sonara runs: "now"
is the question's ask time to the second, and every memory is a point on the same
timeline. Experience inputs come only from train-split feedback given before the
question was asked, so eval labels never become inputs. See SONARA_PLAN.md.

    python3 sonara.py curve [--split YYYY-MM-DD ...]
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
import statistics
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from sonara_memory import Memory
from sonara_recall import K, Store, baseline_rank, baseline_scores, hit, rank_by, tune

SOURCES = ("user_note", "user_question", "qwen_answer", "qwen_follow_up")
CLOCK = ["log_gap_seconds", "same_time_of_day", "same_day_of_week", "same_time_of_year",
         "log_sessions_between", "newer_similar"]
FEATURES = (["cosine"] + CLOCK + [f"source:{s}" for s in SOURCES]
            + ["log_useful", "log_shown_not_useful", "neighbor_vote"])
COS, SRC, USEFUL, SHOWN, VOTE = 0, 1 + len(CLOCK), 11, 12, 13
SESSION_GAP = 3600.0  # a quiet hour or more starts a new session
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SEEDS = 5
EPOCHS = 50
RULE_GRID = (0.1, 0.2, 0.4, 0.8)


def clock_phases(epoch: float) -> Tuple[float, float, float]:
    """Local time of day, day of week and time of year, each as a fraction of its cycle."""
    t = dt.datetime.fromtimestamp(epoch)
    day = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6) / 86400
    return day, (t.weekday() + day) / 7, (t.timetuple().tm_yday - 1 + day) / 365.25


def clock_inputs(mems: Sequence[Memory], vecs: torch.Tensor, now: float) -> torch.Tensor:
    """(N, 6) day-clock inputs for memories, relative to now on one continuous timeline."""
    times = torch.tensor([m.timestamp_epoch for m in mems], dtype=torch.float64)
    x = torch.zeros(len(mems), len(CLOCK))
    x[:, 0] = torch.log1p(now - times).float()
    now_phase = torch.tensor(clock_phases(now))
    phases = torch.tensor([clock_phases(m.timestamp_epoch) for m in mems])
    x[:, 1:4] = torch.cos(2 * math.pi * (phases - now_phase))  # 1 = same point in the cycle
    ordered = sorted(times.tolist()) + [now]
    starts = [b for a, b in zip(ordered, ordered[1:]) if b - a >= SESSION_GAP]
    x[:, 4] = torch.log1p(torch.tensor([len(starts) - bisect.bisect_right(starts, t) for t in times.tolist()],
                                       dtype=torch.float32))
    newer = times[None, :] > times[:, None]  # [i, j]: memory j was written after memory i
    sims = (vecs @ vecs.T).masked_fill(~newer, -1.0)
    x[:, 5] = sims.max(dim=1).values.clamp(min=0.0)
    return x


def features(store: Store, question: str, asked_epoch: float,
             history: Sequence[Dict]) -> Tuple[List[Memory], torch.Tensor]:
    mems = store.before(asked_epoch)
    x = torch.zeros(len(mems), len(FEATURES))
    if not mems:
        return mems, x
    index = {m.id: i for i, m in enumerate(mems)}
    vecs = F.normalize(store.encoder.vectors([m.content for m in mems]), dim=1)
    qvec = F.normalize(store.encoder.vectors([question])[0], dim=0)
    x[:, COS] = vecs @ qvec
    x[:, 1:SRC] = clock_inputs(mems, vecs, asked_epoch)
    for i, m in enumerate(mems):
        if m.source in SOURCES:
            x[i, SRC + SOURCES.index(m.source)] = 1.0
    past = [q for q in history if q["asked_epoch"] < asked_epoch]
    if past:
        sims = F.normalize(store.encoder.vectors([q["question"] for q in past]), dim=1) @ qvec
        for q, sim in zip(past, sims.tolist()):
            for mid in q["correct_ids"]:
                if mid in index:
                    x[index[mid], USEFUL] += 1
                    x[index[mid], VOTE] = max(x[index[mid], VOTE].item(), sim)
            for mid in q.get("candidate_ids", []):
                if mid in index and mid not in q["correct_ids"]:
                    x[index[mid], SHOWN] += 1
        x[:, USEFUL:VOTE] = torch.log1p(x[:, USEFUL:VOTE])
    return mems, x


def train(store: Store, labeled: Sequence[Dict], history: Sequence[Dict], seed: int) -> torch.nn.Linear:
    """Softmax over each question's labeling pool; cross-entropy on its useful memories."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(len(FEATURES), 1)
    with torch.no_grad():  # start as cosine-only ranking; experience moves it from there
        model.weight.zero_()
        model.weight[0, COS] = 1.0
        model.bias.zero_()
    examples = []
    for q in labeled:
        mems, x = features(store, q["question"], q["asked_epoch"], history)
        pool = set(q.get("candidate_ids", [])) | set(q["correct_ids"])
        rows = [i for i, m in enumerate(mems) if m.id in pool]
        positive = torch.tensor([mems[i].id in q["correct_ids"] for i in rows], dtype=torch.bool)
        if positive.any() and not positive.all():
            examples.append((x[rows], positive))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(EPOCHS if examples else 0):
        for i in torch.randperm(len(examples)).tolist():
            x, positive = examples[i]
            loss = -torch.logsumexp(torch.log_softmax(model(x).squeeze(1), 0)[positive], 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def sonara_rank(store: Store, model: torch.nn.Linear, question: str, asked_epoch: float,
                history: Sequence[Dict]) -> List[Memory]:
    mems, x = features(store, question, asked_epoch, history)
    with torch.no_grad():
        return rank_by(mems, model(x).squeeze(1))


def rule_rank(store: Store, params: Dict, bonus: float, question: str, asked_epoch: float,
              history: Sequence[Dict]) -> List[Memory]:
    """Reference, no learning: tuned baseline + bonus * neighbor vote."""
    mems, scores = baseline_scores(store, question, asked_epoch, params)
    _, x = features(store, question, asked_epoch, history)
    return rank_by(mems, scores + bonus * x[:, VOTE])


def tune_rule(store: Store, params: Dict, labeled: Sequence[Dict], history: Sequence[Dict]) -> float:
    def recall(b):
        return sum(hit(rule_rank(store, params, b, q["question"], q["asked_epoch"], history), q["correct_ids"])
                   for q in labeled)
    bonus, best = 0.0, recall(0.0)
    for b in RULE_GRID:
        if (r := recall(b)) > best:
            bonus, best = b, r
    return bonus


def curve(store: Store, split_epoch=None):
    """Eval recall@5 for baseline, feedback rule and Sonara as train experience grows."""
    qs = sorted(store.questions(), key=lambda q: q["asked_epoch"])
    before = lambda q: split_epoch is None or q["asked_epoch"] < split_epoch
    train_all = [q for q in qs if q["split"] == "train" and before(q)]
    test = [q for q in qs if q["split"] == "eval" and q["correct_ids"] and not before(q)] if split_epoch \
        else [q for q in qs if q["split"] == "eval" and q["correct_ids"]]

    def recall(rank: Callable[[Dict], List[Memory]]) -> float:
        return 100 * statistics.mean(hit(rank(q), q["correct_ids"]) for q in test) if test else float("nan")

    rows = []
    for f in FRACTIONS:
        history = train_all[:math.ceil(f * len(train_all))]
        labeled = [q for q in history if q["correct_ids"]]
        params = tune(store, labeled)
        bonus = tune_rule(store, params, labeled, history)
        models = [train(store, labeled, history, s) for s in range(SEEDS)]
        sonara = [recall(lambda q, m=m: sonara_rank(store, m, q["question"], q["asked_epoch"], history))
                  for m in models]
        # Control: same learner, empty history, so the three experience inputs are always 0.
        no_exp = [recall(lambda q, m=train(store, labeled, [], s): sonara_rank(store, m, q["question"],
                                                                              q["asked_epoch"], []))
                  for s in range(SEEDS)]
        rows.append({
            "fraction": f, "n_train": len(labeled), "bonus": bonus, "model": models[0], "history": history,
            "baseline": recall(lambda q: baseline_rank(store, q["question"], q["asked_epoch"], params)),
            "rule": recall(lambda q: rule_rank(store, params, bonus, q["question"], q["asked_epoch"], history)),
            "sonara": statistics.mean(sonara), "sonara_sd": statistics.pstdev(sonara),
            "no_exp": statistics.mean(no_exp),
        })
    return test, rows


def report(store: Store, split_epoch=None) -> None:
    test, rows = curve(store, split_epoch)
    print(f"\n{len(test)} eval questions" + (f" asked after {dt.datetime.fromtimestamp(split_epoch):%Y-%m-%d}"
                                             if split_epoch else ""))
    print(" train qs | baseline | rule (bonus) | no-exp | sonara (sd)  | vs baseline | vs no-exp")
    for r in rows:
        print(f" {r['n_train']:8} | {r['baseline']:7.1f}% | {r['rule']:5.1f}% ({r['bonus']:.1f}) | "
              f"{r['no_exp']:5.1f}% | {r['sonara']:5.1f}% ({r['sonara_sd']:.1f}) | "
              f"{r['sonara'] - r['baseline']:+11.1f} | {r['sonara'] - r['no_exp']:+.1f}")
    last = rows[-1]
    print("weights (seed 0, all train):",
          ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEATURES, last["model"].weight[0].tolist())))
    misses = 0
    for q in test:
        if misses == 20:
            break
        ranked = sonara_rank(store, last["model"], q["question"], q["asked_epoch"], last["history"])
        if not hit(ranked, q["correct_ids"]):
            misses += 1
            correct = [store.memory.memories[i].content[:70] for i in q["correct_ids"] if i in store.memory.memories]
            print(f"MISS: {q['question']}\n  correct: {correct}\n  top {K}: {[m.content[:50] for m in ranked[:K]]}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["curve"])
    parser.add_argument("--dir", default="sonara_memories")
    parser.add_argument("--split", action="append", help="time split date YYYY-MM-DD; repeat for 3 splits")
    args = parser.parse_args()
    store = Store(args.dir)
    for split in args.split or [None]:
        report(store, dt.datetime.fromisoformat(split).timestamp() if split else None)


if __name__ == "__main__":
    main()
