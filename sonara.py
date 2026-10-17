"""sonara.py — Sonara: a linear scorer that learns recall from verified feedback.

Each (question, memory) pair gets the baseline's inputs (cosine, age, source), day-
clock inputs, and three experience inputs. The clock is read when Sonara runs: "now"
is the question's ask time to the second, and every memory is a point on the same
timeline. Experience inputs come only from train-split feedback given before the
question was asked, so eval labels never become inputs. Sonara learns in time order
and carries its weights forward. See SONARA_PLAN.md.

    python3 sonara.py curve
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
import statistics
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from sonara_memory import Memory
from sonara_recall import DEFAULT_PARAMS, K, Store, baseline_rank, baseline_scores, hit, rank_by, tune

SOURCES = ("user_note", "user_question", "qwen_answer", "qwen_follow_up")
CLOCK = ["log_gap_seconds", "same_time_of_day", "same_day_of_week", "same_time_of_year",
         "log_sessions_between", "newer_similar", "log_life_seconds"]
FEATURES = (["cosine"] + CLOCK + [f"source:{s}" for s in SOURCES]
            + ["log_useful", "log_shown_not_useful", "neighbor_vote"])
COS, SRC = 0, 1 + len(CLOCK)
USEFUL, SHOWN, VOTE = SRC + len(SOURCES), SRC + len(SOURCES) + 1, SRC + len(SOURCES) + 2
SESSION_GAP = 3600.0  # a quiet hour or more starts a new session
PERIODS = 4
LR = 0.01
STEPS = 10  # learning steps per verified question
TUNE_EVERY = 10  # labeled train questions between baseline re-tunes
RULE_GRID = (0.1, 0.2, 0.4, 0.8)


def clock_phases(epoch: float) -> Tuple[float, float, float]:
    """Local time of day, day of week and time of year, each as a fraction of its cycle."""
    t = dt.datetime.fromtimestamp(epoch)
    day = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6) / 86400
    return day, (t.weekday() + day) / 7, (t.timetuple().tm_yday - 1 + day) / 365.25


def clock_inputs(mems: Sequence[Memory], vecs: torch.Tensor, now: float) -> torch.Tensor:
    """(N, 7) day-clock inputs for memories on one continuous timeline: relative to now, and
    the memory's own position as seconds since Sonara's first memory."""
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
    x[:, 6] = torch.log1p(times - times.min()).float()
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


class Learner:
    """Sonara's weights, learned one verified question at a time in time order and carried
    forward: never shuffled, never retrained from scratch. use_experience=False is the
    control: the same learner with its three experience inputs always 0."""

    def __init__(self, use_experience: bool = True):
        self.use_experience = use_experience
        self.model = torch.nn.Linear(len(FEATURES), 1)
        with torch.no_grad():  # born as cosine-only ranking; experience moves it from there
            self.model.weight.zero_()
            self.model.weight[0, COS] = 1.0
            self.model.bias.zero_()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)

    def _features(self, store: Store, q: Dict, history: Sequence[Dict]):
        return features(store, q["question"], q["asked_epoch"], history if self.use_experience else [])

    def rank(self, store: Store, q: Dict, history: Sequence[Dict]) -> List[Memory]:
        mems, x = self._features(store, q, history)
        with torch.no_grad():
            return rank_by(mems, self.model(x).squeeze(1))

    def learn(self, store: Store, q: Dict, history: Sequence[Dict]) -> None:
        """Softmax over the question's labeling pool; cross-entropy on its useful memories."""
        mems, x = self._features(store, q, history)
        pool = set(q.get("candidate_ids", [])) | set(q["correct_ids"])
        rows = [i for i, m in enumerate(mems) if m.id in pool]
        positive = torch.tensor([mems[i].id in q["correct_ids"] for i in rows], dtype=torch.bool)
        if not positive.any() or positive.all():
            return
        for _ in range(STEPS):
            loss = -torch.logsumexp(torch.log_softmax(self.model(x[rows]).squeeze(1), 0)[positive], 0)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


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


def run(store: Store) -> List[Dict]:
    """Walk every question in time order. Each eval question is scored by every method as
    it stood at that second; each train question is then learned from. The baseline and
    rule are re-tuned on all earlier train questions every TUNE_EVERY labeled ones."""
    sonara, control = Learner(), Learner(use_experience=False)
    history: List[Dict] = []  # train questions so far, in time order
    params, bonus, labeled = dict(DEFAULT_PARAMS), 0.0, []
    results = []
    for q in sorted(store.questions(), key=lambda q: q["asked_epoch"]):
        if q["split"] == "eval":
            if q["correct_ids"]:
                ranked = sonara.rank(store, q, history)
                results.append({
                    "q": q, "n_train": len(labeled), "sonara_top": ranked[:K],
                    "sonara": hit(ranked, q["correct_ids"]),
                    "control": hit(control.rank(store, q, history), q["correct_ids"]),
                    "baseline": hit(baseline_rank(store, q["question"], q["asked_epoch"], params), q["correct_ids"]),
                    "rule": hit(rule_rank(store, params, bonus, q["question"], q["asked_epoch"], history),
                                q["correct_ids"]),
                })
            continue
        if q["correct_ids"]:
            sonara.learn(store, q, history)
            control.learn(store, q, history)
            labeled.append(q)
            if len(labeled) % TUNE_EVERY == 0:
                params = tune(store, labeled)
                bonus = tune_rule(store, params, labeled, history)
        history.append(q)
    run.final_weights = sonara.model.weight[0].tolist()
    return results


def report(store: Store) -> None:
    results = run(store)
    print(f"\n{len(results)} eval questions, each scored as of the second it was asked, in {PERIODS} periods")
    print(" period ends | train qs learned | baseline |   rule | control | sonara | vs baseline | vs control")
    size = math.ceil(len(results) / PERIODS) if results else 1
    for i in range(0, len(results), size):
        chunk = results[i:i + size]
        pct = {k: 100 * statistics.mean(r[k] for r in chunk) for k in ("baseline", "rule", "control", "sonara")}
        end = dt.datetime.fromtimestamp(chunk[-1]["q"]["asked_epoch"])
        print(f" {end:%Y-%m-%d} | {chunk[-1]['n_train']:16} | {pct['baseline']:7.1f}% | {pct['rule']:5.1f}% | "
              f"{pct['control']:6.1f}% | {pct['sonara']:5.1f}% | {pct['sonara'] - pct['baseline']:+11.1f} | "
              f"{pct['sonara'] - pct['control']:+.1f}")
    if results:
        print("final weights:", ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEATURES, run.final_weights)))
    for r in [r for r in results if not r["sonara"]][-20:]:  # the 20 most recent misses
        q = r["q"]
        correct = [store.memory.memories[i].content[:70] for i in q["correct_ids"] if i in store.memory.memories]
        print(f"MISS: {q['question']}\n  correct: {correct}\n  top {K}: {[m.content[:50] for m in r['sonara_top']]}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["curve"])
    parser.add_argument("--dir", default="sonara_memories")
    args = parser.parse_args()
    report(Store(args.dir))


if __name__ == "__main__":
    main()
