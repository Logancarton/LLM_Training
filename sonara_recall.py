"""sonara_recall.py — Experience log, fixed recall baseline, and recall@5.

Experience comes from real use: you ask Qwen a question, the baseline recalls
memories, Qwen answers from them, and you mark which candidates were useful.
Every note, question and answer is stored once with its Qwen vector. Verified
questions are assigned to train or eval when written (1 in 3 eval, by id hash),
so the evaluation set is frozen and never trained on.

    python3 sonara_recall.py chat                     # ask, label, log
    python3 sonara_recall.py score [--split YYYY-MM-DD]
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import itertools
import json
import math
import random
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sonara_memory import Memory, SonaraMemorySystem, TokenStore
from sonara_text_encoder import PretrainedTextEncoder

CHAT_MODEL = "qwen2.5:1.5b-instruct"
CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
K = 5
# Recency weight, half-life in days, "newer" (weight on newer_similar, default 0), and
# "bias:<source>" (default 0) per memory source, relative to notes (only differences between
# source biases change a ranking, so notes stay at 0). Wide enough to match what Sonara learns.
DEFAULT_PARAMS = {"weight": 0.1, "half_life": 7.0}
GRID = {"weight": (0.0, 0.05, 0.1, 0.2, 0.4), "half_life": (1.0, 7.0, 30.0),
        "newer": (-1.6, -0.8, -0.4, -0.2, -0.1, 0.0)}
BIAS_GRID = (-1.6, -0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.4)
REFERENCE_SOURCE = "user_note"
CHUNK = 20000  # settings combinations scored at once
TUNE_EVERY = 10  # labeled train questions between baseline re-tunes
POOL_COSINE, POOL_NEWEST, POOL_NEIGHBORS = 50, 10, 5  # the bounded candidate set every method ranks from
NOTE_SOURCES = ("user_note", "user_observation")  # written by the user, not restated by Qwen


def split_for(question_id: str) -> str:
    return "eval" if int(hashlib.sha1(question_id.encode()).hexdigest(), 16) % 3 == 0 else "train"


class Store:
    """Memories plus their vectors (embedded once, cached) and verified questions."""

    def __init__(self, memory_dir: str = "sonara_memories", encoder=None):
        self.memory = SonaraMemorySystem(memory_dir)
        self.dir = Path(memory_dir)
        self.vectors_path = self.dir / "vectors.pt"
        self.questions_path = self.dir / "questions.jsonl"
        self.newer_path = self.dir / "newer_similar.pt"
        self._tokens = TokenStore(self.dir / "tokens.json")
        self._newer = torch.load(self.newer_path) if self.newer_path.exists() else {}
        self.encoder = encoder or PretrainedTextEncoder(cache_path=self.vectors_path)
        if self.encoder.ensure(m.content for m in self.memory.memories.values()):
            self.encoder.save(self.vectors_path)

    def add(self, text: str, source: str) -> Memory:
        mem = self.memory.remember(text, source=source)
        self.encoder.ensure([mem.content])
        self.encoder.save(self.vectors_path)
        return mem

    def add_question(self, question_id: str, question: str, asked_epoch: float,
                     correct_ids: List[str], candidate_ids: List[str],
                     answer_ok: Optional[bool], rankings: Optional[Dict[str, List[str]]] = None,
                     proposed_by: Optional[Dict[str, List[str]]] = None) -> Dict:
        row = {"id": question_id, "split": split_for(question_id), "question": question,
               "asked_epoch": asked_epoch, "correct_ids": correct_ids,
               "candidate_ids": candidate_ids, "answer_ok": answer_ok}
        if rankings:
            row["rankings"] = rankings
        if proposed_by:
            row["proposed_by"] = proposed_by  # which system put each candidate in the labeling pool
        with self.questions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def questions(self) -> List[Dict]:
        if not self.questions_path.exists():
            return []
        return [json.loads(l) for l in self.questions_path.open(encoding="utf-8") if l.strip()]

    def before(self, epoch: float) -> List[Memory]:
        return [m for m in self.memory.memories.values()
                if m.timestamp_epoch < epoch and m.source != "sonara_reflection"]

    def tokens(self) -> TokenStore:
        """Every token with every timestamp it was used at, up to date with the memory log."""
        self._tokens.update(self.before(math.inf))
        return self._tokens

    def newer_similar(self, mems: Sequence[Memory], now: float) -> torch.Tensor:
        """For each memory, its highest cosine (at least 0) to a note written after it and before now.
        Only notes count: Qwen restates useful facts, so a newer Qwen memory doesn't mean replaced."""
        self._update_newer()
        out = []
        for m in mems:
            recs = self._newer["records"].get(m.id, [])
            i = bisect.bisect_left([t for t, _ in recs], now)
            out.append(recs[i - 1][1] if i else 0.0)
        return torch.tensor(out)

    def _update_newer(self) -> None:
        """Each memory keeps a running-max record of (time, cosine) against newer memories,
        updated once per new memory (linear) and saved, replacing an N x N comparison."""
        mems = sorted(self.before(math.inf), key=lambda m: m.timestamp_epoch)
        order = [[m.id, m.timestamp_epoch] for m in mems]
        done = len(self._newer.get("order", []))
        if order[:done] != self._newer.get("order") or self._newer.get("sources") != list(NOTE_SOURCES):
            # an older memory appeared, a time changed, or the rule changed: rebuild
            self._newer, done = {"order": [], "records": {}, "sources": list(NOTE_SOURCES)}, 0
        if done == len(mems):
            return
        vecs = torch.nn.functional.normalize(self.encoder.vectors([m.content for m in mems]), dim=1)
        records = self._newer["records"]
        for j in range(done, len(mems)):
            if mems[j].source not in NOTE_SOURCES:
                continue
            t = mems[j].timestamp_epoch
            for i, sim in enumerate((vecs[:j] @ vecs[j]).tolist()):
                if mems[i].timestamp_epoch < t:
                    r = records.setdefault(mems[i].id, [])
                    if sim > (r[-1][1] if r else 0.0):
                        r.append((t, sim))
        self._newer["order"] = order
        torch.save(self._newer, self.newer_path)

    def similarities(self, question: str, mems: Sequence[Memory]) -> torch.Tensor:
        return self.text_similarities(question, [m.content for m in mems])

    def text_similarities(self, question: str, texts: Sequence[str]) -> torch.Tensor:
        vecs = torch.nn.functional.normalize(self.encoder.vectors(list(texts)), dim=1)
        q = torch.nn.functional.normalize(self.encoder.vectors([question])[0], dim=0)
        return vecs @ q


def candidates(store: Store, question: str, asked_epoch: float) -> List[Memory]:
    """The bounded set every method ranks from, written before the question: the top POOL_COSINE
    memories by cosine, the POOL_NEWEST newest, and the memories marked useful for the
    POOL_NEIGHBORS most similar earlier train questions (so memories that only experience
    links to the question can be ranked at all; eval labels are never used)."""
    mems = store.before(asked_epoch)
    if len(mems) <= POOL_COSINE + POOL_NEWEST:
        return mems
    top = [mems[i] for i in store.similarities(question, mems).argsort(descending=True)[:POOL_COSINE].tolist()]
    newest = sorted(mems, key=lambda m: m.timestamp_epoch, reverse=True)[:POOL_NEWEST]
    past = [q for q in store.questions()
            if q["split"] == "train" and q["correct_ids"] and q["asked_epoch"] < asked_epoch]
    known = {m.id: m for m in mems}
    useful = []
    if past:
        sims = store.text_similarities(question, [q["question"] for q in past])
        for i in sims.argsort(descending=True)[:POOL_NEIGHBORS].tolist():
            useful += [known[mid] for mid in past[i]["correct_ids"] if mid in known]
    return list({m.id: m for m in top + newest + useful}.values())


def baseline_scores(store: Store, question: str, asked_epoch: float,
                    params: Dict[str, float] = DEFAULT_PARAMS) -> Tuple[List[Memory], torch.Tensor]:
    """Fixed baseline: cosine similarity + weight * recency (relative to ask time) + newer * the
    memory's similarity to a newer memory (Sonara's newer_similar input) + source bias."""
    mems = candidates(store, question, asked_epoch)
    if not mems:
        return [], torch.zeros(0)
    age_days = torch.tensor([(asked_epoch - m.timestamp_epoch) / 86400 for m in mems])
    recency = torch.exp(-math.log(2) * age_days / params["half_life"])
    bias = torch.tensor([params.get("bias:" + m.source, 0.0) for m in mems])
    newer = params.get("newer", 0.0) * store.newer_similar(mems, asked_epoch)
    return mems, store.similarities(question, mems) + params["weight"] * recency + newer + bias


def rank_by(mems: Sequence[Memory], scores: torch.Tensor) -> List[Memory]:
    return [mems[i] for i in scores.argsort(descending=True).tolist()]


def baseline_rank(store: Store, question: str, asked_epoch: float,
                  params: Dict[str, float] = DEFAULT_PARAMS) -> List[Memory]:
    return rank_by(*baseline_scores(store, question, asked_epoch, params))


def hit(ranked: Sequence[Memory], correct_ids: Sequence[str], k: int = K) -> bool:
    return any(m.id in correct_ids for m in ranked[:k])


def tune(store: Store, train: List[Dict]) -> Dict[str, float]:
    """Try every combination of the baseline's settings on train questions only and keep the
    one with the most top-1 hits (the target), then the most recall@5 hits, then the widest
    margin between correct and other memories (as Sonara's cross-entropy rewards); exact ties go
    to the settings closest to the defaults. Each question's cosines, ages and sources are computed once, then all
    combinations are scored in chunks."""
    params = dict(DEFAULT_PARAMS)
    if not train:
        return params
    free = sorted({m.source for m in store.before(math.inf)} - {REFERENCE_SOURCE})
    keys = list(GRID) + ["bias:" + s for s in free]
    combos = torch.tensor(list(itertools.product(*GRID.values(), *[BIAS_GRID] * len(free))), dtype=torch.float64)
    biases = torch.cat([combos[:, 3:], torch.zeros(len(combos), 1, dtype=torch.float64)], dim=1)  # last: notes
    top1, top5, margin = (torch.zeros(len(combos), dtype=torch.float64) for _ in range(3))
    for q in train:
        mems = candidates(store, q["question"], q["asked_epoch"])
        correct = torch.tensor([m.id in q["correct_ids"] for m in mems], dtype=torch.bool)
        if not correct.any():
            continue
        age_days = torch.tensor([(q["asked_epoch"] - m.timestamp_epoch) / 86400 for m in mems])
        src = torch.tensor([free.index(m.source) if m.source in free else len(free) for m in mems])
        cos = store.similarities(q["question"], mems)
        newer_sim = store.newer_similar(mems, q["asked_epoch"])
        for i in range(0, len(combos), CHUNK):
            c = combos[i:i + CHUNK]
            recency = torch.exp(-math.log(2) * age_days[None, :] / c[:, 1:2])
            scores = cos + c[:, 0:1] * recency + c[:, 2:3] * newer_sim[None, :] + biases[i:i + CHUNK][:, src]
            best_correct = scores[:, correct].max(dim=1).values
            above = (scores > best_correct[:, None]).sum(dim=1)
            top1[i:i + CHUNK] += (above == 0).double()
            top5[i:i + CHUNK] += (above < K).double()
            if (~correct).any():  # capped so no single question dominates
                gap = best_correct - scores[:, ~correct].max(dim=1).values
                margin[i:i + CHUNK] += gap.clamp(-1.0, 1.0)
    keep = torch.ones(len(combos), dtype=torch.bool)
    for objective in (top1, top5, margin):
        keep &= objective == objective[keep].max()
    default = torch.tensor([params.get(k, 0.0) for k in keys], dtype=torch.float64)
    best = int(torch.where(keep, (combos - default).abs().sum(dim=1), math.inf).argmin())
    return {k: v for k, v in zip(keys, combos[best].tolist()) if k in DEFAULT_PARAMS or v != 0.0}


def live_params(store: Store) -> Dict[str, float]:
    """The baseline's settings as `sonara.py curve` has them now: tuned on the labeled train
    questions so far, re-tuned every TUNE_EVERY of them."""
    labeled = sorted((q for q in store.questions() if q["split"] == "train" and q["correct_ids"]),
                     key=lambda q: q["asked_epoch"])
    return tune(store, labeled[:len(labeled) // TUNE_EVERY * TUNE_EVERY])


def score(store: Store, split_epoch: Optional[float] = None):
    """Recall@5 of the tuned baseline on eval questions asked at or after the split."""
    labeled = [q for q in store.questions() if q["correct_ids"]]
    train = [q for q in labeled if q["split"] == "train" and (split_epoch is None or q["asked_epoch"] < split_epoch)]
    test = [q for q in labeled if q["split"] == "eval" and (split_epoch is None or q["asked_epoch"] >= split_epoch)]
    params = tune(store, train)
    results = [(q, baseline_rank(store, q["question"], q["asked_epoch"], params)) for q in test]
    return params, len(train), results


def label_pool(store: Store, asked_epoch: float,
               rankings: Dict[str, Sequence[Memory]]) -> Tuple[List[Memory], Dict[str, List[str]]]:
    """Each system's top K plus the K newest memories, shuffled and shown without scores, so a
    memory only one system finds can still be labeled. Also returns who proposed each one."""
    newest = sorted(store.before(asked_epoch), key=lambda m: m.timestamp_epoch, reverse=True)[:K]
    proposed: Dict[str, List[str]] = {}
    pool: Dict[str, Memory] = {}
    for name, ranked in list(rankings.items()) + [("newest", newest)]:
        for m in ranked[:K]:
            pool[m.id] = m
            proposed.setdefault(m.id, []).append(name)
    pool_list = list(pool.values())
    random.shuffle(pool_list)
    return pool_list, proposed


FOLLOW_UP = ("About your answer: 1) What makes this true? 2) What would make it not true? "
             "3) What are the most relevant variables, and how do they relate to the question and the answer?")


def _chat(messages: List[Dict]) -> str:
    payload = json.dumps({"model": CHAT_MODEL, "messages": messages, "stream": False}).encode()
    request = urllib.request.Request(CHAT_ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        return json.load(response)["message"]["content"].strip()


def ask_qwen(question: str, mems: Sequence[Memory]) -> Tuple[str, str]:
    """Qwen's answer, then its reply to the fixed follow-up questions about that answer."""
    context = "\n".join(f"- ({m.timestamp_human}) {m.content}" for m in mems) or "(none)"
    messages = [
        {"role": "system", "content": "Answer the question. Use these memories when relevant; "
                                      "if they do not contain the answer, say so.\n" + context},
        {"role": "user", "content": question},
    ]
    answer = _chat(messages)
    follow_up = _chat(messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": FOLLOW_UP}])
    return answer, follow_up


def chat(store: Store) -> None:
    """The tuned baseline answers; Sonara ranks alongside it in shadow, and both full rankings are
    logged. Sonara learns live only from train-split questions (load_learner skips eval)."""
    from sonara import load_learner  # sonara.py owns learning; imported here to avoid a cycle
    sonara, params = load_learner(store), live_params(store)
    print("Ask a question, or '/note <text>' to store a note. Empty line quits.")
    while True:
        line = input("\nyou> ").strip()
        if not line:
            return
        if line.startswith("/note "):
            mem = store.add(line[6:], source="user_note")
            print(f"[noted {mem.id}]")
            continue
        asked = time.time()
        baseline = baseline_rank(store, line, asked, params)
        history = [q for q in store.questions() if q["split"] == "train"]
        shadow = sonara.rank(store, {"question": line, "asked_epoch": asked}, history)
        answer, follow_up = ask_qwen(line, baseline[:K])
        print(f"\nqwen> {answer}\n\nqwen (follow-up)> {follow_up}\n")
        pool, proposed = label_pool(store, asked, {"baseline": baseline, "sonara": shadow})
        for i, m in enumerate(pool, 1):
            print(f"  {i:2}. ({m.timestamp_human}) {m.content[:160]}")
        picks = input("Memories useful for this question, true or not (numbers or ids, Enter = none): ").split()
        correct = [pool[int(p) - 1].id if p.isdigit() else p for p in picks
                   if (p.isdigit() and 1 <= int(p) <= len(pool)) or p in store.memory.memories]
        ok = input("Answer correct? [y/n/Enter=skip]: ").strip().lower()
        q_mem = store.add(line, source="user_question")
        store.add(answer, source="qwen_answer")
        store.add(follow_up, source="qwen_follow_up")
        row = store.add_question(q_mem.id, line, asked, correct, [m.id for m in pool],
                                 {"y": True, "n": False}.get(ok),
                                 {"baseline": [m.id for m in baseline], "sonara": [m.id for m in shadow]},
                                 proposed)
        if row["split"] == "train":
            sonara = load_learner(store)
            if correct and sum(q["split"] == "train" and bool(q["correct_ids"]) for q in store.questions()) % TUNE_EVERY == 0:
                params = live_params(store)
        print(f"[logged {row['split']} question with {len(correct)} useful memories]")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["chat", "score"])
    parser.add_argument("--dir", default="sonara_memories")
    parser.add_argument("--split", help="time split date YYYY-MM-DD (local time)")
    args = parser.parse_args()
    store = Store(args.dir)
    if args.command == "chat":
        return chat(store)
    split = dt.datetime.fromisoformat(args.split).timestamp() if args.split else None
    params, n_train, results = score(store, split)
    hits = [hit(r, q["correct_ids"]) for q, r in results]
    print(f"baseline params = {params}, tuned on {n_train} train questions")
    print(f"recall@{K} = {sum(hits)}/{len(hits)}" + (f" = {100 * sum(hits) / len(hits):.1f}%" if hits else ""))
    misses = [(q, r) for (q, r), h in zip(results, hits) if not h]
    for q, r in misses[:20]:
        correct = [store.memory.memories[i].content[:80] for i in q["correct_ids"] if i in store.memory.memories]
        print(f"\nMISS: {q['question']}\n  correct: {correct}\n  top {K}: {[m.content[:60] for m in r[:K]]}")


if __name__ == "__main__":
    main()
