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

import datetime as dt
import hashlib
import json
import math
import random
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sonara_memory import Memory, SonaraMemorySystem
from sonara_text_encoder import PretrainedTextEncoder

CHAT_MODEL = "qwen2.5:1.5b-instruct"
CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
K = 5
DEFAULT_PARAMS = (0.1, 7.0)  # (recency weight, half-life in days)
GRID = [(w, h) for w in (0.0, 0.05, 0.1, 0.2, 0.4) for h in (1.0, 7.0, 30.0)]


def split_for(question_id: str) -> str:
    return "eval" if int(hashlib.sha1(question_id.encode()).hexdigest(), 16) % 3 == 0 else "train"


class Store:
    """Memories plus their vectors (embedded once, cached) and verified questions."""

    def __init__(self, memory_dir: str = "sonara_memories", encoder=None):
        self.memory = SonaraMemorySystem(memory_dir)
        self.vectors_path = Path(memory_dir) / "vectors.pt"
        self.questions_path = Path(memory_dir) / "questions.jsonl"
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
                     answer_ok: Optional[bool]) -> Dict:
        row = {"id": question_id, "split": split_for(question_id), "question": question,
               "asked_epoch": asked_epoch, "correct_ids": correct_ids,
               "candidate_ids": candidate_ids, "answer_ok": answer_ok}
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

    def similarities(self, question: str, mems: Sequence[Memory]) -> torch.Tensor:
        vecs = torch.nn.functional.normalize(self.encoder.vectors([m.content for m in mems]), dim=1)
        q = torch.nn.functional.normalize(self.encoder.vectors([question])[0], dim=0)
        return vecs @ q


def baseline_rank(store: Store, question: str, asked_epoch: float,
                  params: Tuple[float, float] = DEFAULT_PARAMS) -> List[Memory]:
    """Fixed baseline: cosine similarity + weight * recency, both relative to ask time."""
    mems = store.before(asked_epoch)
    if not mems:
        return []
    weight, half_life_days = params
    age_days = torch.tensor([(asked_epoch - m.timestamp_epoch) / 86400 for m in mems])
    scores = store.similarities(question, mems) + weight * torch.exp(-math.log(2) * age_days / half_life_days)
    return [mems[i] for i in scores.argsort(descending=True).tolist()]


def hit(ranked: Sequence[Memory], correct_ids: Sequence[str], k: int = K) -> bool:
    return any(m.id in correct_ids for m in ranked[:k])


def tune(store: Store, train: List[Dict]) -> Tuple[float, float]:
    """Grid-search the baseline's two settings on train questions only."""
    if not train:
        return DEFAULT_PARAMS
    def recall(p):
        return sum(hit(baseline_rank(store, q["question"], q["asked_epoch"], p), q["correct_ids"]) for q in train)
    return max(GRID, key=recall)


def score(store: Store, split_epoch: Optional[float] = None):
    """Recall@5 of the tuned baseline on eval questions asked at or after the split."""
    labeled = [q for q in store.questions() if q["correct_ids"]]
    train = [q for q in labeled if q["split"] == "train" and (split_epoch is None or q["asked_epoch"] < split_epoch)]
    test = [q for q in labeled if q["split"] == "eval" and (split_epoch is None or q["asked_epoch"] >= split_epoch)]
    params = tune(store, train)
    results = [(q, baseline_rank(store, q["question"], q["asked_epoch"], params)) for q in test]
    return params, len(train), results


def label_pool(store: Store, question: str, asked_epoch: float) -> List[Memory]:
    """Top 10 by similarity + 5 most recent, shuffled, shown without scores."""
    mems = store.before(asked_epoch)
    if not mems:
        return []
    by_sim = [mems[i] for i in store.similarities(question, mems).argsort(descending=True)[:10].tolist()]
    recent = sorted(mems, key=lambda m: m.timestamp_epoch, reverse=True)[:5]
    pool = list({m.id: m for m in by_sim + recent}.values())
    random.shuffle(pool)
    return pool


def ask_qwen(question: str, mems: Sequence[Memory]) -> str:
    context = "\n".join(f"- ({m.timestamp_human}) {m.content}" for m in mems) or "(none)"
    messages = [
        {"role": "system", "content": "Answer the question. Use these memories when relevant; "
                                      "if they do not contain the answer, say so.\n" + context},
        {"role": "user", "content": question},
    ]
    payload = json.dumps({"model": CHAT_MODEL, "messages": messages, "stream": False}).encode()
    request = urllib.request.Request(CHAT_ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        return json.load(response)["message"]["content"].strip()


def chat(store: Store) -> None:
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
        answer = ask_qwen(line, baseline_rank(store, line, asked)[:K])
        print(f"\nqwen> {answer}\n")
        pool = label_pool(store, line, asked)
        for i, m in enumerate(pool, 1):
            print(f"  {i:2}. ({m.timestamp_human}) {m.content[:160]}")
        picks = input("Useful memories (numbers or ids, Enter = none): ").split()
        correct = [pool[int(p) - 1].id if p.isdigit() else p for p in picks
                   if (p.isdigit() and 1 <= int(p) <= len(pool)) or p in store.memory.memories]
        ok = input("Answer correct? [y/n/Enter=skip]: ").strip().lower()
        q_mem = store.add(line, source="user_question")
        store.add(answer, source="qwen_answer")
        row = store.add_question(q_mem.id, line, asked, correct, [m.id for m in pool],
                                 {"y": True, "n": False}.get(ok))
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
    print(f"baseline params (weight, half-life days) = {params}, tuned on {n_train} train questions")
    print(f"recall@{K} = {sum(hits)}/{len(hits)}" + (f" = {100 * sum(hits) / len(hits):.1f}%" if hits else ""))
    misses = [(q, r) for (q, r), h in zip(results, hits) if not h]
    for q, r in misses[:20]:
        correct = [store.memory.memories[i].content[:80] for i in q["correct_ids"] if i in store.memory.memories]
        print(f"\nMISS: {q['question']}\n  correct: {correct}\n  top {K}: {[m.content[:60] for m in r[:K]]}")


if __name__ == "__main__":
    main()
