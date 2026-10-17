"""test_sonara_recall.py — baseline, splits and recall@5 with a fake encoder (no Ollama)."""
import hashlib
import shutil
import tempfile

import torch

from sonara_recall import Store, baseline_rank, hit, score, split_for


class FakeEncoder:
    """Bag of hashed words: texts sharing words get similar vectors."""

    def __init__(self):
        self.fetched = []

    def ensure(self, texts):
        texts = list(texts)
        self.fetched += texts
        return len(texts)

    def save(self, path):
        pass

    def vectors(self, texts):
        out = torch.zeros(len(texts), 64)
        for i, t in enumerate(texts):
            for w in t.lower().strip("?.").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out


def test_recall():
    tmp = tempfile.mkdtemp()
    try:
        store = Store(tmp, encoder=FakeEncoder())
        day = 86400.0
        old = store.add("the dog Jake liked tennis balls", "user_note")
        new = store.add("the car needs new brakes", "user_note")
        old.timestamp_epoch, new.timestamp_epoch = 0.0, 10 * day
        assert store.encoder.fetched == [old.content, new.content]  # each memory embedded once

        ranked = baseline_rank(store, "what did Jake like", 20 * day)
        assert ranked[0].id == old.id
        assert [m.id for m in baseline_rank(store, "what did Jake like", 5 * day)] == [old.id]  # no future memories
        assert hit(ranked, [old.id]) and not hit(ranked, ["missing"])

        for i in range(12):
            store.add_question(f"q{i}", "what did Jake like", 20 * day, [old.id], [old.id, new.id], None)
        assert {q["split"] for q in store.questions()} == {"train", "eval"}
        assert all(q["split"] == split_for(q["id"]) for q in store.questions())

        params, n_train, results = score(store)
        assert n_train == sum(q["split"] == "train" for q in store.questions())
        assert results and all(hit(r, q["correct_ids"]) for q, r in results)
        assert score(store, split_epoch=30 * day)[2] == []  # no eval questions after the split
    finally:
        shutil.rmtree(tmp)
    print("✓ recall tests passed")


if __name__ == "__main__":
    test_recall()
