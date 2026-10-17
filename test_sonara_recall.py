"""test_sonara_recall.py — baseline, splits and recall@5 with a fake encoder (no Ollama)."""
import hashlib
import shutil
import tempfile

import torch

from sonara_recall import POOL_COSINE, POOL_NEWEST, Store, baseline_rank, candidates, hit, label_pool, score, split_for, tune


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

        # Six newer generated memories that resemble the question push the note out of
        # the top 5; tuning on train questions should learn a penalty for that source.
        for i in range(6):
            store.add(f"what Jake did {i}", "qwen_answer").timestamp_epoch = 15 * day
        train = [{"question": "Jake liked what", "asked_epoch": 20 * day, "correct_ids": [old.id]}]
        assert not hit(baseline_rank(store, "Jake liked what", 20 * day), [old.id])
        params = tune(store, train)
        assert params["bias:qwen_answer"] < 0
        assert hit(baseline_rank(store, "Jake liked what", 20 * day, params), [old.id])

        # Every method ranks from a bounded set: the top memories by cosine plus the newest.
        for i in range(70):
            store.add(f"filler note {i}", "user_note").timestamp_epoch = 16 * day + i
        pool = candidates(store, "what did Jake like", 20 * day)
        assert len(pool) <= POOL_COSINE + POOL_NEWEST and old in pool
        assert all(store.memory.memories[m.id] in pool for m in store.before(20 * day)[-POOL_NEWEST:])

        # The labeling pool unions each system's top 5 with the newest and records who proposed what.
        ranked = baseline_rank(store, "what did Jake like", 20 * day)
        only_sonara = ranked[-1]  # last for the baseline, first for Sonara
        labels, proposed = label_pool(store, 20 * day, {"baseline": ranked, "sonara": [only_sonara] + ranked[:4]})
        assert only_sonara in labels and proposed[only_sonara.id] == ["sonara"]
        assert set(proposed) == {m.id for m in labels} and all(m in labels for m in ranked[:5])
        assert sum("newest" in v for v in proposed.values()) == 5
    finally:
        shutil.rmtree(tmp)
    print("✓ recall tests passed")


if __name__ == "__main__":
    test_recall()
