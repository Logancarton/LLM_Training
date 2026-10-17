"""test_sonara.py — Sonara v1 learns from feedback where cosine misleads (fake encoder, no Ollama)."""
import shutil
import tempfile

from test_sonara_recall import FakeEncoder
from sonara import FEATURES, VOTE, curve, features
from sonara_recall import Store


def test_sonara():
    tmp = tempfile.mkdtemp()
    try:
        store = Store(tmp, encoder=FakeEncoder())
        day = 86400.0
        target = store.add("spare key under the blue flowerpot", "user_note")
        decoys = [store.add(f"where did I leave the {t}", "user_note")
                  for t in ("car", "bike", "phone", "wallet", "charger", "book")]
        for m in [target] + decoys:
            m.timestamp_epoch = 0.0
        pool = [target.id] + [d.id for d in decoys]
        for i in range(24):  # the same question keeps needing a memory cosine ranks last
            store.add_question(f"q{i}", "where did I leave it", (i + 1) * day, [target.id], pool, None)

        q = store.questions()[5]
        mems, x = features(store, q["question"], q["asked_epoch"], store.questions())
        assert x[[m.id for m in mems].index(target.id), VOTE] > 0.99  # earlier identical questions vote
        mems, x = features(store, q["question"], q["asked_epoch"], [])
        assert x[:, VOTE].abs().sum() == 0  # no history, no experience inputs

        test, rows = curve(store)
        assert test and len(rows) == 4 and len(FEATURES) == x.shape[1]
        assert rows[-1]["baseline"] == 0.0
        assert rows[-1]["sonara"] == 100.0 and rows[-1]["rule"] == 100.0
    finally:
        shutil.rmtree(tmp)
    print("✓ sonara tests passed")


if __name__ == "__main__":
    test_sonara()
