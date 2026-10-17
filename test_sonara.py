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
        # Half the questions need a decoy that cosine ranks first; the other half keep
        # needing a memory cosine ranks last, which only experience can single out.
        things = ("car", "bike", "phone", "wallet", "charger", "book")
        for i in range(36):
            question, correct = (("where did I leave it", target) if i % 2 else
                                 (f"where did I leave the {things[i // 2 % 6]}", decoys[i // 2 % 6]))
            store.add_question(f"q{i}", question, (i + 1) * day, [correct.id], pool, None)

        q = store.questions()[5]
        mems, x = features(store, q["question"], q["asked_epoch"], store.questions())
        assert x[[m.id for m in mems].index(target.id), VOTE] > 0.99  # earlier identical questions vote
        mems, x = features(store, q["question"], q["asked_epoch"], [])
        assert x[:, VOTE].abs().sum() == 0  # no history, no experience inputs

        test, rows = curve(store)
        assert test and len(rows) == 4 and len(FEATURES) == x.shape[1]
        last = rows[-1]
        print({k: last[k] for k in ("n_train", "baseline", "rule", "no_exp", "sonara")})
        assert last["sonara"] > last["no_exp"] and last["sonara"] > last["baseline"]
    finally:
        shutil.rmtree(tmp)
    print("✓ sonara tests passed")


if __name__ == "__main__":
    test_sonara()
