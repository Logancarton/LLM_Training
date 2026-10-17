"""test_sonara.py — Sonara v1 learns from feedback where cosine misleads (fake encoder, no Ollama)."""
import shutil
import tempfile

from test_sonara_recall import FakeEncoder
import torch
import torch.nn.functional as F

from sonara import CLOCK, FEATURES, VOTE, clock_inputs, features, run
from sonara_recall import Store


def test_clock():
    tmp = tempfile.mkdtemp()
    try:
        store = Store(tmp, encoder=FakeEncoder())
        now = 1784203200.0  # mid-July, no daylight-saving change nearby
        offsets = {"dentist on Tuesday": 7 * 86400, "dentist moved to Friday": 86400,
                   "garlic planted": 12 * 3600, "garlic watered": 12 * 3600 - 60, "dentist on Tuesday ": 30}
        mems = []
        for text, back in offsets.items():
            m = store.add(text, "user_note")
            m.timestamp_epoch = now - back
            mems.append(m)
        x = clock_inputs(mems, F.normalize(store.encoder.vectors([m.content for m in mems]), dim=1), now)
        col = {name: x[:, i] for i, name in enumerate(CLOCK)}
        assert abs(col["log_gap_seconds"][4] - torch.log1p(torch.tensor(30.0))) < 1e-4  # seconds, not days
        assert col["same_time_of_day"][0] > 0.99 and col["same_time_of_day"][1] > 0.99  # 7 days, 1 day ago
        assert col["same_time_of_day"][2] < -0.99  # 12 hours ago: opposite side of the day
        assert col["same_day_of_week"][0] > 0.99 and col["same_day_of_week"][1] < 0.7
        # Sessions start after quiet hours: before 7 days ago, 1 day ago, 12 h ago, 30 s ago.
        assert col["log_sessions_between"].tolist() == torch.log1p(torch.tensor([3.0, 2.0, 1.0, 1.0, 0.0])).tolist()
        assert col["newer_similar"][0] > 0.99  # the same text was written again later
        assert col["newer_similar"][4] == 0.0  # nothing is newer than the latest memory
        life = torch.log1p(torch.tensor([0.0, 6 * 86400, 6.5 * 86400, 6.5 * 86400 + 60, 7 * 86400 - 30]))
        assert torch.allclose(col["log_life_seconds"], life)  # position since the first memory
    finally:
        shutil.rmtree(tmp)
    print("✓ clock tests passed")


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

        results = run(store)
        assert results and len(FEATURES) == x.shape[1]
        assert [r["n_train"] for r in results] == sorted(r["n_train"] for r in results)  # time order
        late = results[len(results) // 2:]  # after Sonara has learned from earlier questions
        rate = {k: sum(r[k] for r in late) / len(late) for k in ("baseline", "control", "sonara")}
        print(rate)
        assert rate["sonara"] > rate["control"] and rate["sonara"] > rate["baseline"]
    finally:
        shutil.rmtree(tmp)
    print("✓ sonara tests passed")


if __name__ == "__main__":
    test_clock()
    test_sonara()
