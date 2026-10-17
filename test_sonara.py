"""test_sonara.py — Sonara v1 learns from feedback where cosine misleads (fake encoder, no Ollama)."""
import shutil
import tempfile

from test_sonara_recall import FakeEncoder
import torch
import torch.nn.functional as F

from sonara import CLOCK, FEATURES, STATE_FILE, VOTE, clock_inputs, features, load_learner, run
from sonara_recall import Store, split_for


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
        x = clock_inputs(store, mems, now)
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
        # The incremental index matches comparing every pair, as of any second.
        texts = ["garlic planted", "garlic planted in the bed", "dentist moved", "garlic", "the bed"]
        later = [store.add(t, "qwen_answer") for t in texts]
        for i, m in enumerate(later):
            m.timestamp_epoch = now + 100 * i
        mems += later
        vecs = F.normalize(store.encoder.vectors([m.content for m in mems]), dim=1)
        times = torch.tensor([m.timestamp_epoch for m in mems], dtype=torch.float64)
        for t in (now + 50, now + 250, now + 1000):
            newer = (times[None, :] > times[:, None]) & (times[None, :] < t)
            full = (vecs @ vecs.T).masked_fill(~newer, -1.0).max(dim=1).values.clamp(min=0.0)
            assert torch.allclose(store.newer_similar(mems, t), full)
        assert Store(tmp, encoder=FakeEncoder())._newer == store._newer  # saved and reloaded
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
            if i == 17:
                load_learner(store)  # saved halfway; the rest is learned on top of the saved state

        def state(learner):
            return [learner.model.weight, learner.model.bias] + [
                v for st in learner.opt.state_dict()["state"].values() for v in st.values()]
        saved = load_learner(store)
        (store.dir / STATE_FILE).unlink()
        replayed = load_learner(store)
        assert all(torch.equal(a, b) for a, b in zip(state(saved), state(replayed)))  # saved == replayed
        stale = torch.load(store.dir / STATE_FILE)
        stale["code"], stale["model"]["weight"] = "older code", stale["model"]["weight"] * 0
        torch.save(stale, store.dir / STATE_FILE)
        assert torch.equal(load_learner(store).model.weight, replayed.model.weight)  # code changed: replay

        q = store.questions()[5]
        mems, x = features(store, q["question"], q["asked_epoch"], store.questions())
        assert x[[m.id for m in mems].index(target.id), VOTE] > 0.99  # earlier identical questions vote
        mems, x = features(store, q["question"], q["asked_epoch"], [])
        assert x[:, VOTE].abs().sum() == 0  # no history, no experience inputs

        results = run(store)
        assert saved.model.weight[0].tolist() == run.final_weights  # run learns the same way
        assert results and len(FEATURES) == x.shape[1]
        assert [r["n_train"] for r in results] == sorted(r["n_train"] for r in results)  # time order
        late = results[len(results) // 2:]  # after Sonara has learned from earlier questions
        rate = {k: sum(r[k + "@5"] for r in late) / len(late) for k in ("baseline", "control", "sonara")}
        print("recall@5", rate)  # this test was built for recall@5; top-1 is reported in LAB_LOG Run 9
        assert rate["sonara"] > rate["control"] and rate["sonara"] > rate["baseline"]
    finally:
        shutil.rmtree(tmp)
    print("✓ sonara tests passed")


def test_shadow_chat():
    """chat logs both rankings, and Sonara learns live from train questions, never from eval."""
    import builtins
    import contextlib
    import io
    import sonara_recall
    tmp = tempfile.mkdtemp()
    real_input, real_ask = builtins.input, sonara_recall.ask_qwen
    try:
        store = Store(tmp, encoder=FakeEncoder())
        for t in ("spare key under the blue flowerpot", "car parked on level 3", "bike in the shed"):
            store.add(t, "user_note").timestamp_epoch = 0.0
        sonara_recall.ask_qwen = lambda question, mems: ("an answer", "a follow-up")
        replies = iter([x for i in range(8) for x in (f"where is the key {i}", "1 2", "")] + [""])
        builtins.input = lambda prompt="": next(replies)
        with contextlib.redirect_stdout(io.StringIO()):
            sonara_recall.chat(store)
        rows = store.questions()
        assert len(rows) == 8 and {r["split"] for r in rows} == {"train", "eval"}
        assert all(r["rankings"]["baseline"] and r["rankings"]["sonara"] for r in rows)
        state = torch.load(store.dir / STATE_FILE)
        assert state["learned"] == [r["id"] for r in rows if r["split"] == "train"]  # never eval

        eval_id = next(f"e{i}" for i in range(99) if split_for(f"e{i}") == "eval")
        weight = load_learner(store).model.weight.clone()
        store.add_question(eval_id, "where is the key", 1e12, rows[0]["correct_ids"], rows[0]["candidate_ids"], None)
        assert torch.equal(load_learner(store).model.weight, weight)  # a labeled eval question teaches nothing
        assert torch.load(store.dir / STATE_FILE)["learned"] == state["learned"]
    finally:
        builtins.input, sonara_recall.ask_qwen = real_input, real_ask
        shutil.rmtree(tmp)
    print("✓ shadow chat tests passed")

if __name__ == "__main__":
    test_clock()
    test_sonara()
    test_shadow_chat()
