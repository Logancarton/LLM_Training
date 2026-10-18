"""test_sonara.py — Sonara v1 learns from feedback where cosine misleads (fake encoder, no Ollama)."""
import datetime as dt
import shutil
import tempfile

from test_sonara_recall import FakeEncoder
import torch
import torch.nn.functional as F

from sonara import (CLOCK, CLOCK_DECAY, FEATURES, LOG_YEAR, LR, STATE_FILE, STEPS, VOTE, Learner, clock_inputs, features,
                    load_learner, occurrence_vectors, run, time_vector)
from sonara_recall import DEFAULT_PARAMS, TUNE_EVERY, Store, split_for


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
        assert abs(col["log_gap_seconds"][4] - torch.log1p(torch.tensor(30.0)) / LOG_YEAR) < 1e-4  # seconds, not days
        assert col["same_time_of_day"][0] > 0.99 and col["same_time_of_day"][1] > 0.99  # 7 days, 1 day ago
        assert col["same_time_of_day"][2] < -0.99  # 12 hours ago: opposite side of the day
        assert col["same_day_of_week"][0] > 0.99 and col["same_day_of_week"][1] < 0.7
        # Sessions start after quiet hours: before 7 days ago, 1 day ago, 12 h ago, 30 s ago.
        assert col["log_sessions_between"].tolist() == torch.log1p(torch.tensor([3.0, 2.0, 1.0, 1.0, 0.0])).tolist()
        assert col["newer_similar"][0] > 0.99  # the same text was written again later
        assert col["newer_similar"][4] == 0.0  # nothing is newer than the latest memory
        life = torch.log1p(torch.tensor([0.0, 6 * 86400, 6.5 * 86400, 6.5 * 86400 + 60, 7 * 86400 - 30]))
        assert torch.allclose(col["log_life_seconds"], life / LOG_YEAR)  # position since the first memory
        assert x[:, [0, 6]].max() < 1.0  # a week of gaps and positions stays below a year's 1
        # The incremental index matches comparing every pair, as of any second; only newer notes count.
        texts = [("garlic planted", "qwen_answer"), ("garlic planted in the bed", "user_note"),
                 ("dentist moved", "qwen_answer"), ("garlic", "user_note"), ("the bed", "qwen_answer")]
        later = [store.add(t, src) for t, src in texts]
        for i, m in enumerate(later):
            m.timestamp_epoch = now + 100 * i
        mems += later
        vecs = F.normalize(store.encoder.vectors([m.content for m in mems]), dim=1)
        times = torch.tensor([m.timestamp_epoch for m in mems], dtype=torch.float64)
        note = torch.tensor([m.source == "user_note" for m in mems])
        for t in (now + 50, now + 250, now + 1000):
            newer = (times[None, :] > times[:, None]) & (times[None, :] < t) & note[None, :]
            full = (vecs @ vecs.T).masked_fill(~newer, -1.0).max(dim=1).values.clamp(min=0.0)
            assert torch.allclose(store.newer_similar(mems, t), full)
        # "garlic planted" (a note) was restated word for word by Qwen at now; that doesn't count,
        # but the newer note "garlic planted in the bed" (now + 100) does.
        assert later[0].content == mems[2].content and later[0].source == "qwen_answer"
        assert store.newer_similar([mems[2]], now + 50).item() < 0.99
        assert store.newer_similar([mems[2]], now + 150).item() > store.newer_similar([mems[2]], now + 50).item()
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

        # Every memory here shares one timestamp, so life position is 0 for all: no gradient, only decay.
        learner = Learner()
        life = 1 + CLOCK.index("log_life_seconds")
        with torch.no_grad():
            learner.model.weight[0, life] = 1.0
        learner.learn(store, store.questions()[3], store.questions()[:3])
        assert abs(learner.model.weight[0, life].item() - (1 - LR * CLOCK_DECAY) ** STEPS) < 1e-6

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
    real_input, real_ask, real_tune, real_split = (builtins.input, sonara_recall.ask_qwen, sonara_recall.tune,
                                                   sonara_recall.split_for)
    try:
        store = Store(tmp, encoder=FakeEncoder())
        notes = [store.add(t, "user_note") for t in
                 ("spare key under the blue flowerpot", "car parked on level 3", "bike in the shed")]
        for m in notes:
            m.timestamp_epoch = 0.0
        # Qwen's stored answers resemble the question and crowd the key note out of the top 5.
        sonara_recall.ask_qwen = lambda question, mems: ("where is the key: not sure", "where is the key, maybe")
        replies = iter([x for i in range(20) for x in (f"where is the key {i}", notes[0].id, "")] + [""])
        builtins.input = lambda prompt="": next(replies)
        tuned_on = []  # the tuner itself is tested in test_sonara_recall; here, when chat calls it
        def fake_tune(store, train):
            tuned_on.append(len(train))
            return {"weight": 0.0, "half_life": 7.0, "bias:user_note": 1.0} if train else dict(DEFAULT_PARAMS)
        sonara_recall.tune = fake_tune
        asked = []  # memory ids are random, so fix the split here: every third question is eval
        sonara_recall.split_for = lambda qid: asked.append(qid) or ("eval" if len(asked) % 3 == 0 else "train")
        with contextlib.redirect_stdout(io.StringIO()):
            sonara_recall.chat(store)
        rows = store.questions()
        assert len(rows) == 20 and {r["split"] for r in rows} == {"train", "eval"}
        assert all(r["rankings"]["baseline"] and r["rankings"]["sonara"] for r in rows)
        assert all(set(r["proposed_by"]) == set(r["candidate_ids"]) for r in rows)
        assert all(set(r["rankings"]["sonara"][:5]) <= set(r["candidate_ids"]) for r in rows)
        state = torch.load(store.dir / STATE_FILE)
        assert state["learned"] == [r["id"] for r in rows if r["split"] == "train"]  # never eval

        # The answering baseline is tuned like curve's: at start, then every 10 labeled train questions.
        train = [r for r in rows if r["split"] == "train"]
        assert tuned_on == [0] + [TUNE_EVERY * k for k in range(1, len(train) // TUNE_EVERY + 1)]
        i = rows.index(train[TUNE_EVERY - 1])
        assert rows[i]["rankings"]["baseline"][0] != notes[0].id  # default settings: crowded out
        assert rows[i + 1]["rankings"]["baseline"][0] == notes[0].id  # re-tuned settings answer next

        sonara_recall.split_for = real_split
        eval_id = next(f"e{i}" for i in range(99) if split_for(f"e{i}") == "eval")
        weight = load_learner(store).model.weight.clone()
        store.add_question(eval_id, "where is the key", 1e12, rows[0]["correct_ids"], rows[0]["candidate_ids"], None)
        assert torch.equal(load_learner(store).model.weight, weight)  # a labeled eval question teaches nothing
        assert torch.load(store.dir / STATE_FILE)["learned"] == state["learned"]
    finally:
        builtins.input, sonara_recall.ask_qwen, sonara_recall.tune, sonara_recall.split_for = (
            real_input, real_ask, real_tune, real_split)
        shutil.rmtree(tmp)
    print("✓ shadow chat tests passed")

def test_tokens():
    """A new token gets a new place, a returning token only a timestamp; each use becomes a vector
    with its date and time."""
    from sonara_memory import TokenStore, tokenize
    tmp = tempfile.mkdtemp()
    try:
        store = Store(tmp, encoder=FakeEncoder())
        noon = dt.datetime(2026, 7, 16, 12, 0).timestamp()
        first = store.add("Dentist moved to Friday 9:30", "user_note")
        again = store.add("dentist again", "qwen_answer")
        first.timestamp_epoch, again.timestamp_epoch = noon, noon + 86400 * 1.5
        assert tokenize("Rent is $1,450 at 9:30, don't") == ["rent", "is", "1,450", "at", "9:30", "don't"]
        tokens = store.tokens()
        assert list(tokens.tokens) == ["dentist", "moved", "to", "friday", "9:30", "again"]  # places
        assert tokens.tokens["dentist"] == {"times": [noon, noon + 86400 * 1.5], "memory_ids": [first.id, again.id]}
        store.add("Friday dentist", "user_question").timestamp_epoch = noon + 86400 * 3
        assert len(store.tokens().tokens) == 6 and len(store.tokens().tokens["friday"]["times"]) == 2  # no new place
        assert TokenStore(store.dir / "tokens.json").tokens == store.tokens().tokens  # saved and reloaded

        occ = occurrence_vectors(store, "dentist", noon + 86400 * 2)  # the use on day 3 isn't known yet
        assert occ.shape == (2, 64 + 7) and torch.allclose(occ[0, 64:], time_vector(noon))
        assert torch.allclose(occ[0, :64], occ[1, :64]) and not torch.allclose(occ[0, 64:], occ[1, 64:])
        tv, midnight = time_vector(noon), time_vector(noon + 12 * 3600)
        assert tv[1] < -0.99 and midnight[1] > 0.99  # time of day: noon vs midnight on the cycle
        assert abs(tv[6] - (noon - dt.datetime(2026, 1, 1).timestamp()) / (365.25 * 86400)) < 1e-5
    finally:
        shutil.rmtree(tmp)
    print("✓ token tests passed")


def test_story():
    """The story's tests use only unseen episodes; the gap lives in timestamps, never in words."""
    import re
    import sonara_memory
    from sonara_story import IMPORTANT, PEOPLE, TRAIN_KINDS, UNIMPORTANT, generate
    tmp = tempfile.mkdtemp()
    try:
        real = sonara_memory.get_timestamps
        episodes = generate(tmp)
        assert sonara_memory.get_timestamps is real  # the story's clock is put back
        memory = sonara_memory.SonaraMemorySystem(tmp)
        split = {e["split"]: [x for x in episodes if x["split"] == e["split"]] for e in episodes}
        assert {k: len(v) for k, v in split.items()} == {"train": 100, "familiar": 20, "unfamiliar": 20, "time": 20}
        assert sum(TRAIN_KINDS.values()) == 100
        train_ids = {i for e in split["train"] for i in e["memory_ids"]}
        assert not train_ids & {i for k in ("familiar", "unfamiliar", "time") for e in split[k] for i in e["memory_ids"]}
        times = [e["query_epoch"] for e in split["train"]]
        assert times == sorted(times) and max(times) < min(e["query_epoch"] for e in episodes if e["split"] != "train")
        text = " ".join(m.content.lower() for m in memory.memories.values())
        assert not re.search(r"minute|hour|day|later|ago|soon|yesterday|morning|evening", text)  # no time words
        train_text = " ".join(memory.memories[i].content for i in train_ids)
        for word in [n for n, _ in PEOPLE["unfamiliar"]] + IMPORTANT["unfamiliar"] + UNIMPORTANT["unfamiliar"]:
            assert word not in train_text  # unfamiliar people and objects never appear in training
        for soon, later in zip(split["time"][::2], split["time"][1::2]):
            assert soon["memory_ids"] == later["memory_ids"] and (soon["mood"], later["mood"]) == ("sad", "happy")
            assert later["query_epoch"] - soon["query_epoch"] > 86400  # only the time differs
    finally:
        shutil.rmtree(tmp)
    print("✓ story tests passed")


if __name__ == "__main__":
    test_clock()
    test_sonara()
    test_shadow_chat()
    test_tokens()
    test_story()
