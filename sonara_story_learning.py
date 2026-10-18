"""Controlled incremental timestamp-learning experiment for sonara_story.py.

Runs without Ollama. Trains identical small PyTorch learners with and without
time, plus a diagnostic supplied with correct synthetic event meanings. A
successful diagnostic is NOT evidence Sonara independently understands events
or learns causal relationships.

    python3 sonara_story_learning.py --seeds 5 --steps 12
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sonara_memory import SonaraMemorySystem
from sonara_story import generate

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
CHECKPOINTS = (0, 10, 30, 60, 100)
SPLITS = ("familiar", "unfamiliar", "time")
MAX_EVENTS, MAX_WORDS = 4, 24


@dataclass
class Example:
    id: str
    split: str
    query_epoch: float
    label: int
    events: list[tuple[float, str]]
    kind: str = ""


def load_examples(directory: str, seed: int) -> list[Example]:
    """Read ONLY observations from this episode strictly before its query time."""
    episodes = generate(directory, seed=seed)
    memory = SonaraMemorySystem(directory)
    out = []
    for item in episodes:
        events = [(memory.memories[mid].timestamp_epoch, memory.memories[mid].content)
                  for mid in item["memory_ids"]
                  if memory.memories[mid].timestamp_epoch < item["query_epoch"]]
        assert len(events) >= 2
        assert all(t < item["query_epoch"] for t, _ in events)
        out.append(Example(item["id"], item["split"], item["query_epoch"],
                           int(item["mood"] == "sad"), events, item["kind"]))
    train = [e for e in out if e.split == "train"]
    held_out = [e for e in out if e.split != "train"]
    assert len(train) == 100 and len(held_out) == 60
    assert [e.query_epoch for e in train] == sorted(e.query_epoch for e in train)
    assert max(e.query_epoch for e in train) < min(e.query_epoch for e in held_out)
    return out


def vocabulary(train: list[Example]) -> dict[str, int]:
    """Only training text constructs the vocabulary; no held-out text or labels."""
    words = sorted({w for ex in train for _, phrase in ex.events
                    for w in TOKEN.findall(phrase.lower())})
    return {"<pad>": 0, "<unknown>": 1,
            **{w: index + 2 for index, w in enumerate(words)}}


def time_features(event_time: float, query_time: float) -> list[float]:
    """Continuous elapsed-time and clock coordinates; no half-life or fixed relevance."""
    gap_minutes = max(0.0, (query_time - event_time) / 60.0)
    event_phase = (event_time % 86400) / 86400
    query_phase = (query_time % 86400) / 86400
    return [math.log1p(gap_minutes) / math.log1p(7 * 1440),
            math.sin(2 * math.pi * event_phase),
            math.cos(2 * math.pi * event_phase),
            math.sin(2 * math.pi * query_phase),
            math.cos(2 * math.pi * query_phase)]


def encode(ex: Example, vocab: dict[str, int]):
    tokens = torch.zeros(MAX_EVENTS, MAX_WORDS, dtype=torch.long)
    times = torch.zeros(MAX_EVENTS, 5)
    mask = torch.zeros(MAX_EVENTS, dtype=torch.bool)
    for i, (when, phrase) in enumerate(ex.events[-MAX_EVENTS:]):
        ids = [vocab.get(w, 1) for w in TOKEN.findall(phrase.lower())][:MAX_WORDS]
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        times[i] = torch.tensor(time_features(when, ex.query_epoch))
        mask[i] = True
    return tokens, times, mask


class TemporalLearner(nn.Module):
    """A shared time encoder; the ablation erases clocks with identical weights."""
    def __init__(self, n_words: int, use_time: bool):
        super().__init__()
        self.use_time = use_time
        self.words = nn.Embedding(n_words, 24, padding_idx=0)
        self.clock = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 24))
        self.events = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        self.attend = nn.Linear(32, 1)
        self.output = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, tokens, times, mask):
        word_mask = (tokens != 0).float()
        text = (self.words(tokens) * word_mask.unsqueeze(-1)).sum(2)
        text = text / word_mask.sum(2).clamp(min=1).unsqueeze(-1)
        temporal = self.clock(times if self.use_time else torch.zeros_like(times))
        events = self.events(torch.cat((text, temporal), dim=-1))
        weights = self.attend(events).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = (torch.softmax(weights, dim=-1).unsqueeze(-1) * events).sum(1)
        return self.output(pooled)


def metrics(predictions: list[int], examples: list[Example]) -> dict[str, float]:
    result = {}
    for split in SPLITS:
        subset = [i for i, e in enumerate(examples) if e.split == split]
        result[split] = sum(predictions[i] == examples[i].label for i in subset) / len(subset)
    result["overall"] = sum(p == e.label for p, e in zip(predictions, examples)) / len(examples)
    pairs = [i for i, e in enumerate(examples[:-1])
             if e.split == "time" and e.id.endswith("soon")
             and examples[i + 1].split == "time"
             and examples[i + 1].events == e.events]
    assert len(pairs) == 10
    result["time_both"] = sum(predictions[i] == examples[i].label
                              and predictions[i + 1] == examples[i + 1].label for i in pairs) / len(pairs)
    return result


def train_and_measure(data: list[Example], seed: int, steps: int) -> list[dict]:
    """Train after every experience; sample only earlier/current train episodes."""
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    train = [e for e in data if e.split == "train"]
    evals = [e for e in data if e.split != "train"]
    vocab = vocabulary(train)
    cached = [encode(e, vocab) for e in train]
    model = TemporalLearner(len(vocab), True)
    control = TemporalLearner(len(vocab), False)
    control.load_state_dict(model.state_dict())
    models = {"time": model, "no_time": control}
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=0.008) for name, m in models.items()}
    test_inputs = [encode(e, vocab) for e in evals]
    eval_tensors = [torch.stack([x[i] for x in test_inputs]) for i in range(3)]
    reports = []
    for seen in range(101):
        if seen in CHECKPOINTS:
            for name, m in models.items():
                m.eval()
                with torch.no_grad():
                    predictions = m(*eval_tensors).argmax(-1).tolist()
                reports.append({"seed": seed, "seen": seen, "arm": name,
                                **metrics(predictions, evals)})
        if seen == 100:
            break
        # Identical rehearsal schedule in both arms; no eval observations or labels.
        sample_ids = [seen] + random.choices(range(seen + 1), k=3)
        subset = [cached[i] for i in sample_ids]
        xs = [torch.stack([x[i] for x in subset]) for i in range(3)]
        ys = torch.tensor([train[i].label for i in sample_ids], dtype=torch.long)
        for name, m in models.items():
            m.train()
            for _ in range(steps):
                loss = nn.functional.cross_entropy(m(*xs), ys)
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()
    return reports


def semantic_diagnostic(data: list[Example], seed: int, steps: int) -> list[dict]:
    """Oracle event meaning isolates learning of TIME. Not an independent
    language-understanding or causal-reasoning result.

    Important-loss-soon and important-loss-days-later get exactly the SAME
    semantic category, so only elapsed time can distinguish these situations.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    train = [e for e in data if e.split == "train"]
    evals = [e for e in data if e.split != "train"]

    def semantic(ex):
        category = {"lose_important_soon": 0, "lose_important_days_later": 0,
                    "lose_unimportant": 1, "sad_other_reason": 2, "lose_then_find": 3}[ex.kind]
        vector = torch.zeros(5)
        vector[category] = 1
        vector[4] = time_features(ex.events[-1][0], ex.query_epoch)[0]
        return vector

    class Diagnostic(nn.Module):
        def __init__(self, timed):
            super().__init__()
            self.timed = timed
            self.net = nn.Sequential(nn.Linear(5, 24), nn.Tanh(), nn.Linear(24, 2))

        def forward(self, x):
            if not self.timed:
                x = x.clone()
                x[:, 4] = 0
            return self.net(x)

    model = Diagnostic(True)
    control = Diagnostic(False)
    control.load_state_dict(model.state_dict())
    models = {"oracle_time": model, "oracle_no_time": control}
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=0.008) for name, m in models.items()}
    xs = torch.stack([semantic(e) for e in train])
    ys = torch.tensor([e.label for e in train], dtype=torch.long)
    xt = torch.stack([semantic(e) for e in evals])
    reports = []
    for seen in range(101):
        if seen in CHECKPOINTS:
            for name, m in models.items():
                m.eval()
                with torch.no_grad():
                    predictions = m(xt).argmax(-1).tolist()
                reports.append({"seed": seed, "seen": seen, "arm": name,
                                **metrics(predictions, evals)})
        if seen == 100:
            break
        sample_ids = [seen] + random.choices(range(seen + 1), k=3)
        for name, m in models.items():
            m.train()
            for _ in range(steps):
                loss = nn.functional.cross_entropy(m(xs[sample_ids]), ys[sample_ids])
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()
    return reports


def experiment(seeds: int = 5, steps: int = 12) -> list[dict]:
    rows = []
    for seed in range(seeds):
        with tempfile.TemporaryDirectory(prefix="sonara_story_learn_") as directory:
            data = load_examples(directory, seed)
            rows.extend(train_and_measure(data, seed, steps))
            rows.extend(semantic_diagnostic(data, seed, steps))
    return rows


def summarize(rows: list[dict]) -> None:
    n_seeds = len({r["seed"] for r in rows})
    for seen in CHECKPOINTS:
        print(f"\nAfter {seen} experiences (mean across {n_seeds} seeds)")
        print("                 familiar unfamiliar     time time_both  overall")
        for name in ("no_time", "time", "oracle_no_time", "oracle_time"):
            subset = [r for r in rows if r["seen"] == seen and r["arm"] == name]
            values = " ".join(f"{statistics.mean(r[k] for r in subset):8.1%}"
                              for k in ("familiar", "unfamiliar", "time", "time_both", "overall"))
            print(f"{name:>16} {values}")
    print("\nORACLE MEANING is provided from the generator; this is NOT evidence of "
          "autonomous understanding or causal reasoning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not (1 <= args.seeds <= 100 and 1 <= args.steps <= 100):
        parser.error("--seeds and --steps must be between 1 and 100")
    rows = experiment(args.seeds, args.steps)
    summarize(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
