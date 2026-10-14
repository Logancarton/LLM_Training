"""Sonara's world-state owner: evidence-grounded states plus a bounded neural transition learner.

One memory repository remains authoritative for episodes. Neural output estimates whether
an observation signals a state change; it never silently rewrites observed evidence.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

STATE_VERBS = r"(?:moved|carried|put|placed|dropped|threw|took|brought|rolled|retrieved)"
ENTITY = r"[a-z][a-z0-9_-]*"
CHANGE = re.compile(
    rf"\b{STATE_VERBS}\s+(?:the\s+|a\s+)?(?P<entity>{ENTITY})\s+"
    rf"(?:to|into|in|at|on|from)\s+(?:the\s+|a\s+)?(?P<place>{ENTITY})\b", re.I)
STATE = re.compile(
    rf"\b(?:the\s+|a\s+)?(?P<entity>{ENTITY})\s+(?:is|was|stayed|remains)\s+"
    rf"(?:in|at|on|inside)\s+(?:the\s+|a\s+)?(?P<place>{ENTITY})\b", re.I)
WHERE = re.compile(r"\bwhere\s+(?:is|was|did)\s+(?:the\s+|a\s+)?(?P<entity>[a-z][a-z0-9_-]*)", re.I)
WHERE_GO = re.compile(r"\bwhere\s+did\s+(?:the\s+|a\s+)?(?P<entity>[a-z][a-z0-9_-]*)\s+go\b", re.I)
TIME = re.compile(r"\b(\d{4}-\d\d-\d\dT\d\d:\d\d(?::\d\d)?(?:Z|[+-]\d\d:\d\d))\b")
STOP = {"a", "the", "to", "at", "is", "was", "did", "go", "where"}


def parse_state(text: str, recorded_at: str) -> dict | None:
    """Extract only explicitly asserted state, never invent a destination."""
    match = CHANGE.search(text) or STATE.search(text)
    if not match:
        return None
    ts = TIME.search(text)
    observed_at = ts.group(1).replace("Z", "+00:00") if ts else recorded_at
    return {"entity": match["entity"].lower(), "location": match["place"].lower(),
            "observed_at": observed_at, "relation": "at", "extraction": "bounded-pattern"}


def token_ids(text: str, max_length: int = 32) -> torch.Tensor:
    words = re.findall(r"[a-z0-9_-]+", text.lower())[:max_length]
    # Stable hashing allows unseen words without vocabulary/embedding expansion.
    ids = [1 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 2047
           for word in words]
    return torch.tensor([ids or [0]], dtype=torch.long)


class WorldModel(nn.Module):
    """Small, independently trainable estimator of observation-driven state change."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2048, 24)
        self.sequence = nn.GRU(24, 32, batch_first=True)
        self.change_head = nn.Linear(32, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, state = self.sequence(self.embedding(tokens))
        return self.change_head(state[-1])


class WorldCognition:
    """Integrates the neural world model with existing SonaraMemorySystem evidence."""
    def __init__(self, memories, data_dir: str | Path):
        self.memories = memories
        self.checkpoint = Path(data_dir) / "world_model.pt"
        self.net = WorldModel()
        self.trained = False
        if self.checkpoint.exists():
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state["weights"])
            self.trained = bool(state.get("validated", False))
        self.net.eval()

    def observe(self, text: str, session_id: str):
        # Store in THE existing memory owner; no shadow episodic store.
        from sonara_memory import get_timestamps
        recorded_at, _, _ = get_timestamps()
        extracted = parse_state(text, recorded_at)
        frame = extracted if extracted is not None else None
        return self.memories.remember(text, category="episodic", source="user_observation",
                                      session_id=session_id, semantic_frame=frame)

    def change_probability(self, text: str) -> float | None:
        if not self.trained:
            return None
        self.net.eval()
        with torch.no_grad():
            return float(F.softmax(self.net(token_ids(text)), dim=-1)[0, 1])

    def answer_where(self, question: str) -> str | None:
        match = WHERE_GO.search(question) or WHERE.search(question)
        if not match:
            return None
        entity = match["entity"].lower()
        if re.search(r"\\b(?:at|on)\\s+\\d", question, re.I):
            return "Unknown: time-qualified world queries need an explicit event-time reasoning test."
        observations = []
        for memory_id in self.memories.entity_index.get(entity, ()):
            memory = self.memories.memories[memory_id]
            if memory.category != "episodic" or memory.source != "user_observation":
                continue
            state = memory.semantic_frame
            if state.get("entity") == entity and state.get("location"):
                observations.append((state.get("observed_at", memory.timestamp_iso), memory, state))
        if not observations:
            return f"Unknown: no explicit location observation for {entity}."
        # Compare instants, not ISO strings with different timezone offsets.
        observations.sort(key=lambda row: (datetime.fromisoformat(row[0]),
                                           datetime.fromisoformat(row[1].timestamp_iso),
                                           row[1].id))
        _, evidence, state = observations[-1]
        probability = self.change_probability(evidence.content)
        suffix = (f" World-model state-change estimate: {probability:.2f} (experimental, not verification)."
                  if probability is not None else "")
        return (f"Observed report: {entity} was at {state['location']} "
                f"(event time {state.get('observed_at', 'unknown')}; memory {evidence.id})."
                f" This is attributed evidence, not independently verified truth.{suffix}")

    def learn_bounded(self, max_examples: int = 64, steps: int = 40, seed: int = 19) -> dict:
        """Weak supervision: fit from a bounded selection of observed location assertions.

        This is intentionally NOT automatic ingestion or generic causal learning.
        Acceptance uses a held-out sample; original evidence is never overwritten.
        """
        categories = defaultdict(list)
        for item in sorted(self.memories.memories.values(), key=lambda m: m.id):
            if item.category != "episodic" or item.source != "user_observation":
                continue
            target = int(bool(item.semantic_frame.get("location") and item.semantic_frame.get("entity")))
            categories[target].append(item.content)
        if min(len(categories[0]), len(categories[1])) < 4:
            return {"accepted": False, "reason": "Need at least 4 explicit location and 4 unrelated observations"}
        generator = torch.Generator().manual_seed(seed)
        pools = {}
        cap = min(max_examples // 2, len(categories[0]), len(categories[1]))
        if cap < 4:
            raise ValueError("max_examples must be at least 8")
        for label in (0, 1):
            order = torch.randperm(len(categories[label]), generator=generator).tolist()
            pools[label] = [categories[label][i] for i in order[:cap]]
        held = [(pools[label][-1], label) for label in (0, 1)]
        train = [(text, label) for label in (0, 1) for text in pools[label][:-1]]
        original = copy.deepcopy(self.net.state_dict())
        self.net.eval()

        def loss_on(samples):
            with torch.no_grad():
                return sum(float(F.cross_entropy(self.net(token_ids(text)),
                                      torch.tensor([label]))) for text, label in samples) / len(samples)
        before = loss_on(held)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.005)
        self.net.train()
        for step in range(steps):
            text, label = train[step % len(train)]
            loss = F.cross_entropy(self.net(token_ids(text)), torch.tensor([label]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.net.eval()
        after = loss_on(held)
        accepted = after + 1e-5 < before
        if not accepted:
            self.net.load_state_dict(original)
        else:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"weights": self.net.state_dict(), "validated": True,
                        "examples": len(train), "heldout_loss": after}, self.checkpoint)
            self.trained = True
        return {"accepted": accepted, "examples": len(train), "holdout": len(held),
                "loss_before": round(before, 4), "loss_after": round(after, 4),
                "note": "Weak location-pattern supervision, not causal-world understanding"}
