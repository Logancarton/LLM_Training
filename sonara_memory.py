"""A bounded paragraph -> temporal event graph -> sparse recall prototype.

Extraction is intentionally conservative: unsupported sentences remain source notes.
The association scorer is genuinely fitted from labeled examples (see train.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sqlite3

from association_signals import brain_signals


@dataclass
class Episode:
    actor: str = ""
    item: str = ""
    action: str = ""
    location: str = ""
    stated_time: str = ""
    kind: str = "note"
    source: str = ""
    novelty: bool = False
    id: int = 0

    @property
    def identity(self) -> str:
        return f"{self.actor.lower()}::{self.item.lower()}" if self.actor and self.item else ""


class ParagraphEncoder:
    """Conservative teaching grammar, not an open-domain language model."""

    ACT = re.compile(
        r"^(?:(?:At|On|Later) (?P<time>[^,]+),\s*)?"
        r"(?P<actor>[A-Z][a-z]+|she|he)\s+"
        r"(?P<verb>put|placed|moved|carried|left|stored|kept)\s+"
        r"(?:(?:the|her|his|a|an)\s+)?(?P<item>[a-z][a-z -]*?|them|it)\s+"
        r"(?:on|in|into|to|at|by)\s+(?:(?:the|her|his|a|an)\s+)?"
        r"(?P<location>[a-z][a-z -]+)$", re.I
    )
    HABIT = re.compile(
        r"^Usually,\s+(?P<actor>[A-Z][a-z]+|she|he)\s+"
        r"(?:keeps|stores|puts)\s+(?:(?:the|her|his)\s+)?"
        r"(?P<item>[a-z][a-z -]*?|them|it)\s+"
        r"(?:at|on|in|by|near)\s+(?:(?:the|her|his)\s+)?"
        r"(?P<location>[a-z][a-z -]+)$", re.I
    )

    def encode(self, paragraph: str) -> list[Episode]:
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise ValueError("A nonempty paragraph is required")
        output = []
        recent_actors: set[str] = set()
        recent_items: set[str] = set()
        for raw in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
            sentence = raw.strip().rstrip(".!?")
            if not sentence:
                continue
            match = self.ACT.fullmatch(sentence) or self.HABIT.fullmatch(sentence)
            if match is None:
                output.append(Episode(source=raw.strip()))  # preserve evidence, don't invent slots
                continue
            vals = match.groupdict()
            actor = vals["actor"]
            item = vals["item"]
            if actor.lower() in {"she", "he"}:
                actor = next(iter(recent_actors)) if len(recent_actors) == 1 else ""
            if item.lower() in {"them", "it"}:
                item = next(iter(recent_items)) if len(recent_items) == 1 else ""
            if not actor or not item:
                output.append(Episode(source=raw.strip()))
                continue
            kind = "habit" if "verb" not in vals else "location"
            entry = Episode(
                actor=actor.title(), item=item.lower().strip(),
                action=(vals.get("verb") or "usually").lower(),
                location=vals["location"].lower().strip(),
                stated_time=(vals.get("time") or "").strip(),
                kind=kind, source=raw.strip(),
            )
            output.append(entry)
            recent_actors.add(entry.actor)
            recent_items.add(entry.item)
        return output


def features(a: Episode, b: Episode) -> dict[str, float]:
    """General relation signals, not actor/object memorization."""
    return {
        "same_identity": float(bool(a.identity) and a.identity == b.identity),
        "same_actor": float(bool(a.actor) and a.actor == b.actor),
        "same_item": float(bool(a.item) and a.item == b.item),
        "same_location": float(bool(a.location) and a.location == b.location),
        "location_changed": float(bool(a.location and b.location) and a.location != b.location),
        "habit_pair": float(a.kind == "habit" or b.kind == "habit"),
        "both_locations": float(a.kind == "location" and b.kind == "location"),
    } | brain_signals(a, b)


class LinkLearner:
    """Two supervised logistic scorers: identity continuity vs contextual association.

    Scores are uncalibrated affinity estimates, not percent-confidence in a fact.
    Contextual edges can never override the strict factual identity boundary.
    """
    def __init__(self, weights: dict[str, float] | None = None, bias: float = 0.0,
                 context_weights: dict[str, float] | None = None, context_bias: float = 0.0):
        keys = features(Episode(), Episode())
        self.weights = dict(weights) if weights is not None else {k: 0.0 for k in keys}
        self.bias = bias
        self.context_weights = (dict(context_weights) if context_weights is not None
                                else {k: 0.0 for k in keys})
        self.context_bias = context_bias

    @staticmethod
    def _score(a: Episode, b: Episode, weights: dict[str, float], bias: float) -> float:
        x = bias + sum(weights[k] * v for k, v in features(a, b).items())
        return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))

    def probability(self, a: Episode, b: Episode) -> float:
        return self._score(a, b, self.weights, self.bias)

    def context_probability(self, a: Episode, b: Episode) -> float:
        return self._score(a, b, self.context_weights, self.context_bias)

    @staticmethod
    def _fit(samples: list[tuple[Episode, Episode, int]], weights: dict[str, float],
             bias: float, epochs: int, rate: float) -> float:
        if not samples or {label for _, _, label in samples} != {0, 1}:
            raise ValueError("Training needs positive and negative associations")
        # Precompute fixed evidence cues once: train parameters, not tokenization.
        cached = [({k: v for k, v in features(a, b).items() if v}, label)
                  for a, b, label in samples]
        for _ in range(epochs):
            for vals, label in cached:
                logit = bias + sum(weights[k] * v for k, v in vals.items())
                predicted = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
                error = label - predicted
                for k, value in vals.items():
                    weights[k] += rate * error * value
                bias += rate * error
            for k in weights:
                weights[k] *= 1.0 - rate * 0.001
        return bias

    def fit(self, samples: list[tuple[Episode, Episode, int]], epochs: int = 100,
            rate: float = 0.08) -> None:
        self.bias = self._fit(samples, self.weights, self.bias, epochs, rate)

    def fit_context(self, samples: list[tuple[Episode, Episode, int]], epochs: int = 100,
                    rate: float = 0.08) -> None:
        self.context_bias = self._fit(samples, self.context_weights,
                                      self.context_bias, epochs, rate)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": 2, "weights": self.weights,
                "bias": self.bias, "context_weights": self.context_weights,
                "context_bias": self.context_bias}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LinkLearner:
        raw = json.loads(Path(path).read_text())
        expected = set(features(Episode(), Episode()))
        if (raw.get("version") != 2 or set(raw["weights"]) != expected or
                set(raw["context_weights"]) != expected):
            raise ValueError("Model version changed: run python train.py --model model.json")
        return cls({k: float(v) for k, v in raw["weights"].items()},
                   float(raw["bias"]),
                   {k: float(v) for k, v in raw["context_weights"].items()},
                   float(raw["context_bias"]))


class Memory:
    """SQLite evidence store; sparse indexed seeds followed by bounded learned edges."""
    def __init__(self, database: str | Path, learner: LinkLearner | None = None,
                 threshold: float = 0.70):
        self.conn = sqlite3.connect(str(database))
        self.conn.row_factory = sqlite3.Row
        self.learner = learner or LinkLearner()
        self.threshold = threshold
        self.encoder = ParagraphEncoder()
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY, identity TEXT NOT NULL, actor TEXT, item TEXT,
              action TEXT, location TEXT, stated_time TEXT, kind TEXT,
              source TEXT NOT NULL, novelty INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS event_lookup ON events(identity,kind,id);
            CREATE INDEX IF NOT EXISTS event_actor ON events(actor,id);
            CREATE INDEX IF NOT EXISTS event_place ON events(location,id);
            CREATE TABLE IF NOT EXISTS links (
              source INTEGER NOT NULL, target INTEGER NOT NULL, strength REAL NOT NULL,
              relation TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '[]',
              PRIMARY KEY(source,target)
            );
            CREATE INDEX IF NOT EXISTS link_target ON links(target,strength);
        """)
        # Forward migration for existing persistent memory files.
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(links)")}
        if "evidence" not in columns:
            self.conn.execute("ALTER TABLE links ADD COLUMN evidence TEXT NOT NULL DEFAULT '[]'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _episode(row: sqlite3.Row) -> Episode:
        return Episode(actor=row["actor"], item=row["item"], action=row["action"],
                       location=row["location"], stated_time=row["stated_time"],
                       kind=row["kind"], source=row["source"],
                       novelty=bool(row["novelty"]), id=row["id"])

    def ingest(self, paragraph: str) -> list[Episode]:
        entries = self.encoder.encode(paragraph)
        with self.conn:
            for entry in entries:
                cursor = self.conn.execute("""INSERT INTO events
                    (identity,actor,item,action,location,stated_time,kind,source,novelty)
                    VALUES (?,?,?,?,?,?,?,?,0)""",
                    (entry.identity, entry.actor, entry.item, entry.action, entry.location,
                     entry.stated_time, entry.kind, entry.source))
                entry.id = cursor.lastrowid
                if entry.identity:
                    self._update_novelty(entry.identity)
            # Re-read after novelty updates so edges incorporate the current evidence.
            for entry in entries:
                entry.novelty = bool(self.conn.execute(
                    "SELECT novelty FROM events WHERE id=?", (entry.id,)).fetchone()[0])
                if not entry.identity:
                    continue  # unmatched notes stay evidence-only, not speculative links
                same_identity = self.conn.execute("""SELECT * FROM events
                    WHERE identity=? AND id<? ORDER BY id DESC LIMIT 12""",
                    (entry.identity, entry.id)).fetchall()
                identity_edges = []
                for row in same_identity:
                    old = self._episode(row)
                    score = self.learner.probability(entry, old)
                    if score >= self.threshold:
                        identity_edges.append((score, old, "identity-continuity"))
                # Enforce sparse storage too: at most 3 identity edges per episode.
                for score, old, relation in sorted(identity_edges, key=lambda x: -x[0])[:3]:
                    self._link(entry, old, score, relation)
                # Context links are *not* factual identity links; require shared
                # explicit actor/place. Only the 12 most recent per index, bounded.
                candidates = {}
                for column, value in (("actor", entry.actor), ("location", entry.location)):
                    if not value:
                        continue
                    for row in self.conn.execute(f"""SELECT * FROM events WHERE {column}=?
                        AND id<? AND identity!=? ORDER BY id DESC LIMIT 12""",
                        (value, entry.id, entry.identity)):
                        candidates[row["id"]] = row
                context_edges = []
                for row in candidates.values():
                    old = self._episode(row)
                    score = self.learner.context_probability(entry, old)
                    if score >= self.threshold:
                        relation = ("shared-actor" if entry.actor == old.actor else
                                    "shared-place")
                        context_edges.append((score, old, relation))
                for score, old, relation in sorted(context_edges, key=lambda x: -x[0])[:2]:
                    self._link(entry, old, score, relation)
        return entries

    def _link(self, new: Episode, old: Episode, score: float, relation: str) -> None:
        # Explainable evidence cues, not fake confidences about particular facts.
        weights = (self.learner.weights if relation == "identity-continuity"
                   else self.learner.context_weights)
        cues = sorted(((key, round(val, 4), round(weights[key] * val, 4))
                       for key, val in features(new, old).items() if val > 0),
                      key=lambda cue: (-abs(cue[2]), cue[0]))[:8]
        self.conn.execute("""INSERT OR REPLACE INTO links
            (source,target,strength,relation,evidence) VALUES (?,?,?,?,?)""",
            (new.id, old.id, score, relation, json.dumps(cues)))

    def reactivate(self, actor: str, item: str, limit: int = 5,
                   hops: int = 2, beam: int = 8) -> dict:
        """Specific episode cue -> associated context -> reinstated source episodes.

        This is a bounded learned graph, not a simulated hippocampus. No graph
        path certifies a fact or creates an identity, causal or temporal relation.
        """
        if not 1 <= limit <= 20 or not 1 <= hops <= 3 or not 1 <= beam <= 20:
            raise ValueError("limit, hops and beam must be within bounds")
        identity = f"{actor.casefold()}::{item.casefold()}"
        row = self.conn.execute("""SELECT * FROM events WHERE identity=?
            ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
        empty = {"detail_cue": [], "broader_context": [],
                 "detail_reinstatement": []}
        if row is None:
            return {"seed": None, "related": [], "examined": 0, "phases": empty}
        seed = self._episode(row)
        cue = {"id": seed.id, "identity": identity, "source": seed.source}
        # The only routes are the existing supervised association edges.
        frontier = [(seed.id, 1.0, (seed.id,), ())]
        winners, contexts = {}, []
        examined = 0
        for depth in range(1, hops + 1):
            next_frontier = {}
            for current_id, activation, path, path_links in frontier:
                rows = self.conn.execute("""SELECT l.source, l.target, l.strength,
                    l.relation, l.evidence, e.id AS neighbor_id
                    FROM links l JOIN events e ON e.id =
                      CASE WHEN l.source=? THEN l.target ELSE l.source END
                    WHERE l.source=? OR l.target=?
                    ORDER BY l.strength DESC, e.id DESC LIMIT ?""",
                    (current_id, current_id, current_id, beam)).fetchall()
                examined += len(rows)
                for edge in rows:
                    neighbor_id = edge["neighbor_id"]
                    if neighbor_id in path:
                        continue
                    score = activation * edge["strength"] * (0.9 if depth > 1 else 1)
                    if score < 0.1:
                        continue
                    link = {"from_id": current_id, "to_id": neighbor_id,
                            "relation": edge["relation"],
                            "edge_strength": round(edge["strength"], 4),
                            "evidence_cues": json.loads(edge["evidence"])}
                    candidate = (neighbor_id, score, path + (neighbor_id,),
                                 path_links + (link,))
                    if (neighbor_id not in next_frontier or
                            score > next_frontier[neighbor_id][1]):
                        next_frontier[neighbor_id] = candidate
                    if neighbor_id not in winners or score > winners[neighbor_id][1]:
                        winners[neighbor_id] = candidate
                    contexts.append({"from_id": current_id, "to_id": neighbor_id,
                                     "relation": edge["relation"],
                                     "hop": depth})
            frontier = sorted(next_frontier.values(),
                              key=lambda c: (-c[1], -c[0]))[:beam]
            if not frontier:
                break
        selected = sorted(winners.values(), key=lambda c: (-c[1], -c[0]))[:limit]
        episodes = {}
        if selected:
            marks = ",".join("?" for _ in selected)
            for event in self.conn.execute(f"SELECT * FROM events WHERE id IN ({marks})",
                                           [entry[0] for entry in selected]):
                episodes[event["id"]] = self._episode(event)
        related = []
        for event_id, score, path, path_links in selected:
            event = episodes[event_id]
            related.append({"id": event.id, "identity": event.identity,
                            "source": event.source, "relation": path_links[-1]["relation"],
                            "association_score": round(score, 4), "hops": len(path_links),
                            "path": list(path), "supporting_links": list(path_links),
                            "evidence_cues": path_links[-1]["evidence_cues"]})
        return {"seed": cue, "related": related, "examined": examined,
                "phases": {"detail_cue": [cue], "broader_context": contexts,
                           "detail_reinstatement": [
                               {"id": r["id"], "identity": r["identity"],
                                "source": r["source"], "via_path": r["path"]}
                               for r in related]}}

    def associate(self, actor: str, item: str, limit: int = 5) -> dict:
        """Legacy API name; delegates to the single associative retrieval owner."""
        return self.reactivate(actor, item, limit=limit)

    def _update_novelty(self, identity: str) -> None:
        habit = self.conn.execute("""SELECT location FROM events
            WHERE identity=? AND kind='habit' ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
        if habit:
            self.conn.execute("""UPDATE events SET novelty=(location != ?)
                WHERE identity=? AND kind='location'""", (habit["location"], identity))

    def recall(self, query: str, limit: int = 3) -> dict:
        """Answers a bounded query grammar, safely abstaining otherwise."""
        requested = re.fullmatch(
            r"What is (?:connected|associated|related) to "
            r"(?P<actor>[A-Z][a-z]+)'s (?P<item>[a-z][a-z -]*?)\?",
            query.strip(), re.I)
        if requested:
            identity = f"{requested['actor'].lower()}::{requested['item'].lower()}"
            state = self.reactivate(requested["actor"], requested["item"], limit=20)
            related = next((r for r in state["related"] if r["identity"] != identity), None)
            return {"answer": related["source"] if related else None,
                    "reason": None if related else "No stored associative path",
                    "activated": [related] if related else [],
                    "examined": state["examined"], "phases": state["phases"]}

        connected = re.fullmatch(
            r"How are (?P<a>[A-Z][a-z]+)'s (?P<ia>[a-z][a-z -]*?) "
            r"connected to (?P<b>[A-Z][a-z]+)'s (?P<ib>[a-z][a-z -]*?)\?",
            query.strip(), re.I)
        if connected:
            subject = f"{connected['a'].lower()}::{connected['ia'].lower()}"
            target = f"{connected['b'].lower()}::{connected['ib'].lower()}"
            state = self.reactivate(connected["a"], connected["ia"],
                                    limit=20, hops=2, beam=12)
            related = next((r for r in state["related"]
                            if r["identity"] == target and target != subject), None)
            return {"answer": "Connected via recorded associations only, not "
                    "evidence of causation, co-presence, or shared identity."
                    if related else None,
                    "reason": None if related else "No stored associative path",
                    "activated": [related] if related else [],
                    "examined": state["examined"], "phases": state["phases"]}
        match = re.fullmatch(
            r"(?P<prompt>Where (?:are|were|is|was)|What is unusual about|Why did) "
            r"(?P<actor>[A-Z][a-z]+)'s (?P<item>[a-z][a-z -]*?)"
            r"(?: (?:now|before|currently))?\?", query.strip(), re.I)
        if not match:
            return {"answer": None, "reason": "Unsupported question", "activated": [], "examined": 0}
        actor = match["actor"].title()
        item = match["item"].lower()
        identity = f"{actor.lower()}::{item}"
        purpose = match["prompt"].lower()
        if purpose.startswith("why"):
            return {"answer": None, "reason": "No verified cause was recorded",
                    "activated": [], "examined": 0}
        if purpose.startswith("where"):
            # Bounded indexed lookup: only two latest location states for this identity.
            rows = self.conn.execute("""SELECT * FROM events WHERE identity=? AND kind='location'
                ORDER BY id DESC LIMIT 2""", (identity,)).fetchall()
            before = bool(re.search(r"\bbefore\?", query, re.I)) or purpose.startswith("where were") or purpose.startswith("where was")
            target_index = 1 if before else 0
            selected = self._episode(rows[target_index]) if len(rows) > target_index else None
            answer = selected.location if selected else None
            seeds = [selected] if selected else []
        else:
            habit = self.conn.execute("""SELECT * FROM events WHERE identity=? AND kind='habit'
                ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
            recent = self.conn.execute("""SELECT * FROM events WHERE identity=? AND kind='location'
                ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
            latest = self._episode(recent) if recent else None
            normal = self._episode(habit) if habit else None
            answer = (f"{latest.location} differs from usual {normal.location}"
                      if latest and normal and latest.novelty else None)
            seeds = [v for v in (latest, normal) if v is not None]
            rows = [v for v in (recent, habit) if v is not None]
        # One-hop bounded graph expansion from verified seeds, never global scan.
        ids = {x.id for x in seeds}
        activated = [{"id": x.id, "source": x.source, "strength": 1.0} for x in seeds]
        for seed in seeds:
            if len(activated) >= limit:
                break
            neighbors = self.conn.execute("""SELECT e.*, l.strength FROM links l
                JOIN events e ON e.id=l.target WHERE l.source=? AND e.identity=?
                ORDER BY l.strength DESC LIMIT ?""", (seed.id, identity, limit)).fetchall()
            for neighbor in neighbors:
                if neighbor["id"] not in ids and len(activated) < limit:
                    ids.add(neighbor["id"])
                    activated.append({"id": neighbor["id"], "source": neighbor["source"],
                                      "strength": round(neighbor["strength"], 4)})
        return {"answer": answer, "reason": None if answer else "Insufficient recorded evidence",
                "activated": activated, "examined": len(rows) + len(activated) - len(seeds)}

    def counts(self) -> dict[str, int]:
        return {table: self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("events", "links")}


def encode_json(entries: list[Episode]) -> str:
    return json.dumps([asdict(x) | {"identity": x.identity} for x in entries], indent=2)
