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
    }


class LinkLearner:
    """Logistic association learner. Scores are model estimates, not calibrated truth."""
    def __init__(self, weights: dict[str, float] | None = None, bias: float = 0.0):
        self.weights = weights or {key: 0.0 for key in features(Episode(), Episode())}
        self.bias = bias

    def probability(self, a: Episode, b: Episode) -> float:
        x = self.bias + sum(self.weights[k] * v for k, v in features(a, b).items())
        return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))

    def fit(self, samples: list[tuple[Episode, Episode, int]], epochs: int = 100,
            rate: float = 0.15) -> None:
        if not samples or {label for _, _, label in samples} != {0, 1}:
            raise ValueError("Training needs positive and negative associations")
        for _ in range(epochs):
            for a, b, label in samples:
                predicted = self.probability(a, b)
                error = label - predicted
                vals = features(a, b)
                for k, value in vals.items():
                    self.weights[k] += rate * (error * value - 0.001 * self.weights[k])
                self.bias += rate * error

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"weights": self.weights, "bias": self.bias}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LinkLearner:
        raw = json.loads(Path(path).read_text())
        expected = set(features(Episode(), Episode()))
        if set(raw["weights"]) != expected:
            raise ValueError("Incompatible association model")
        return cls({k: float(v) for k, v in raw["weights"].items()}, float(raw["bias"]))


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
            CREATE TABLE IF NOT EXISTS links (
              source INTEGER NOT NULL, target INTEGER NOT NULL, strength REAL NOT NULL,
              relation TEXT NOT NULL, PRIMARY KEY(source,target)
            );
            CREATE INDEX IF NOT EXISTS link_target ON links(target,strength);
        """)

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
                earlier = [self._episode(row) for row in self.conn.execute(
                    "SELECT * FROM events WHERE identity=? AND identity != '' ORDER BY id DESC LIMIT 12",
                    (entry.identity,))]
                cursor = self.conn.execute("""INSERT INTO events
                    (identity,actor,item,action,location,stated_time,kind,source,novelty)
                    VALUES (?,?,?,?,?,?,?,?,0)""",
                    (entry.identity, entry.actor, entry.item, entry.action, entry.location,
                     entry.stated_time, entry.kind, entry.source))
                entry.id = cursor.lastrowid
                # Hard identity boundary: learned scores cannot link unrelated identities.
                for old in earlier:
                    p = self.learner.probability(entry, old)
                    if p >= self.threshold:
                        self.conn.execute("""INSERT OR REPLACE INTO links
                          (source,target,strength,relation) VALUES (?,?,?,?)""",
                          (entry.id, old.id, p, "temporal-association"))
                if entry.identity:
                    self._update_novelty(entry.identity)
            for entry in entries:
                row = self.conn.execute("SELECT novelty FROM events WHERE id=?", (entry.id,)).fetchone()
                entry.novelty = bool(row["novelty"])
        return entries

    def _update_novelty(self, identity: str) -> None:
        habit = self.conn.execute("""SELECT location FROM events
            WHERE identity=? AND kind='habit' ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
        if habit:
            self.conn.execute("""UPDATE events SET novelty=(location != ?)
                WHERE identity=? AND kind='location'""", (habit["location"], identity))

    def recall(self, query: str, limit: int = 3) -> dict:
        """Answers a bounded query grammar, safely abstaining otherwise."""
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
