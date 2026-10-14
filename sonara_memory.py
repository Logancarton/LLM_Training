"""
sonara_memory.py — Structured Episodic & Semantic Memory System for Sonara
=============================================================================

PURPOSE
    Provides Sonara with persistent, timestamped, retrievable memory:
      1. EPISODIC MEMORY: Timestamped logs of events, observations, and conversations.
      2. SEMANTIC MEMORY: Consolidated facts, concepts, and relationships.
      3. PREFERENCE MEMORY: User preferences, habits, and communication styles.
      4. REFLECTION MEMORY: Self-synthesized insights generated during deliberation.

FEATURES
    - High-resolution ISO-8601 timestamps, local readable formats, and relative times ("3 minutes ago").
    - Cognitive retrieval: Composite scoring balancing Content Relevance, Recency Decay, and Importance.
    - Semantic Frame mapping: Aligned with Sonara's IdeaReader (WHO / ACTION / WHAT / WHERE / WHEN).
    - Entity indexing for fast multi-hop or associative recall.
    - Memory Consolidation / Reflection: Turns clusters of episodic events into durable knowledge.
    - Persistent storage: Append-only JSONL log + Human-readable Markdown Journal.
    - Interactive CLI and drop-in integration with LLM_Training.py sessions.
"""

import os
import re
import json
import time
import math
import uuid
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple


# =====================================================================
# 1. TIME UTILITIES
# =====================================================================

def get_timestamps() -> Tuple[str, float, str]:
    """Return (iso_format, epoch_seconds, human_readable_local)."""
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    iso = now.isoformat(timespec="seconds")
    epoch = now.timestamp()
    human = now.strftime("%b %d, %Y at %I:%M:%S %p %Z")
    return iso, epoch, human


def format_relative_time(epoch_time: float) -> str:
    """Convert an epoch timestamp into a human-friendly relative string."""
    delta = time.time() - epoch_time
    if delta < 0:
        return "in the future"
    if delta < 45:
        return "just now"
    if delta < 90:
        return "1 minute ago"
    if delta < 3600:
        mins = int(delta / 60)
        return f"{mins} minutes ago"
    if delta < 7200:
        return "1 hour ago"
    if delta < 86400:
        hours = int(delta / 3600)
        return f"{hours} hours ago"
    if delta < 172800:
        return "yesterday"
    days = int(delta / 86400)
    if days < 30:
        return f"{days} days ago"
    months = int(days / 30)
    return f"{months} month{'s' if months > 1 else ''} ago"


# =====================================================================
# 2. MEMORY DATA MODEL
# =====================================================================

@dataclass
class Memory:
    id: str
    timestamp_iso: str
    timestamp_epoch: float
    timestamp_human: str
    content: str
    category: str = "episodic"       # episodic, semantic, preference, reflection
    source: str = "user_observation" # user_observation, user_explicit, sonara_reflection, conversation
    importance: float = 5.0          # 1.0 (trivial) to 10.0 (foundational)
    entities: List[str] = field(default_factory=list)
    semantic_frame: Dict[str, str] = field(default_factory=dict) # who, action, what, where, when
    access_count: int = 0
    last_accessed_iso: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(**data)

    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp_epoch)

    def relative_time(self) -> str:
        return format_relative_time(self.timestamp_epoch)


# =====================================================================
# 3. SEMANTIC & ENTITY PARSING (LIGHTWEIGHT, LOCAL)
# =====================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "that", "this", "it", "my",
    "your", "his", "her", "their", "our", "me", "you", "him", "them", "us"
}


def extract_entities(text: str) -> List[str]:
    """Extract key nouns, proper nouns, and salient concepts."""
    # Find words with capitalization or significant content
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", text)
    entities = []
    for token in tokens:
        lower = token.lower()
        if lower not in STOPWORDS and len(lower) > 2:
            # Preserve proper nouns or normalized unique tokens
            entities.append(token if token[0].isupper() else lower)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result[:8]


def infer_semantic_frame(text: str) -> Dict[str, str]:
    """
    Attempt to map text into Sonara's standard IdeaReader frame:
    WHO / ACTION / WHAT / WHERE / WHEN
    """
    frame = {"who": "none", "action": "none", "what": "none", "where": "none", "when": "none"}
    cleaned = text.strip().rstrip(".!?")

    # If the user supplied explicit pipe format: "Jake | chases | the ball"
    if "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|")]
        if len(parts) >= 2:
            frame["who"] = parts[0]
            frame["action"] = parts[1]
        if len(parts) >= 3:
            frame["what"] = parts[2]
        return frame

    tokens = cleaned.split()
    if not tokens:
        return frame

    # Heuristic parsing for simple English expressions
    first_two = " ".join(tokens[:2]).lower()
    if first_two.startswith("the ") or first_two.startswith("my "):
        if len(tokens) >= 3:
            frame["who"] = tokens[1]
            frame["action"] = tokens[2]
            if len(tokens) >= 4:
                frame["what"] = " ".join(tokens[3:])
    else:
        frame["who"] = tokens[0]
        if len(tokens) >= 2:
            frame["action"] = tokens[1]
        if len(tokens) >= 3:
            frame["what"] = " ".join(tokens[2:])

    return frame


# =====================================================================
# 4. SONARA MEMORY SYSTEM
# =====================================================================

class SonaraMemorySystem:
    """
    Complete persistent memory manager for Sonara.
    """

    def __init__(self, memory_dir: str = "sonara_memories"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.memory_dir / "memories.jsonl"
        self.journal_file = self.memory_dir / "sonara_memory_journal.md"
        self.memories: Dict[str, Memory] = {}
        self.entity_index: Dict[str, List[str]] = {}
        self._load()

    def _load(self):
        """Load all memories from the append-only JSONL log."""
        self.memories.clear()
        self.entity_index.clear()
        if not self.memories_file.exists():
            return

        with self.memories_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    mem = Memory.from_dict(data)
                    self.memories[mem.id] = mem
                    self._index_memory(mem)
                except Exception as e:
                    print(f"[SonaraMemory] Warning: skipping corrupted record: {e}")

    def _index_memory(self, mem: Memory):
        """Index memory by its entities for fast associative lookup."""
        # Reuse one index for both legacy extracted names and explicitly
        # asserted world-state entities; no shadow state database.
        indexed = set(mem.entities)
        world_entity = mem.semantic_frame.get("entity")
        if world_entity:
            indexed.add(world_entity)
        for ent in indexed:
            key = ent.lower()
            if key not in self.entity_index:
                self.entity_index[key] = []
            if mem.id not in self.entity_index[key]:
                self.entity_index[key].append(mem.id)

    def _save_record(self, mem: Memory):
        """Persist a memory to the append-only JSONL file."""
        with self.memories_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")
        self._index_memory(mem)
        self.export_journal()

    # -----------------------------------------------------------------
    # CORE API: REMEMBER / SAVE
    # -----------------------------------------------------------------

    def remember(
        self,
        content: str,
        category: str = "episodic",
        source: str = "user_observation",
        importance: float = 5.0,
        entities: Optional[List[str]] = None,
        semantic_frame: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> Memory:
        """
        Save a new memory with automatic timestamps, entities, and frame.
        """
        iso, epoch, human = get_timestamps()
        unique_id = f"mem_{int(epoch)}_{uuid.uuid4().hex[:6]}"

        if entities is None:
            entities = extract_entities(content)
        if semantic_frame is None:
            semantic_frame = infer_semantic_frame(content)

        # Automatic importance boost for explicit user preferences or core facts
        lower_content = content.lower()
        if any(w in lower_content for w in ["always", "never", "prefer", "favorite", "love", "hate", "important"]):
            importance = max(importance, 8.0)
        if category == "preference":
            importance = max(importance, 8.5)

        mem = Memory(
            id=unique_id,
            timestamp_iso=iso,
            timestamp_epoch=epoch,
            timestamp_human=human,
            content=content.strip(),
            category=category,
            source=source,
            importance=float(importance),
            entities=entities,
            semantic_frame=semantic_frame,
            access_count=0,
            session_id=session_id,
        )

        self.memories[mem.id] = mem
        self._save_record(mem)
        return mem

    # -----------------------------------------------------------------
    # CORE API: RECALL / SEARCH
    # -----------------------------------------------------------------

    def recall(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.15,
        recency_weight: float = 0.25,
        importance_weight: float = 0.20,
        relevance_weight: float = 0.55,
        half_life_hours: float = 48.0,
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories using cognitive multi-factor scoring:
          Score = w_rel * Relevance + w_rec * Recency + w_imp * Importance
        """
        if not self.memories:
            return []

        query_tokens = set(re.findall(r"\b[A-Za-z0-9_-]+\b", query.lower())) - STOPWORDS
        results = []

        half_life_sec = half_life_hours * 3600.0
        decay_constant = math.log(2) / max(half_life_sec, 1.0)

        for mem in self.memories.values():
            if category and mem.category != category:
                continue

            # 1. Relevance: Token overlap + Entity match + Substring
            mem_tokens = set(re.findall(r"\b[A-Za-z0-9_-]+\b", mem.content.lower())) - STOPWORDS
            overlap = query_tokens.intersection(mem_tokens)
            jaccard = len(overlap) / max(len(query_tokens.union(mem_tokens)), 1)

            # Direct entity or substring match bonus
            bonus = 0.0
            query_lower = query.lower()
            if query_lower in mem.content.lower():
                bonus += 0.4
            for ent in mem.entities:
                if ent.lower() in query_lower:
                    bonus += 0.3
            for frame_val in mem.semantic_frame.values():
                if frame_val != "none" and frame_val.lower() in query_tokens:
                    bonus += 0.2

            relevance = min(1.0, jaccard + bonus)
            if relevance <= 0.05 and not overlap:
                continue

            # 2. Recency: Exponential decay
            age_sec = mem.age_seconds()
            recency = math.exp(-decay_constant * age_sec)

            # 3. Importance: Normalized 0 to 1
            norm_importance = min(1.0, max(0.0, mem.importance / 10.0))

            # Composite Score
            score = (
                relevance_weight * relevance
                + recency_weight * recency
                + importance_weight * norm_importance
            )

            if score >= min_score:
                results.append((mem, score))

        # Sort highest score first
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:limit]

        # Update access count and timestamp for retrieved memories
        iso_now, _, _ = get_timestamps()
        for mem, _ in top:
            mem.access_count += 1
            mem.last_accessed_iso = iso_now

        return top

    # -----------------------------------------------------------------
    # TIMELINE & RECENT MEMORIES
    # -----------------------------------------------------------------

    def get_recent(self, limit: int = 10, category: Optional[str] = None) -> List[Memory]:
        """Return the most recent memories chronologically descending."""
        mems = list(self.memories.values())
        if category:
            mems = [m for m in mems if m.category == category]
        mems.sort(key=lambda m: m.timestamp_epoch, reverse=True)
        return mems[:limit]

    def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Structured timeline with timestamps, relative time, and content."""
        recent = self.get_recent(limit=limit)
        timeline = []
        for m in reversed(recent):
            timeline.append({
                "id": m.id,
                "timestamp_iso": m.timestamp_iso,
                "timestamp_human": m.timestamp_human,
                "relative_time": m.relative_time(),
                "category": m.category,
                "importance": m.importance,
                "content": m.content,
                "entities": m.entities,
            })
        return timeline

    # -----------------------------------------------------------------
    # COGNITIVE CONSOLIDATION / REFLECTION
    # -----------------------------------------------------------------

    def reflect(self) -> List[Memory]:
        """
        Consolidates recent episodic memories into semantic knowledge.
        Finds repeated entities, patterns, and themes.
        """
        recent = self.get_recent(limit=30, category="episodic")
        if len(recent) < 2:
            return []

        entity_counts: Dict[str, List[Memory]] = {}
        for m in recent:
            for ent in m.entities:
                key = ent.lower()
                entity_counts.setdefault(key, []).append(m)

        new_reflections = []
        # Find entities appearing in multiple episodic memories
        for ent_name, cluster in entity_counts.items():
            if len(cluster) >= 2:
                # Check if we already reflected on this recently
                already_reflected = any(
                    ent_name in r.entities for r in self.memories.values() if r.category == "reflection"
                )
                if already_reflected:
                    continue

                summary_snippets = [c.content for c in cluster[:3]]
                reflection_content = (
                    f"Sonara observes frequent experiences with '{ent_name}': "
                    + "; ".join(summary_snippets)
                )

                ref_mem = self.remember(
                    content=reflection_content,
                    category="reflection",
                    source="sonara_reflection",
                    importance=7.5,
                    entities=[ent_name],
                )
                new_reflections.append(ref_mem)

        return new_reflections

    # -----------------------------------------------------------------
    # MARKDOWN JOURNAL EXPORT
    # -----------------------------------------------------------------

    def export_journal(self, output_path: Optional[Path] = None) -> Path:
        """
        Renders an elegant Markdown journal of Sonara's memories.
        """
        path = output_path or self.journal_file
        all_mems = list(self.memories.values())
        all_mems.sort(key=lambda m: m.timestamp_epoch, reverse=True)

        categories: Dict[str, int] = {}
        for m in all_mems:
            categories[m.category] = categories.get(m.category, 0) + 1

        lines = [
            "# Sonara's Memory Journal",
            "",
            f"> *Last synchronized: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Memory Overview",
            f"- **Total Memories**: {len(all_mems)}",
            f"- **Episodic (Experiences)**: {categories.get('episodic', 0)}",
            f"- **Semantic (Facts & Knowledge)**: {categories.get('semantic', 0)}",
            f"- **Preferences & Habits**: {categories.get('preference', 0)}",
            f"- **Reflections & Deliberations**: {categories.get('reflection', 0)}",
            f"- **Unique Entities Known**: {len(self.entity_index)}",
            "",
            "---",
            "",
            "## Timeline of Recent Memories",
            "",
        ]

        if not all_mems:
            lines.append("*Sonara has no recorded memories yet.*")
        else:
            for m in all_mems[:25]:
                badge = f"**[{m.category.upper()}]**"
                entities_str = f" `Entities: {', '.join(m.entities)}`" if m.entities else ""
                lines.append(f"### {m.timestamp_human} *({m.relative_time()})*")
                lines.append(f"- {badge} {m.content}")
                lines.append(f"- *ID: `{m.id}` | Importance: {m.importance}/10{entities_str}*")
                if m.semantic_frame and m.semantic_frame.get("who") != "none":
                    frame_str = " | ".join(f"{k.upper()}={v}" for k, v in m.semantic_frame.items() if v != "none")
                    lines.append(f"- *Frame: `{frame_str}`*")
                lines.append("")

        lines.extend([
            "---",
            "",
            "## Known Entities & Concept Index",
            "",
        ])

        top_entities = sorted(self.entity_index.items(), key=lambda kv: len(kv[1]), reverse=True)[:15]
        if not top_entities:
            lines.append("*No indexed entities yet.*")
        else:
            for ent, mem_ids in top_entities:
                lines.append(f"- **{ent.capitalize()}**: Referenced in {len(mem_ids)} memor{'ies' if len(mem_ids) != 1 else 'y'}")

        lines.append("")
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        return path


# =====================================================================
# 5. CLI INTERFACE & DEMONSTRATION
# =====================================================================

def interactive_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Sonara Memory System")
    parser.add_argument("--dir", default="sonara_memories", help="Directory where memories are saved")
    parser.add_argument("--remember", default="", help="Save a memory directly")
    parser.add_argument("--recall", default="", help="Query memories")
    parser.add_argument("--timeline", action="store_true", help="Print memory timeline")
    parser.add_argument("--reflect", action="store_true", help="Consolidate recent memories")
    args = parser.parse_args()

    memory = SonaraMemorySystem(args.dir)

    if args.remember:
        mem = memory.remember(args.remember)
        print(f"\n[Sonara Memory Saved]")
        print(f"  ID:        {mem.id}")
        print(f"  Timestamp: {mem.timestamp_human} ({mem.relative_time()})")
        print(f"  Content:   {mem.content}")
        print(f"  Entities:  {mem.entities}")
        print(f"  Frame:     {mem.semantic_frame}")
        return

    if args.recall:
        print(f"\nSearching Sonara's memories for: '{args.recall}'...")
        results = memory.recall(args.recall)
        if not results:
            print("  No relevant memories found.")
        for mem, score in results:
            print(f"  [{score:.2f}] ({mem.relative_time()} | {mem.timestamp_human}) [{mem.category.upper()}] {mem.content}")
        return

    if args.timeline:
        print("\n=== SONARA'S MEMORY TIMELINE ===")
        timeline = memory.get_timeline(limit=15)
        if not timeline:
            print("  No memories recorded yet.")
        for item in timeline:
            print(f"  • {item['timestamp_human']} ({item['relative_time']}) [{item['category']}]")
            print(f"    \"{item['content']}\"")
        return

    if args.reflect:
        reflections = memory.reflect()
        print(f"\nConsolidation complete: Generated {len(reflections)} new reflections.")
        for r in reflections:
            print(f"  - [{r.timestamp_human}] {r.content}")
        return

    # Default: Interactive shell for testing Sonara's memory
    print("=" * 65)
    print("      SONARA MEMORY CONSOLE — Interactive Memory Manager")
    print("=" * 65)
    print(f"Storage path: {memory.memory_dir.resolve()}")
    print("Commands:")
    print("  /remember <text>   - Save an episodic memory")
    print("  /fact <text>       - Save a semantic fact")
    print("  /pref <text>       - Save a user preference")
    print("  /recall <query>    - Search memories with cognitive scoring")
    print("  /timeline          - Show chronological memories")
    print("  /reflect           - Consolidate experiences into insights")
    print("  /quit              - Exit")
    print("=" * 65)

    while True:
        try:
            line = input("\nSonara-Mem> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not line or line.lower() in ("/quit", "/exit"):
            break

        if line.startswith("/remember "):
            text = line[len("/remember "):].strip()
            mem = memory.remember(text, category="episodic")
            print(f"[Saved at {mem.timestamp_human} ({mem.relative_time()})]")
            print(f"  Entities: {mem.entities} | Frame: {mem.semantic_frame}")

        elif line.startswith("/fact "):
            text = line[len("/fact "):].strip()
            mem = memory.remember(text, category="semantic", importance=8.0)
            print(f"[Saved Fact at {mem.timestamp_human}]")

        elif line.startswith("/pref "):
            text = line[len("/pref "):].strip()
            mem = memory.remember(text, category="preference", importance=9.0)
            print(f"[Saved Preference at {mem.timestamp_human}]")

        elif line.startswith("/recall "):
            query = line[len("/recall "):].strip()
            matches = memory.recall(query)
            if not matches:
                print("No matching memories found.")
            else:
                print(f"Found {len(matches)} relevant memories:")
                for m, score in matches:
                    print(f"  [{score:.2f}] {m.timestamp_human} ({m.relative_time()}) [{m.category}] {m.content}")

        elif line == "/timeline":
            items = memory.get_timeline(limit=15)
            if not items:
                print("No memories recorded yet.")
            for it in items:
                print(f"  [{it['timestamp_human']}] ({it['relative_time']}) [{it['category']}] {it['content']}")

        elif line == "/reflect":
            refs = memory.reflect()
            if not refs:
                print("No new reflections could be consolidated (requires at least 2 related episodic events).")
            else:
                print(f"Consolidated {len(refs)} new reflection(s):")
                for r in refs:
                    print(f"  • {r.content}")

        else:
            # Natural input defaults to saving as episodic memory
            mem = memory.remember(line, category="episodic")
            print(f"[Recorded at {mem.timestamp_human} ({mem.relative_time()})]")


if __name__ == "__main__":
    interactive_cli()
