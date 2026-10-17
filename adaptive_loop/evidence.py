"""Bridge to the repo's evidence store (sonara_memory.py, one folder up).

Hippocampal indexing: the graph in sonara.db points to where answers live; this store holds
the content. It is the same append-only store the LLM_Training.py session writes to, so the
loop and the session share one memory of what the user said.

Rules kept from the repo:
- Only the user's own words are written (source="conversation"). Model output, the brief,
  and system-derived salience (affect shifts, recurring topics) never enter this store.
- Recalled records are quoted as the user's words with their age, never as verified facts.
- Legacy "sonara_reflection" records are skipped (sonara_memory.recall already does this).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import List, Optional

import config

if str(config.REPO_DIR) not in sys.path:
    sys.path.append(str(config.REPO_DIR))
try:
    from sonara_memory import SonaraMemorySystem
except ImportError:   # running outside the repo: the loop still works without evidence
    SonaraMemorySystem = None


class Evidence:
    def __init__(self, memory_dir: Optional[Path] = None, enabled: Optional[bool] = None):
        enabled = config.MEMORY_BRIDGE if enabled is None else enabled
        self.session_id = "loop_" + uuid.uuid4().hex[:8]
        self.store = None
        if enabled and SonaraMemorySystem is not None:
            self.store = SonaraMemorySystem(str(memory_dir or config.MEMORY_DIR))

    @property
    def available(self) -> bool:
        return self.store is not None

    def count(self) -> int:
        return len(self.store.memories) if self.store else 0

    def recall(self, query: str) -> List[dict]:
        """Past user statements relevant to this turn. Call BEFORE recording the turn,
        so a message can never match itself."""
        if not self.store or not query.strip():
            return []
        hits = self.store.recall(query, limit=config.EVIDENCE_LIMIT,
                                 min_score=config.EVIDENCE_MIN_SCORE)
        return [{"id": m.id, "content": m.content, "when": m.relative_time(),
                 "source": m.source, "score": round(score, 3)} for m, score in hits]

    def record_user_turn(self, text: str) -> Optional[str]:
        """Append the user's own words. Nothing else is ever written here."""
        if not self.store or not text.strip():
            return None
        return self.store.remember(text, category="episodic", source="conversation",
                                   session_id=self.session_id).id
