"""Post-loop learning: the teacher for turn N is the user's turn N+1.

Outcome is scored in code from the new turn's sensor tags. DA = outcome - expected.
Only the edges used on turn N are updated. Silence is never scored as success.
"""
from __future__ import annotations

import json
from typing import Optional

import config
import layers


def score_outcome(sensor: dict, sim_to_prev: float, prev_valence: Optional[float],
                  embedder_name: str) -> float:
    if sensor["is_correction"]:
        return -1.0
    if sim_to_prev >= config.REPEAT_SIMILARITY[embedder_name]:
        return -1.0   # the same question again: the answer did not land
    v, a = sensor["affect"]["valence"], sensor["affect"]["arousal"]
    if prev_valence is not None and prev_valence - v >= config.DISTRESS_DROP \
            and a >= config.DISTRESS_AROUSAL:
        return -1.0
    if sensor["builds_on_previous"]:
        return 1.0
    return config.NEUTRAL_OUTCOME   # moved on without complaint


def learn(conn, prev_turn, outcome: float, ach: float, now: float) -> float:
    """Update the edges used on the previous turn. Returns DA."""
    edges = json.loads(prev_turn["edges_used"] or "[]")
    expected = prev_turn["expected_outcome"]
    conn.execute("UPDATE turns SET outcome=? WHERE id=?", (outcome, prev_turn["id"]))
    if not edges or expected is None:
        return 0.0
    da = outcome - expected
    lr = config.EDGE_LR_BASE * (0.5 + ach)
    for cue_id, location_id in edges:
        row = conn.execute("SELECT e.weight, l.kind FROM edges e JOIN locations l "
                           "ON l.id = e.location_id WHERE e.cue_id=? AND e.location_id=?",
                           (cue_id, location_id)).fetchone()
        if row is None:
            continue
        weight = max(0.0, min(1.0, row["weight"] + lr * da))
        if row["kind"] == "correction":
            weight = max(weight, config.CORRECTION_WEIGHT_FLOOR)
        conn.execute("UPDATE edges SET weight=?, uses=uses+1, successes=successes+?, "
                     "updated_at=? WHERE cue_id=? AND location_id=?",
                     (weight, 1 if outcome > 0 else 0, now, cue_id, location_id))
    if outcome >= 0:
        used = [r["ref_id"] for r in conn.execute(
            "SELECT ref_id FROM locations WHERE kind='correction' AND id IN (%s)"
            % ",".join("?" * len(edges)), [loc for _, loc in edges])]
        layers.confirm_corrections(conn, used)
    return da
