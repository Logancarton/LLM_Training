"""The three v0 layers. Each learns only the residual the LLM cannot supply.

Affect (amygdala / insula): this user's baseline and what shifts it.
Temporal window (prefrontal working memory): what is hot right now.
Correction (cerebellum): where the LLM was wrong for this user, and what is right.

Every update here takes fields from the user's own turn. Model output never enters.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config
import db

# --- Affect ------------------------------------------------------------------


def affect_load(conn) -> Optional[Dict[str, float]]:
    rows = {r["key"]: r for r in conn.execute(
        "SELECT key, value, baseline, updated_at FROM layer_state WHERE layer='affect'")}
    if "valence" not in rows:
        return None
    return {"valence": rows["valence"]["value"], "valence_base": rows["valence"]["baseline"],
            "arousal": rows["arousal"]["value"], "arousal_base": rows["arousal"]["baseline"]}


def _affect_save(conn, state: Dict[str, float], now: float) -> None:
    for key in ("valence", "arousal"):
        conn.execute(
            "INSERT INTO layer_state(layer, key, value, baseline, half_life, updated_at) "
            "VALUES('affect', ?, ?, ?, ?, ?) ON CONFLICT(layer, key) DO UPDATE SET "
            "value=excluded.value, baseline=excluded.baseline, updated_at=excluded.updated_at",
            (key, state[key], state[key + "_base"], config.AFFECT_HALF_LIFE_MINUTES, now))


def affect_update(conn, valence: float, arousal: float, label: str, ach: float,
                  cue_id: int, cue_text: str, now: float) -> dict:
    state = affect_load(conn)
    if state is None:   # first observation sets the baseline
        state = {"valence": valence, "valence_base": valence,
                 "arousal": arousal, "arousal_base": arousal}
    else:
        fast = config.AFFECT_FAST_RATE_BASE + config.AFFECT_FAST_RATE_ACH * ach
        state["valence_base"] += config.AFFECT_BASELINE_RATE * (valence - state["valence_base"])
        state["arousal_base"] += config.AFFECT_BASELINE_RATE * (arousal - state["arousal_base"])
        state["valence"] += fast * (valence - state["valence"])
        state["arousal"] += fast * (arousal - state["arousal"])
    _affect_save(conn, state, now)
    shift = state["valence"] - state["valence_base"]
    triggered = abs(shift) > config.AFFECT_SHIFT_THRESHOLD
    if triggered:   # link this situation to an affect shift: a learned trigger
        db.add_episode(conn, f"Affect shift ({label}, {shift:+.2f}) after: {cue_text[:160]}",
                       min(1.0, abs(shift)), now, cue_id, config.EPISODE_EDGE_WEIGHT)
    return {**state, "shift": shift, "triggered": triggered, "distress": distress(state)}


def distress(state: Optional[Dict[str, float]]) -> float:
    if not state:
        return 0.0
    return max(0.0, state["valence_base"] - state["valence"])


def affect_decay_minutes(conn, minutes: float, now: float) -> None:
    state = affect_load(conn)
    if state is None or minutes <= 0:
        return
    factor = 0.5 ** (minutes / config.AFFECT_HALF_LIFE_MINUTES)
    for key in ("valence", "arousal"):
        base = state[key + "_base"]
        state[key] = base + (state[key] - base) * factor
    _affect_save(conn, state, now)


# --- Temporal window ---------------------------------------------------------


def window_observe(conn, items: List[dict], ne: float, now: float) -> List[str]:
    """Decay one turn, reheat or add items, evict past capacity. Returns promoted summaries."""
    conn.execute("UPDATE window_items SET heat = heat * ?", (config.WINDOW_DECAY_PER_TURN,))
    for item in items:
        row = conn.execute("SELECT heat FROM window_items WHERE key=?", (item["key"],)).fetchone()
        if row:
            conn.execute("UPDATE window_items SET heat=MIN(1.0, heat + ?), reheats=reheats+1, "
                         "cue_id=?, updated_at=? WHERE key=?",
                         (config.WINDOW_REHEAT, item["cue_id"], now, item["key"]))
        else:
            conn.execute("INSERT INTO window_items(key, text, cue_id, heat, reheats, updated_at) "
                         "VALUES(?,?,?,1.0,0,?)", (item["key"], item["text"], item["cue_id"], now))
    capacity = config.WINDOW_CAP_BASE + round(config.WINDOW_CAP_NE * ne)
    return _evict(conn, capacity, now)


def _evict(conn, capacity: int, now: float) -> List[str]:
    rows = conn.execute("SELECT key, text, cue_id, heat, reheats FROM window_items "
                        "ORDER BY heat DESC").fetchall()
    promoted = []
    for i, r in enumerate(rows):
        if i < capacity and r["heat"] >= config.WINDOW_MIN_HEAT:
            continue
        if r["reheats"] >= config.WINDOW_PROMOTE_REHEATS:   # recurred: worth remembering
            summary = f"Recurring topic: {r['text']}"
            db.add_episode(conn, summary, min(1.0, 0.2 * r["reheats"]), now, r["cue_id"],
                           config.EPISODE_EDGE_WEIGHT)
            promoted.append(summary)
        conn.execute("DELETE FROM window_items WHERE key=?", (r["key"],))
    return promoted


def window_decay_minutes(conn, minutes: float, now: float) -> List[str]:
    if minutes <= 0:
        return []
    factor = 0.5 ** (minutes / config.WINDOW_HALF_LIFE_MINUTES)
    conn.execute("UPDATE window_items SET heat = heat * ?", (factor,))
    return _evict(conn, 10 ** 6, now)   # only heat below the minimum is evicted here


def window_hot(conn, n: int) -> List[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT key, text, heat, reheats FROM window_items ORDER BY heat DESC LIMIT ?", (n,))]


# --- Correction --------------------------------------------------------------


def capture_correction(conn, cue_id: int, wrong: str, right: str, now: float) -> int:
    """Store what the user said is right. A newer correction on the same cue replaces the old
    one; the old one stays in history (inactive)."""
    conn.execute("UPDATE corrections SET active=0 WHERE cue_id=? AND active=1", (cue_id,))
    cur = conn.execute("INSERT INTO corrections(cue_id, wrong_text, right_text, created_at) "
                       "VALUES(?,?,?,?)", (cue_id, wrong, right, now))
    loc = db.add_location(conn, "correction", cur.lastrowid, f"correction: {right[:120]}")
    db.upsert_edge(conn, cue_id, loc, config.CORRECTION_EDGE_WEIGHT, now)
    return cur.lastrowid


def confirm_corrections(conn, correction_ids: List[int]) -> None:
    for cid in correction_ids:
        conn.execute("UPDATE corrections SET confirmations = confirmations + 1 WHERE id=?",
                     (cid,))
