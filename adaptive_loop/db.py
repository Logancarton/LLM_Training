"""SQLite storage: one graph that acts as the hippocampal index.

The graph stores WHERE answers live (locations), not the answers themselves.
A location of kind 'llm_general' stores nothing: it means "the model already knows this".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS cues (
    id INTEGER PRIMARY KEY, text TEXT NOT NULL, embedding TEXT NOT NULL,
    first_seen REAL, last_seen REAL, hit_count INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, ref_id INTEGER, summary TEXT);
CREATE TABLE IF NOT EXISTS edges (
    cue_id INTEGER NOT NULL, location_id INTEGER NOT NULL, weight REAL NOT NULL,
    uses INTEGER DEFAULT 0, successes INTEGER DEFAULT 0, updated_at REAL,
    PRIMARY KEY (cue_id, location_id));
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, timestamp REAL, summary TEXT, salience REAL,
    heat REAL, half_life REAL);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY, cue_id INTEGER, wrong_text TEXT, right_text TEXT,
    created_at REAL, confirmations INTEGER DEFAULT 0, active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS layer_state (
    layer TEXT NOT NULL, key TEXT NOT NULL, value REAL, baseline REAL,
    half_life REAL, updated_at REAL, PRIMARY KEY (layer, key));
CREATE TABLE IF NOT EXISTS window_items (
    key TEXT PRIMARY KEY, text TEXT, cue_id INTEGER, heat REAL,
    reheats INTEGER DEFAULT 0, updated_at REAL);
CREATE TABLE IF NOT EXISTS modulators (
    turn_id INTEGER PRIMARY KEY, da REAL, ne REAL, ht REAL, ach REAL, timestamp REAL);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY, timestamp REAL, user_text TEXT, sensor_json TEXT,
    cue_id INTEGER, brief TEXT, action TEXT, action_scores TEXT, response TEXT,
    edges_used TEXT, expected_outcome REAL, outcome REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_edges_location ON edges(location_id);
CREATE INDEX IF NOT EXISTS idx_corrections_cue ON corrections(cue_id);
"""

TABLES = ("cues", "locations", "edges", "episodes", "corrections",
          "layer_state", "window_items", "modulators", "turns", "meta")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def llm_general_id(conn: sqlite3.Connection) -> int:
    """The single location that means 'the model already knows this'."""
    row = conn.execute("SELECT id FROM locations WHERE kind='llm_general'").fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO locations(kind, summary) VALUES('llm_general', "
                       "'general knowledge: nothing stored')")
    return cur.lastrowid


def add_location(conn, kind: str, ref_id: Optional[int], summary: str) -> int:
    cur = conn.execute("INSERT INTO locations(kind, ref_id, summary) VALUES(?,?,?)",
                       (kind, ref_id, summary))
    return cur.lastrowid


def upsert_edge(conn, cue_id: int, location_id: int, weight: float, now: float) -> None:
    conn.execute("INSERT INTO edges(cue_id, location_id, weight, updated_at) VALUES(?,?,?,?) "
                 "ON CONFLICT(cue_id, location_id) DO NOTHING",
                 (cue_id, location_id, weight, now))


def add_episode(conn, summary: str, salience: float, now: float, cue_id: Optional[int],
                edge_weight: float) -> int:
    cur = conn.execute("INSERT INTO episodes(timestamp, summary, salience, heat, half_life) "
                       "VALUES(?,?,?,1.0,NULL)", (now, summary, salience))
    loc = add_location(conn, "episode", cur.lastrowid, summary)
    if cue_id is not None:
        upsert_edge(conn, cue_id, loc, edge_weight, now)
    return cur.lastrowid
