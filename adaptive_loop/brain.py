"""The loop. One user turn runs: post-loop for the previous turn, then the pre-loop.

    sensor -> post-loop(N-1) -> correction capture -> retrieval -> affect -> modulators
    -> window -> action selection -> state brief -> generator -> log

Everything except the two LLM calls is plain arithmetic over SQLite.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import action as act
import brief as brief_mod
import config
import db
import layers
import llm
import modulators as mod
import postloop
import retrieval
from embed import cosine, get_embedder
from evidence import Evidence
from generate import generate
from sensor import sense


class Brain:
    def __init__(self, db_path: Optional[Path] = None, embedder=None,
                 evidence: Optional[Evidence] = None):
        self.evidence = evidence if evidence is not None else Evidence()
        path = Path(db_path or config.DB_PATH)
        self.conn = db.connect(path)          # writes: always under self.lock
        self.read_conn = db.connect(path)     # panel reads: never blocked by a slow LLM call
        self.lock = threading.RLock()
        self.embedder = embedder or get_embedder()
        with self.lock:
            db.llm_general_id(self.conn)
            self._check_embedder()
            if db.get_meta(self.conn, "last_decay") is None:
                db.set_meta(self.conn, "last_decay", time.time())

    # --- housekeeping ---------------------------------------------------------

    def _check_embedder(self) -> None:
        """If the embedder changed, re-embed stored cues so similarity stays meaningful."""
        stored = db.get_meta(self.conn, "embedder")
        if stored and stored != self.embedder.name:
            for row in self.conn.execute("SELECT id, text FROM cues").fetchall():
                self.conn.execute("UPDATE cues SET embedding=? WHERE id=?",
                                  (json.dumps(self.embedder.embed(row["text"])), row["id"]))
        db.set_meta(self.conn, "embedder", self.embedder.name)

    def tick(self, now: Optional[float] = None) -> float:
        """Heartbeat: apply wall-clock decay for all time elapsed since the last decay.
        No LLM calls. Returns the minutes applied (the whole gap after a pause)."""
        now = now or time.time()
        with self.lock:
            last = float(db.get_meta(self.conn, "last_decay", now))
            minutes = max(0.0, (now - last) / 60.0)
            if minutes <= 0:
                return 0.0
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                state = mod.decay_minutes(mod.load(self.conn), minutes)
                mod.save(self.conn, state, now)
                layers.affect_decay_minutes(self.conn, minutes, now)
                layers.window_decay_minutes(self.conn, minutes, now)
                db.set_meta(self.conn, "last_decay", now)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return minutes

    def _new_or_matched_cue(self, r: dict, cue_text: str, vec, now: float) -> int:
        if not r["is_novel"]:
            cue_id = r["matched"][0]["cue_id"]
            self.conn.execute("UPDATE cues SET last_seen=?, hit_count=hit_count+1 WHERE id=?",
                              (now, cue_id))
            return cue_id
        cur = self.conn.execute("INSERT INTO cues(text, embedding, first_seen, last_seen) "
                                "VALUES(?,?,?,?)", (cue_text, json.dumps(vec), now, now))
        db.upsert_edge(self.conn, cur.lastrowid, db.llm_general_id(self.conn),
                       config.LLM_GENERAL_PRIOR, now)
        return cur.lastrowid

    def _history(self):
        rows = self.conn.execute("SELECT user_text, response FROM turns ORDER BY id DESC LIMIT ?",
                                 (config.HISTORY_TURNS,)).fetchall()
        msgs = []
        for r in reversed(rows):
            msgs += [{"role": "user", "content": r["user_text"]},
                     {"role": "assistant", "content": r["response"] or ""}]
        return msgs

    # --- one turn -------------------------------------------------------------

    def handle_turn(self, text: str, now: Optional[float] = None) -> dict:
        with self.lock:
            self.tick(now)
            now = now or time.time()
            prev = self.conn.execute("SELECT * FROM turns ORDER BY id DESC LIMIT 1").fetchone()
            sensor = sense(text, prev["user_text"] if prev else None,
                           prev["response"] if prev else None)
            vec = self.embedder.embed(sensor["cue"])
            history = self._history()
            evidence = self.evidence.recall(text)   # before recording: no self-match
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                result = self._turn(text, sensor, vec, prev, history, now, evidence)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            # Append-only evidence is written only after the turn commits, and only the
            # user's own words.
            result["evidence_id"] = self.evidence.record_user_turn(text)
            return result

    def _turn(self, text, sensor, vec, prev, history, now, evidence) -> dict:
        conn = self.conn
        mods = mod.load(conn)

        # Post-loop for the previous turn: the user's reaction now is its teacher.
        da, outcome, sim_prev, topic_shift = 0.0, None, 0.0, False
        if prev is not None:
            prev_sensor = json.loads(prev["sensor_json"])
            prev_cue = conn.execute("SELECT embedding FROM cues WHERE id=?",
                                    (prev["cue_id"],)).fetchone()
            sim_prev = cosine(vec, json.loads(prev_cue["embedding"])) if prev_cue else 0.0
            topic_shift = sim_prev < config.TOPIC_SHIFT_SIMILARITY[self.embedder.name]
            if prev["outcome"] is None:
                outcome = postloop.score_outcome(sensor, sim_prev,
                                                 prev_sensor["affect"]["valence"],
                                                 self.embedder.name)
                da = postloop.learn(conn, prev, outcome, mods["ach"], now)
            if sensor["is_correction"] and prev["cue_id"] is not None:
                layers.capture_correction(conn, prev["cue_id"], sensor["correction"]["wrong"],
                                          sensor["correction"]["right"] or text, now)

        # Pre-loop.
        r = retrieval.retrieve(conn, self.embedder.name, vec, mods)
        cue_id = self._new_or_matched_cue(r, sensor["cue"], vec, now)
        affect = layers.affect_update(conn, sensor["affect"]["valence"],
                                      sensor["affect"]["arousal"], sensor["affect"]["label"],
                                      mods["ach"], cue_id, sensor["cue"], now)
        mods = mod.update_turn(mods, {
            "novelty": r["novelty"], "surprise": abs(da), "stakes": sensor["stakes"],
            "distress": affect["distress"], "confidence": r["confidence"],
            "is_correction": sensor["is_correction"], "topic_shift": topic_shift})
        mod.save(conn, mods, now)
        promoted = layers.window_observe(
            conn, [{"key": "topic:" + sensor["topic"], "text": sensor["topic"],
                    "cue_id": cue_id}], mods["ne"], now)

        chosen, scores = act.select(r["confidence"], sensor["ambiguity"], mods, retried=False)
        if chosen == act.RETRIEVE_MORE:
            first_novelty = r["novelty"]
            exclude = cue_id if r["is_novel"] else None   # don't match the cue just stored
            r = retrieval.retrieve(conn, self.embedder.name, vec, mods, extra_hop=True,
                                   exclude_cue_id=exclude)
            r["novelty"] = first_novelty
            chosen, retry_scores = act.select(r["confidence"], sensor["ambiguity"], mods,
                                              retried=True)
            scores = {"first_pass": scores, "second_pass": retry_scores}

        state_brief = brief_mod.build(chosen, mods, sensor["stakes"], affect, r["corrections"],
                                      layers.window_hot(conn, config.BRIEF_MAX_HOT),
                                      r["locations"], evidence)
        try:
            response, ok = generate(text, state_brief, chosen, history), True
        except llm.LLMError as exc:
            response, ok = f"[generator unavailable: {exc}]", False

        # Which edges this turn relied on (learned from on the next turn).
        edges_used, expected = [], None
        if ok:
            if r["locations"]:
                top = r["locations"][0]
                edges_used.append([top["cue_id"], top["location_id"]])
                weight = top["weight"]
            else:
                edges_used.append([cue_id, db.llm_general_id(conn)])
                weight = config.LLM_GENERAL_PRIOR
            edges_used += [[c["cue_id"], c["location_id"]] for c in r["corrections"]]
            expected = 2 * weight - 1

        cur = conn.execute(
            "INSERT INTO turns(timestamp, user_text, sensor_json, cue_id, brief, action, "
            "action_scores, response, edges_used, expected_outcome) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (now, text, json.dumps(sensor), cue_id, state_brief, chosen, json.dumps(scores),
             response, json.dumps(edges_used), expected))
        mod.log(conn, cur.lastrowid, da, mods, now)
        return {"turn_id": cur.lastrowid, "response": response, "action": chosen,
                "scores": scores, "brief": state_brief, "modulators": mods, "da": da,
                "previous_outcome": outcome, "sensor": sensor, "novelty": r["novelty"],
                "confidence": r["confidence"], "corrections": len(r["corrections"]),
                "evidence": evidence, "promoted": promoted}

    # --- read-only view for the control panel -----------------------------------

    def snapshot(self) -> dict:
        conn = self.read_conn
        last = conn.execute("SELECT da FROM modulators ORDER BY turn_id DESC LIMIT 1").fetchone()
        recent = [dict(r) for r in conn.execute(
            "SELECT id, timestamp, action, substr(user_text, 1, 80) AS text, outcome "
            "FROM turns ORDER BY id DESC LIMIT 10")]
        last_turn = conn.execute("SELECT MAX(timestamp) AS t FROM turns").fetchone()["t"]
        count = lambda table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        return {
            "modulators": {**mod.load(conn), "da": last["da"] if last else 0.0},
            "affect": layers.affect_load(conn),
            "hot": layers.window_hot(conn, 5),
            "recent": recent,
            "last_turn_at": last_turn,
            "counts": {"turns": count("turns"), "cues": count("cues"),
                       "corrections": count("corrections"), "episodes": count("episodes"),
                       "evidence": self.evidence.count()},
            "embedder": self.embedder.name,
            "providers": {"sensor": llm.resolve_provider(config.SENSOR_PROVIDER),
                          "generator": llm.resolve_provider(config.GENERATOR_PROVIDER)},
            "llm_calls": llm.CALL_COUNT,
        }
