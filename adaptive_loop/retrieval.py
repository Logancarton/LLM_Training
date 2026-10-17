"""Retrieval (hippocampus): find similar past cues and follow their edges to answer locations.

Returns where to look, how novel the situation is, and how confident the system is.
Corrections linked to any matched cue are always returned and outrank everything else.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import config
from embed import cosine


def _all_cues(conn):
    return conn.execute("SELECT id, text, embedding FROM cues").fetchall()


def retrieve(conn, embedder_name: str, cue_vec: List[float], mods: Dict[str, float],
             extra_hop: bool = False, exclude_cue_id: Optional[int] = None) -> dict:
    k = config.RETRIEVAL_K_BASE + round(config.RETRIEVAL_K_RANGE * mods["ne"])
    floor = config.SIMILARITY_FLOOR[embedder_name]
    sims = sorted(((cosine(cue_vec, json.loads(r["embedding"])), r["id"], r["text"])
                   for r in _all_cues(conn) if r["id"] != exclude_cue_id),
                  reverse=True)[:k]
    best_sim = sims[0][0] if sims else 0.0
    matched = [(sim, cid, text) for sim, cid, text in sims if sim >= floor]

    locations: Dict[int, dict] = {}
    for sim, cid, _ in matched:
        for e in conn.execute(
                "SELECT e.location_id, e.weight, l.kind, l.summary, l.ref_id FROM edges e "
                "JOIN locations l ON l.id = e.location_id WHERE e.cue_id=?", (cid,)):
            score = sim * e["weight"]
            current = locations.get(e["location_id"])
            if current is None or score > current["score"]:
                locations[e["location_id"]] = {
                    "location_id": e["location_id"], "cue_id": cid, "weight": e["weight"],
                    "score": score, "kind": e["kind"], "summary": e["summary"],
                    "ref_id": e["ref_id"], "hop": 0}

    hop = bool(locations) and (extra_hop or (mods["ne"] > config.EXTRA_HOP_NE
                                             and mods["ht"] < config.SAFETY_HT))
    if hop:
        top = sorted(locations.values(), key=lambda x: x["score"], reverse=True)[:3]
        for loc in top:
            if loc["kind"] == "llm_general":
                continue   # everything links to general knowledge; hopping through it is noise
            for linked in conn.execute("SELECT cue_id FROM edges WHERE location_id=? "
                                       "AND cue_id != ?", (loc["location_id"], loc["cue_id"])):
                for e in conn.execute(
                        "SELECT e.location_id, e.weight, l.kind, l.summary, l.ref_id "
                        "FROM edges e JOIN locations l ON l.id = e.location_id "
                        "WHERE e.cue_id=?", (linked["cue_id"],)):
                    score = loc["score"] * e["weight"] * config.HOP_ATTENUATION
                    if e["location_id"] not in locations:
                        locations[e["location_id"]] = {
                            "location_id": e["location_id"], "cue_id": linked["cue_id"],
                            "weight": e["weight"], "score": score, "kind": e["kind"],
                            "summary": e["summary"], "ref_id": e["ref_id"], "hop": 1}

    matched_ids = [cid for _, cid, _ in matched]
    corrections = []
    if matched_ids:
        marks = ",".join("?" * len(matched_ids))
        for c in conn.execute(
                f"SELECT c.id, c.cue_id, c.wrong_text, c.right_text, c.confirmations, "
                f"l.id AS location_id, e.weight FROM corrections c JOIN locations l "
                f"ON l.kind='correction' AND l.ref_id=c.id "
                f"JOIN edges e ON e.location_id=l.id AND e.cue_id=c.cue_id "
                f"WHERE c.active=1 AND c.cue_id IN ({marks}) "
                f"ORDER BY c.confirmations DESC, c.created_at DESC", matched_ids):
            corrections.append(dict(c))

    ranked = sorted((l for l in locations.values() if l["kind"] != "correction"),
                    key=lambda x: x["score"], reverse=True)
    confidence = ranked[0]["weight"] if ranked else config.LLM_GENERAL_PRIOR
    if corrections:   # a correction the user gave is the most reliable answer location
        confidence = max(confidence, corrections[0]["weight"])
    return {
        "matched": [{"cue_id": cid, "similarity": round(sim, 3), "text": text}
                    for sim, cid, text in matched],
        "best_sim": best_sim,
        "novelty": 1.0 - best_sim,
        "is_novel": not matched,
        "locations": ranked,
        "corrections": corrections,
        "confidence": confidence,
        "hopped": hop,
        "k": k,
    }
