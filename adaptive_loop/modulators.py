"""Neuromodulator bus: four channels that turn the loop's knobs.

DA  (dopamine)        signed prediction error per turn; sets the sign and size of learning
NE  (norepinephrine)  novelty and surprise; widens retrieval (curiosity)
5-HT ("ht", serotonin) stakes, distress, low confidence; raises the answer bar (caution)
ACh (acetylcholine)   corrections and topic shifts; trust in new input over memory

Each turn a tonic channel decays toward baseline, then rises to at least its driven level.
Drives set a floor rather than adding up, so repeated drives cannot ratchet a channel to 1.
"""
from __future__ import annotations

from typing import Dict

import config

CHANNELS = ("ne", "ht", "ach")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def load(conn) -> Dict[str, float]:
    state = {}
    for ch in CHANNELS:
        row = conn.execute("SELECT value FROM layer_state WHERE layer='modulator' AND key=?",
                           (ch,)).fetchone()
        state[ch] = row["value"] if row else config.MOD_BASELINE[ch]
    return state


def save(conn, state: Dict[str, float], now: float) -> None:
    for ch in CHANNELS:
        conn.execute(
            "INSERT INTO layer_state(layer, key, value, baseline, half_life, updated_at) "
            "VALUES('modulator', ?, ?, ?, ?, ?) ON CONFLICT(layer, key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (ch, state[ch], config.MOD_BASELINE[ch], config.MOD_HALF_LIFE_TURNS[ch], now))


def _toward_baseline(value: float, baseline: float, factor: float) -> float:
    return baseline + (value - baseline) * factor


def decay_turns(state: Dict[str, float], turns: float = 1.0) -> Dict[str, float]:
    return {ch: _toward_baseline(state[ch], config.MOD_BASELINE[ch],
                                 0.5 ** (turns / config.MOD_HALF_LIFE_TURNS[ch]))
            for ch in CHANNELS}


def decay_minutes(state: Dict[str, float], minutes: float) -> Dict[str, float]:
    return {ch: _toward_baseline(state[ch], config.MOD_BASELINE[ch],
                                 0.5 ** (minutes / config.MOD_HALF_LIFE_MINUTES[ch]))
            for ch in CHANNELS}


def drive_levels(inputs: dict) -> Dict[str, float]:
    """The level each channel is pushed to this turn by what was perceived."""
    b = config.MOD_BASELINE
    return {
        "ne": b["ne"] + config.NE_NOVELTY_GAIN * inputs.get("novelty", 0.0)
              + config.NE_SURPRISE_GAIN * inputs.get("surprise", 0.0),
        "ht": b["ht"] + config.HT_STAKES_GAIN * inputs.get("stakes", 0.0)
              + config.HT_DISTRESS_GAIN * inputs.get("distress", 0.0)
              + config.HT_LOW_CONFIDENCE_GAIN * (1.0 - inputs.get("confidence", 1.0)),
        "ach": b["ach"] + config.ACH_CORRECTION_GAIN * float(inputs.get("is_correction", False))
               + config.ACH_TOPIC_SHIFT_GAIN * float(inputs.get("topic_shift", False)),
    }


def update_turn(state: Dict[str, float], inputs: dict) -> Dict[str, float]:
    decayed = decay_turns(state)
    drive = drive_levels(inputs)
    new = {}
    for ch in CHANNELS:
        target = max(decayed[ch], drive[ch])
        delta = target - state[ch]
        fast_caution = ch == "ht" and delta > 0 and config.CAUTION_FAST_ONSET
        if not fast_caution:
            delta = max(-config.MOD_MAX_STEP, min(config.MOD_MAX_STEP, delta))
        new[ch] = _clamp(state[ch] + delta)
    return new


def log(conn, turn_id: int, da: float, state: Dict[str, float], now: float) -> None:
    conn.execute("INSERT OR REPLACE INTO modulators(turn_id, da, ne, ht, ach, timestamp) "
                 "VALUES(?,?,?,?,?,?)", (turn_id, da, state["ne"], state["ht"],
                                         state["ach"], now))
