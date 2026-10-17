"""Sensor (thalamus): one cheap LLM call turns the raw message into structured tags.

Perception is done by the LLM, so no layer ever learns to perceive.
If the LLM is unavailable (or the provider is "mock"), simple heuristics stand in.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import config
import llm

SYSTEM = """You are the perception stage of a memory system. Read the user's newest message \
(the previous exchange is given only for context) and return ONLY a JSON object with exactly \
these keys:
{"cue": one sentence restating what the newest message is really asking or saying,
 "topic": a 1-4 word label,
 "intent": one of "ask", "tell", "correct", "vent", "decide", "other",
 "affect": {"valence": number -1..1, "arousal": number 0..1, "label": one word},
 "is_correction": true if the user is correcting the previous reply,
 "correction": {"wrong": what the previous reply got wrong, in the user's terms,
                "right": what the user says is right},
 "stakes": number 0..1 (health, money, legal or safety consequences),
 "ambiguity": number 0..1 (how unclear the request is without more information),
 "builds_on_previous": true if the user accepts, thanks, or builds on the previous reply,
 "mentions_open_loop": true if the user refers to an unfinished promise or question}
Describe only what the user said. Do not give advice. Output JSON only."""

INTENTS = ("ask", "tell", "correct", "vent", "decide", "other")

_POS = {"thanks", "thank", "great", "love", "good", "glad", "happy", "awesome", "perfect",
        "helpful", "nice", "excited", "yes", "works", "amazing", "relieved"}
_NEG = {"sad", "angry", "frustrated", "upset", "worried", "anxious", "scared", "hate",
        "terrible", "awful", "annoyed", "tired", "stressed", "overwhelmed", "bad", "hurt",
        "afraid", "confused", "lost", "exhausted"}
_STAKES = ("urgent", "emergency", "medication", "dose", "dosage", "suicid", "overdose",
           "lawsuit", "legal", "contract", "safety", "dangerous", "surgery", "diagnos",
           "chest pain", "money", "payroll", "taxes", "deadline", "insurance", "lithium")
_CORRECTION = ("that's wrong", "that is wrong", "not what i", "you're wrong", "incorrect",
               "i said", "i meant", "not quite", "actually,", "no,", "no.", "wrong")
_BUILDS = ("thanks", "thank you", "great", "perfect", "that works", "got it", "makes sense",
           "ok so", "okay so", "and then", "next,", "that helped", "love it")
_VAGUE = {"it", "that", "this", "they", "them", "those", "one", "thing", "stuff"}


def _clamp(x, lo, hi, default):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


def normalize(d: dict, text: str) -> dict:
    """Force any sensor output into the exact shape and ranges the loop expects."""
    affect = d.get("affect") if isinstance(d.get("affect"), dict) else {}
    corr = d.get("correction") if isinstance(d.get("correction"), dict) else {}
    intent = d.get("intent") if d.get("intent") in INTENTS else "other"
    cue = str(d.get("cue") or text).strip()[:300] or text[:300]
    return {
        "cue": cue,
        "topic": str(d.get("topic") or "general").strip()[:60].lower(),
        "intent": intent,
        "affect": {"valence": _clamp(affect.get("valence"), -1, 1, 0.0),
                   "arousal": _clamp(affect.get("arousal"), 0, 1, 0.3),
                   "label": str(affect.get("label") or "neutral")[:30]},
        "is_correction": bool(d.get("is_correction")) or intent == "correct",
        "correction": {"wrong": str(corr.get("wrong") or "")[:300],
                       "right": str(corr.get("right") or "")[:300]},
        "stakes": _clamp(d.get("stakes"), 0, 1, 0.1),
        "ambiguity": _clamp(d.get("ambiguity"), 0, 1, 0.2),
        "builds_on_previous": bool(d.get("builds_on_previous")),
        "mentions_open_loop": bool(d.get("mentions_open_loop")),
    }


def heuristic(text: str, prev_text: Optional[str] = None) -> dict:
    """Offline stand-in for the sensor. Crude on purpose; the real sensor is the LLM."""
    low = text.lower().strip()
    words = re.findall(r"[a-z']+", low)
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    is_corr = prev_text is not None and any(m in low for m in _CORRECTION)
    question = "?" in low or (words[:1] and words[0] in {"what", "how", "why", "when",
                                                         "where", "who", "which", "can",
                                                         "should", "is", "are", "do", "does"})
    if is_corr:
        intent = "correct"
    elif "should i" in low:
        intent = "decide"
    elif question:
        intent = "ask"
    elif neg >= 2:
        intent = "vent"
    else:
        intent = "tell"
    builds = prev_text is not None and any(b in low for b in _BUILDS)
    vague = len(words) <= 5 and any(w in _VAGUE for w in words) and not builds
    content = [w for w in words if w not in _VAGUE and len(w) > 3]
    return normalize({
        "cue": text.strip(),
        "topic": " ".join(content[:3]) or "general",
        "intent": intent,
        "affect": {"valence": max(-1.0, min(1.0, 0.4 * (pos - neg))),
                   "arousal": min(1.0, 0.2 + 0.2 * (pos + neg) + 0.15 * low.count("!")),
                   "label": "positive" if pos > neg else "negative" if neg > pos else "neutral"},
        "is_correction": is_corr,
        "correction": {"wrong": "", "right": text.strip() if is_corr else ""},
        "stakes": 0.8 if any(s in low for s in _STAKES) else 0.1,
        "ambiguity": 0.75 if vague else 0.2,
        "builds_on_previous": builds,
        "mentions_open_loop": False,
    }, text)


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("no JSON object in sensor output")
    return json.loads(match.group(0))


def sense(text: str, prev_text: Optional[str] = None,
          prev_response: Optional[str] = None) -> dict:
    llm.count_call()
    if llm.resolve_provider(config.SENSOR_PROVIDER) == "mock":
        return heuristic(text, prev_text)
    context = ""
    if prev_text is not None:
        context = (f"Previous user message: {prev_text}\n"
                   f"Previous reply: {(prev_response or '')[:1200]}\n\n")
    try:
        raw = llm.complete(config.SENSOR_PROVIDER, "sensor", SYSTEM,
                           [{"role": "user", "content": context + "Newest message: " + text}],
                           config.SENSOR_MAX_TOKENS, json_mode=True)
        return normalize(_extract_json(raw), text)
    except (llm.LLMError, ValueError):
        out = heuristic(text, prev_text)
        out["sensor_fallback"] = True
        return out
