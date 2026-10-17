"""experiment_world_generalization.py - Harder generalization benchmark for SonaraWorldModel.
=========================================================================================

WHY THIS EXISTS
    experiment_world_learning.py (Experiment 1) generates every scenario from ONE sentence
    template with places always in the same roles and the same 9 questions. "Held-out"
    there only means new names. A 25-line rule script with no learning scores 98.7% on
    that held-out set, so 100% there cannot distinguish learned event understanding from
    template matching.

    This benchmark separates the two by varying what Experiment 1 holds fixed:
      * story structure: 2-3 objects, 2-3 actors, random places/roles, interleaved moves,
        random timestamps, distractor and noise events, optional corrections
      * answers: yes/no questions are balanced; "unknown" questions are about objects
        or moves that genuinely are not in the story (not a fixed "what did the cat chase")
      * phrasing: event and question templates are split into TRAIN and HELD-OUT sets

TEST CONDITIONS (all use entities never seen in training)
    A  new_entities          train phrasings, new names              (= what Experiment 1 tested)
    B  new_event_phrasing    held-out EVENT phrasings
    C  new_event_and_query   held-out event AND question phrasings
    D  longer_stories        train phrasings, 2x longer stories

SYSTEMS COMPARED
    rule_baseline        positional heuristics, no learning (the script that scored 98.7%
                         on Experiment 1, given both question phrasings for fairness)
    majority_baseline    most common training answer per question category
    world_model          SonaraWorldModel trained by gradient descent
    world_model_ablated  same, with the hand-computed progression features zeroed
                         (event_temporal_features, event_local_features, global_rel_features)
    shuffled_time        world model trained and tested with event order shuffled
    learned              Experiment 3: learned vocabulary + entity gate, no hand-computed
                         features (state-based event scoring instead), extra TRAINING
                         phrasings, cosine learning-rate decay
    learned_original_phrasing, learned_with_features, structural_extended_phrasing
                         the same with one change removed or applied alone (see NEURAL_SYSTEMS)

A result counts as learned event understanding only where the world model clearly beats
the rule baseline AND survives the ablation. Test stories use their own RNG and only the
original phrasings, so every system and training variety is scored on identical tests.

Usage:
    python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 7 \
        --systems world_model,learned --output artifacts/world_generalization_v2_report.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sonara_world_model import SonaraWorldModel

# =====================================================================
# 1. VOCABULARY AND TEMPLATES
# =====================================================================

TRAIN_ACTORS = ("jake", "lucy", "max", "piper", "ava", "leo")
TRAIN_OBJECTS = ("ball", "stick", "toy", "bone", "cup", "hat")
TRAIN_PLACES = ("yard", "pond", "shed", "kitchen", "hall", "garden")

TEST_ACTORS = ("sophia", "oliver", "emma", "noah", "mia", "ethan")
TEST_OBJECTS = ("frisbee", "blanket", "journal", "lantern", "kettle", "scarf")
TEST_PLACES = ("attic", "porch", "garage", "basement", "cellar", "studio")

# A move puts object {o} in place {p}. Held-out phrasings deliberately break
# "the place is the last word" and "actor comes first".
MOVE = {
    "train": [
        "{a} carried the {o} into the {p}.",
        "{a} moved the {o} into the {p}.",
        "{a} placed the {o} into the {p}.",
        "{a} took the {o} to the {p}.",
    ],
    "heldout": [
        "{a} put the {o} down in the {p}.",
        "the {o} was taken to the {p} by {a}.",
        "in the {p}, {a} set down the {o}.",
        "{a} dropped off the {o} at the {p} and walked away.",
    ],
}
# Mentions an object (and sometimes a place) WITHOUT moving it.
DISTRACT = {
    "train": [
        "{a} looked at the {o}.",
        "{a} wanted to take the {o} to the {p} but stopped.",
    ],
    "heldout": [
        "{a} asked about the {o}.",
        "{a} thought about bringing the {o} to the {p} but did not.",
    ],
}
NOISE = {
    "train": ["{a} drank some water.", "{a} opened a window."],
    "heldout": ["{a} answered the phone.", "{a} sat down to rest."],
}
# The object's most recent move did not happen; it stayed where it was.
CORRECTION = {
    "train": ["correction: the {o} never reached the {pbad}; it stayed in the {pgood}."],
    "heldout": ["correction: {a} actually left the {o} in the {pgood}, not the {pbad}."],
}

QUESTIONS = {
    "current_state": {
        "train": ["where is the {o}?"],
        "heldout": ["what room is the {o} in now?"],
    },
    "previous_state_time": {
        "train": ["where was the {o} before {t}?"],
        "heldout": ["before {t}, where was the {o}?"],
    },
    "previous_state_notime": {
        "train": ["where was the {o} before it went to the {p}?"],
        "heldout": ["where had the {o} been before it reached the {p}?"],
    },
    "first_location": {
        "train": ["where was the {o} first?"],
        "heldout": ["where did the {o} start out?"],
    },
    "actor": {
        "train": ["who moved the {o} into the {p}?"],
        "heldout": ["who brought the {o} to the {p}?"],
    },
    "object_identity": {
        "train": ["what did {a} put in the {p}?"],
        "heldout": ["which item did {a} leave in the {p}?"],
    },
    "event_order": {
        "train": ["did {t1} happen before {t2}?"],
        "heldout": ["was {t1} earlier than {t2}?"],
    },
    "membership": {
        "train": ["is the {o} in the {p}?"],
        "heldout": ["is the {o} currently in the {p}?"],
    },
    "state_transition": {
        "train": ["where will the {o} be if {a} moves it into the {p}?"],
        "heldout": ["if {a} takes the {o} to the {p}, where will it end up?"],
    },
}

# Extra TRAINING-only phrasings for --variety extended (recommendation 3: train on more
# phrasing variety than is tested). They add new verbs and word orders but deliberately
# avoid the held-out constructions (passive "was taken ... by", place-first or
# clause-first sentences, "put/set/dropped off", "room", "now", "start out", "brought").
EXTRA_TRAIN = {
    "move": [
        "{a} stashed the {o} inside the {p}.",
        "{a} quickly hauled the {o} over to the {p}.",
        "{a} slid the {o} across into the {p} after lunch.",
        "{a} returned the {o} to the {p}.",
    ],
    "distract": [
        "{a} pointed at the {o}.",
        "{a} nearly moved the {o} to the {p} but changed plans.",
    ],
    "noise": ["{a} read a book.", "{a} made some tea."],
    "correction": ["correction: the move of the {o} to the {pbad} never happened; it remained in the {pgood}."],
    "current_state": ["where can the {o} be found?", "which place holds the {o}?"],
    "previous_state_time": ["where was the {o} located before {t}?"],
    "previous_state_notime": ["what place was the {o} in before it went to the {p}?"],
    "first_location": ["where was the {o} at the beginning?"],
    "actor": ["which person moved the {o} into the {p}?"],
    "object_identity": ["what object did {a} put in the {p}?"],
    "event_order": ["did {t1} come before {t2}?"],
    "membership": ["is the {o} located in the {p}?"],
    "state_transition": ["where would the {o} be after {a} moves it into the {p}?"],
}

VARIETIES = ("original", "extended")


def _tmpl(table: Dict[str, List[str]], key: str, split: str, variety: str) -> List[str]:
    """Templates for one split; extended variety adds EXTRA_TRAIN phrasings to TRAIN only."""
    extra = EXTRA_TRAIN.get(key, []) if (split == "train" and variety == "extended") else []
    return table[split] + extra


@dataclass
class Q:
    question: str
    target: str
    category: str


@dataclass
class Story:
    events: List[str]
    questions: List[Q]
    meta: Dict = field(default_factory=dict)


def _fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# =====================================================================
# 2. STORY GENERATOR WITH GROUND-TRUTH WORLD STATE
# =====================================================================

def generate_story(
    rng: random.Random,
    actors_pool,
    objects_pool,
    places_pool,
    event_split: str,
    query_split: str,
    length_scale: int = 1,
    variety: str = "original",
) -> Story:
    n_obj = rng.choice([2, 3])
    objs = rng.sample(objects_pool, n_obj)
    absent_obj = rng.choice([o for o in objects_pool if o not in objs])
    actors = rng.sample(actors_pool, rng.choice([2, 3]))
    places = rng.sample(places_pool, 4)

    # Plan a sequence of abstract actions, then render with templates.
    plan: List[Tuple] = []
    for o in objs:
        plan.append(("move", o))  # initial placement
    extra_moves = rng.randint(n_obj, 2 * n_obj) * length_scale
    for _ in range(extra_moves):
        plan.append(("move", rng.choice(objs)))
    for _ in range(rng.randint(1, 2) * length_scale):
        plan.append(("distract", rng.choice(objs)))
    for _ in range(rng.randint(0, 1) * length_scale):
        plan.append(("noise", None))
    # keep each object's initial placement first for that object
    head, tail = plan[:n_obj], plan[n_obj:]
    rng.shuffle(tail)
    plan = head + tail

    corrected_obj = rng.choice(objs) if rng.random() < 0.4 else None

    minute = rng.randint(8 * 60, 16 * 60)
    events: List[str] = []
    times: List[str] = []
    # history[o] = list of dicts {time, place, actor, idx, corrected}
    history: Dict[str, List[Dict]] = {o: [] for o in objs}
    current: Dict[str, Optional[str]] = {o: None for o in objs}

    def emit(text: str) -> str:
        nonlocal minute
        minute += rng.randint(1, 25)
        t = _fmt_time(minute)
        events.append(f"{t} {text}")
        times.append(t)
        return t

    for kind, o in plan:
        if kind == "move":
            choices = [p for p in places if p != current[o]]
            p = rng.choice(choices)
            a = rng.choice(actors)
            t = emit(rng.choice(_tmpl(MOVE, "move", event_split, variety)).format(a=a, o=o, p=p))
            history[o].append({"time": t, "place": p, "actor": a, "idx": len(events) - 1, "corrected": False})
            current[o] = p
        elif kind == "distract":
            a = rng.choice(actors)
            p = rng.choice([q for q in places if q != current[o]])
            emit(rng.choice(_tmpl(DISTRACT, "distract", event_split, variety)).format(a=a, o=o, p=p))
        else:
            emit(rng.choice(_tmpl(NOISE, "noise", event_split, variety)).format(a=rng.choice(actors)))

    if corrected_obj is not None and len(history[corrected_obj]) >= 2:
        o = corrected_obj
        last, prev = history[o][-1], history[o][-2]
        emit(rng.choice(_tmpl(CORRECTION, "correction", event_split, variety)).format(a=last["actor"], o=o, pbad=last["place"], pgood=prev["place"]))
        last["corrected"] = True
        current[o] = prev["place"]
    else:
        corrected_obj = None

    true_hist = {o: [h for h in history[o] if not h["corrected"]] for o in objs}

    def qt(cat: str) -> str:
        return rng.choice(_tmpl(QUESTIONS[cat], cat, query_split, variety))

    qs: List[Q] = []
    for o in objs:
        qs.append(Q(qt("current_state").format(o=o), current[o], "current_state"))
        qs.append(Q(qt("first_location").format(o=o), true_hist[o][0]["place"], "first_location"))
    # previous-state questions on a non-initial true move
    cand_moves = [(o, i) for o in objs for i in range(1, len(true_hist[o]))]
    if cand_moves:
        o, i = rng.choice(cand_moves)
        h, prev = true_hist[o][i], true_hist[o][i - 1]
        qs.append(Q(qt("previous_state_time").format(o=o, t=h["time"]), prev["place"], "previous_state_time"))
        # timestamp-free version needs the destination to be unique in this object's history
        if sum(1 for x in true_hist[o] if x["place"] == h["place"]) == 1:
            qs.append(Q(qt("previous_state_notime").format(o=o, p=h["place"]), prev["place"], "previous_state_notime"))
    # actor: unique (object, place) true move
    pairs = Counter((o, h["place"]) for o in objs for h in true_hist[o])
    uniq_moves = [(o, h) for o in objs for h in true_hist[o] if pairs[(o, h["place"])] == 1]
    if uniq_moves:
        o, h = rng.choice(uniq_moves)
        qs.append(Q(qt("actor").format(o=o, p=h["place"]), h["actor"], "actor"))
    # object identity: (actor, place) pair with exactly one object
    ap = defaultdict(set)
    for o in objs:
        for h in true_hist[o]:
            ap[(h["actor"], h["place"])].add(o)
    uniq_ap = [(k, next(iter(v))) for k, v in ap.items() if len(v) == 1]
    if uniq_ap:
        (a, p), o = rng.choice(uniq_ap)
        qs.append(Q(qt("object_identity").format(a=a, p=p), o, "object_identity"))
    # event order, balanced
    i1, i2 = rng.sample(range(len(times)), 2)
    if rng.random() < 0.5:
        i1, i2 = min(i1, i2), max(i1, i2)
    else:
        i1, i2 = max(i1, i2), min(i1, i2)
    qs.append(Q(qt("event_order").format(t1=times[i1], t2=times[i2]), "yes" if i1 < i2 else "no", "event_order"))
    # membership, balanced
    o = rng.choice(objs)
    if rng.random() < 0.5:
        qs.append(Q(qt("membership").format(o=o, p=current[o]), "yes", "membership"))
    else:
        mentioned = [h["place"] for h in history[o] if h["place"] != current[o]]
        p = rng.choice(mentioned or [q for q in places if q != current[o]])
        qs.append(Q(qt("membership").format(o=o, p=p), "no", "membership"))
    # state transition (destination must appear in the story so it is a pointer candidate)
    o = rng.choice(objs)
    seen_places = sorted({h["place"] for x in objs for h in history[x]} - {current[o]})
    if seen_places:
        p = rng.choice(seen_places)
        qs.append(Q(qt("state_transition").format(o=o, a=rng.choice(actors), p=p), p, "state_transition"))
    # missing info: object not in story, or a move that never happened
    if rng.random() < 0.5:
        qs.append(Q(qt("current_state").format(o=absent_obj), "unknown", "missing_info"))
    else:
        o = rng.choice(objs)
        never = [p for p in places if all(h["place"] != p for h in history[o])]
        if never:
            qs.append(Q(qt("actor").format(o=o, p=rng.choice(never)), "unknown", "missing_info"))

    return Story(events=events, questions=qs, meta={"corrected": corrected_obj is not None})


def build_splits(
    seed: int, n_train: int, n_test: int, variety: str = "original",
) -> Tuple[List[Story], Dict[str, List[Story]]]:
    """Test stories come from their own RNG and never use EXTRA_TRAIN phrasings, so
    every system and training variety is scored on the identical test set."""
    rng = random.Random(seed)
    train = [
        generate_story(rng, TRAIN_ACTORS, TRAIN_OBJECTS, TRAIN_PLACES, "train", "train", variety=variety)
        for _ in range(n_train)
    ]
    rng = random.Random(seed + 10_000)
    tests = {
        "A_new_entities": [generate_story(rng, TEST_ACTORS, TEST_OBJECTS, TEST_PLACES, "train", "train") for _ in range(n_test)],
        "B_new_event_phrasing": [generate_story(rng, TEST_ACTORS, TEST_OBJECTS, TEST_PLACES, "heldout", "train") for _ in range(n_test)],
        "C_new_event_and_query_phrasing": [generate_story(rng, TEST_ACTORS, TEST_OBJECTS, TEST_PLACES, "heldout", "heldout") for _ in range(n_test)],
        "D_longer_stories": [generate_story(rng, TEST_ACTORS, TEST_OBJECTS, TEST_PLACES, "train", "train", length_scale=2) for _ in range(n_test)],
    }
    return train, tests


# =====================================================================
# 3. NON-LEARNING BASELINES
# =====================================================================

def _ts(event: str) -> int:
    m = re.match(r"(\d+):(\d+)", event)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _last_word(event: str) -> str:
    return re.findall(r"[a-z]+", event.split(";")[-1].lower())[-1]


def rule_answer(q: str, events: List[str]) -> str:
    """Positional heuristics tuned to Experiment 1's template. No learning.
    Understands BOTH question phrasings so it is only judged on reading events."""
    q = q.lower()

    def mentions(o, evs):
        return [e for e in evs if re.search(rf"\b{o}\b", e.lower())]

    def where(o, evs):
        ms = mentions(o, evs)
        return _last_word(ms[-1]) if ms else "unknown"

    def tmin(h, m):
        return int(h) * 60 + int(m)

    m = re.search(r"(?:did|was) (\d+):(\d+) (?:happen before|earlier than) (\d+):(\d+)", q)
    if m:
        return "yes" if tmin(m[1], m[2]) < tmin(m[3], m[4]) else "no"
    m = re.search(r"where was the (\w+) before (\d+):(\d+)", q) or re.search(r"before (\d+):(\d+), where was the (\w+)", q)
    if m:
        if m.re.pattern.startswith("before"):
            o, t = m[3], tmin(m[1], m[2])
        else:
            o, t = m[1], tmin(m[2], m[3])
        return where(o, [e for e in events if _ts(e) < t])
    m = re.search(r"where (?:was|had) the (\w+) (?:been )?before it (?:went to|reached) the (\w+)", q)
    if m:
        o, p = m[1], m[2]
        ms = mentions(o, events)
        for i, e in enumerate(ms):
            if _last_word(e) == p and i > 0:
                return _last_word(ms[i - 1])
        return "unknown"
    m = re.search(r"where (?:was the (\w+) first|did the (\w+) start out)", q)
    if m:
        ms = mentions(m[1] or m[2], events)
        return _last_word(ms[0]) if ms else "unknown"
    m = re.search(r"where will the \w+ be if .* into the (\w+)", q) or re.search(r"if \w+ takes the \w+ to the (\w+)", q)
    if m:
        return m[1]
    m = re.search(r"where is the (\w+)", q) or re.search(r"what room is the (\w+) in", q)
    if m:
        return where(m[1], events)
    m = re.search(r"who (?:moved|brought) the (\w+) (?:into|to) the (\w+)", q)
    if m:
        for e in events:
            if m[1] in e and m[2] in e:
                return re.findall(r"[a-z]+", e.lower())[0]
        return "unknown"
    m = re.search(r"what did (\w+) put in the (\w+)", q) or re.search(r"which item did (\w+) leave in the (\w+)", q)
    if m:
        for e in events:
            if m[1] in e and m[2] in e:
                mm = re.search(r"the (\w+)", e)
                return mm[1] if mm else "unknown"
        return "unknown"
    m = re.search(r"is the (\w+) (?:currently )?in the (\w+)", q)
    if m:
        return "yes" if where(m[1], events) == m[2] else "no"
    return "unknown"


def experiment1_rule_answer(q: str, events: List[str]) -> str:
    """The original 25-line rule script written against Experiment 1's single template.
    No learning. Scores 98.7% on Experiment 1's held-out set."""
    def where(o, evs):
        ms = [e for e in evs if o in e and "inspected" not in e]
        return _last_word(ms[-1]) if ms else "unknown"
    m = re.match(r"did (\d+):(\d+) happen before (\d+):(\d+)", q)
    if m:
        return "yes" if int(m[1] + m[2]) < int(m[3] + m[4]) else "no"
    m = re.match(r"where was the (\w+) before (\d+):(\d+)", q)
    if m:
        return where(m[1], [e for e in events if int(e[:5].replace(":", "")) < int(m[2] + m[3])])
    m = re.match(r"where will the \w+ be if .* into the (\w+)", q)
    if m:
        return m[1]
    m = re.match(r"where is the (\w+)", q)
    if m:
        return where(m[1], events)
    m = re.match(r"what did (\w+) place into the (\w+)", q)
    if m:
        for e in events:
            if m[1] in e and m[2] in e:
                return re.search(r"the (\w+) into", e)[1]
    m = re.match(r"is the (\w+) in the (\w+)", q)
    if m:
        return "yes" if where(m[1], events) == m[2] else "no"
    return "unknown"


def score(answer_fn: Callable[[str, List[str]], str], stories: List[Story]) -> Dict:
    tot, cor = 0, 0
    by_cat_n, by_cat_c = Counter(), Counter()
    for s in stories:
        for q in s.questions:
            ok = answer_fn(q.question, s.events).lower().strip() == q.target
            tot += 1
            cor += ok
            by_cat_n[q.category] += 1
            by_cat_c[q.category] += ok
    return {
        "accuracy": round(cor / max(tot, 1), 4),
        "n": tot,
        "by_category": {c: round(by_cat_c[c] / by_cat_n[c], 4) for c in sorted(by_cat_n)},
    }


def majority_fn(train: List[Story]) -> Callable[[str, List[str]], str]:
    cat_answers: Dict[str, Counter] = defaultdict(Counter)
    for s in train:
        for q in s.questions:
            cat_answers[q.category][q.target] += 1
    # Detect category from question text with the same templates used to build it.
    def cat_of(question: str) -> str:
        for cat, splits in QUESTIONS.items():
            for tmpl in splits["train"] + splits["heldout"]:
                pat = re.escape(tmpl)
                pat = re.sub(r"\\\{\w+\\\}", r".+", pat)
                if re.fullmatch(pat, question):
                    return cat
        return "missing_info"
    majority = {c: cnt.most_common(1)[0][0] for c, cnt in cat_answers.items()}
    return lambda q, ev: majority.get(cat_of(q), "unknown")


# =====================================================================
# 4. NEURAL CONDITIONS
# =====================================================================

def ablate_engineered_features(wm: SonaraWorldModel) -> None:
    """Zero the hand-computed progression features (is_latest_ent, is_earliest_ent,
    is_before_anchor, anchor matches, left/right tau, final_end_entity_match).
    Token-level string-match scalars used by the copy mechanism are left intact."""
    m = wm.model
    tr_fwd, rs_fwd, hd_fwd = m.tracker.forward, m.reasoner.forward, m.head.forward

    def tracker(event_vectors, event_local_features=None):
        if event_local_features is not None:
            event_local_features = torch.zeros_like(event_local_features)
        return tr_fwd(event_vectors, event_local_features=event_local_features)

    def reasoner(*args, event_temporal_features=None, **kw):
        if event_temporal_features is not None:
            event_temporal_features = torch.zeros_like(event_temporal_features)
        return rs_fwd(*args, event_temporal_features=event_temporal_features, **kw)

    def head(*args, global_rel_features=None, **kw):
        if global_rel_features is not None:
            global_rel_features = torch.zeros_like(global_rel_features)
        return hd_fwd(*args, global_rel_features=global_rel_features, **kw)

    m.tracker.forward, m.reasoner.forward, m.head.forward = tracker, reasoner, head


def train_world_model(
    train: List[Story],
    steps: int,
    seed: int,
    system: Dict,
    val: Optional[List[Story]] = None,
    eval_every: int = 150,
) -> Tuple[SonaraWorldModel, Dict]:
    """Train with the model's own learn_step (AdamW + replay). Training is not monotonic,
    so when a validation set is given (training entities and phrasings, never test data)
    the best checkpoint by validation accuracy is kept. With lr_decay the learning rate
    follows a cosine from 0.002 down to 0.0002 (recommendation 4)."""
    torch.manual_seed(seed)
    base_lr = 0.002
    wm = SonaraWorldModel(
        device="cpu", dim=64, lr=base_lr,
        lexicon=system["lexicon"], engineered_features=system["features"],
    )
    if system["zero_features"]:
        ablate_engineered_features(wm)
    shuffle_time = system["shuffle"]
    rng = random.Random(seed + 7)
    by_cat: Dict[str, List[Tuple[List[str], str, str]]] = defaultdict(list)
    for s in train:
        evs = list(s.events)
        if shuffle_time:
            rng.shuffle(evs)
        for q in s.questions:
            by_cat[q.category].append((evs, q.question, q.target))
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    losses = []
    curve = []
    best_acc, best_state, best_step = -1.0, None, 0
    t0 = time.monotonic()
    for step in range(1, steps + 1):
        if system["lr_decay"]:
            lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / steps)))
            for group in wm.optimizer.param_groups:
                group["lr"] = lr
        pool = by_cat[cats[(step - 1) % len(cats)]]
        ev, q, tgt = pool[((step - 1) // len(cats)) % len(pool)]
        losses.append(wm.learn_step(ev, q, tgt)["loss"])
        if val is not None and (step % eval_every == 0 or step == steps):
            acc = score(model_fn(wm, shuffle_seed=seed + 5 if shuffle_time else None), val)["accuracy"]
            curve.append((step, acc))
            if acc > best_acc:
                best_acc, best_step = acc, step
                best_state = copy.deepcopy(wm.model.state_dict())
    if best_state is not None:
        wm.model.load_state_dict(best_state)
    info = {
        "steps": steps,
        "selected_step": best_step if val is not None else steps,
        "val_accuracy_selected": best_acc if val is not None else None,
        "val_curve": curve,
        "train_seconds": round(time.monotonic() - t0, 1),
        "loss_first_50": round(sum(losses[:50]) / max(len(losses[:50]), 1), 4),
        "loss_last_50": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4),
    }
    return wm, info


def model_fn(wm: SonaraWorldModel, shuffle_seed: Optional[int] = None) -> Callable[[str, List[str]], str]:
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None

    def fn(q: str, events: List[str]) -> str:
        evs = list(events)
        if rng is not None:
            rng.shuffle(evs)
        return wm.predict(q, events=evs).answer
    return fn


# =====================================================================
# 5. DRIVER
# =====================================================================

def _system(lexicon="structural", features=True, zero_features=False,
            variety="original", lr_decay=False, shuffle=False) -> Dict:
    return {"lexicon": lexicon, "features": features, "zero_features": zero_features,
            "variety": variety, "lr_decay": lr_decay, "shuffle": shuffle}


NEURAL_SYSTEMS = {
    # Experiment 2 designs (fixed STRUCTURAL_WORDS vocabulary, original training phrasings).
    "world_model": _system(),
    "world_model_ablated": _system(zero_features=True),
    "shuffled_time": _system(shuffle=True),
    # Experiment 3: the four recommended fixes together, and with one fix removed at a time.
    "learned": _system(lexicon="learned", features=False, variety="extended", lr_decay=True),
    "learned_original_phrasing": _system(lexicon="learned", features=False, lr_decay=True),
    "learned_with_features": _system(lexicon="learned", features=True, variety="extended", lr_decay=True),
    "structural_extended_phrasing": _system(variety="extended", lr_decay=True),
}
DEFAULT_SYSTEMS = ("world_model", "learned", "structural_extended_phrasing")


def build_val(seed: int, variety: str = "original", n: int = 30) -> List[Story]:
    """Validation stories from the TRAINING distribution (training entities and phrasings)."""
    rng = random.Random(seed + 555)
    return [
        generate_story(rng, TRAIN_ACTORS, TRAIN_OBJECTS, TRAIN_PLACES, "train", "train", variety=variety)
        for _ in range(n)
    ]


def run_neural_job(job: Tuple[int, str, int, int, int]) -> Tuple[int, str, Dict]:
    seed, name, steps, n_train, n_test = job
    torch.set_num_threads(1)
    system = NEURAL_SYSTEMS[name]
    train, tests = build_splits(seed, n_train, n_test, variety=system["variety"])
    wm, info = train_world_model(train, steps, seed, system, val=build_val(seed, system["variety"]))
    fn = model_fn(wm, shuffle_seed=seed + 99 if system["shuffle"] else None)
    entry: Dict = {"system": system, "training": info, "train": score(fn, train[:60])}
    conds = tests if not system["shuffle"] else {"A_new_entities": tests["A_new_entities"]}
    for k, v in conds.items():
        entry[k] = score(fn, v)
    print(f"  seed {seed}: {name} done ({info['train_seconds']}s, best step {info['selected_step']}) "
          + " ".join(f"{k[:1]}={entry[k]['accuracy']:.1%}" for k in conds), flush=True)
    return seed, name, entry


def run_baselines(seed: int, n_train: int, n_test: int) -> Dict:
    train, tests = build_splits(seed, n_train, n_test)
    maj = majority_fn(train)
    return {
        "rule_baseline": {"train": score(rule_answer, train), **{k: score(rule_answer, v) for k, v in tests.items()}},
        "majority_baseline": {k: score(maj, v) for k, v in tests.items()},
    }


def summarize(runs: List[Dict]) -> Dict:
    out: Dict = {}
    for sysname in runs[0]["systems"]:
        if not all(sysname in r["systems"] for r in runs):
            continue
        out[sysname] = {}
        for cond, val in runs[0]["systems"][sysname].items():
            if isinstance(val, dict) and "accuracy" in val:
                accs = [r["systems"][sysname][cond]["accuracy"] for r in runs]
                cats = val["by_category"].keys()
                out[sysname][cond] = {
                    "mean_accuracy": round(sum(accs) / len(accs), 4),
                    "per_seed": accs,
                    "by_category_mean": {
                        c: round(sum(r["systems"][sysname][cond]["by_category"].get(c, 0) for r in runs) / len(runs), 4)
                        for c in cats
                    },
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--train-stories", type=int, default=240)
    ap.add_argument("--test-stories", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1, help="parallel training processes")
    ap.add_argument("--systems", type=str, default=",".join(DEFAULT_SYSTEMS),
                    help="comma-separated neural systems: " + ", ".join(NEURAL_SYSTEMS))
    ap.add_argument("--output", type=str, default="artifacts/world_generalization_report.json")
    ap.add_argument("--show-examples", action="store_true", help="print one story per condition and exit")
    ap.add_argument("--score-experiment1", action="store_true", help="score the rule baseline on Experiment 1 held-out data and exit")
    args = ap.parse_args()

    if args.score_experiment1:
        # Rule baseline (no learning) on Experiment 1's own held-out set.
        from experiment_world_learning import build_experiment_datasets
        for seed in (42, 43, 44):
            _, unseen = build_experiment_datasets(seed)
            n = sum(len(sc.questions) for sc in unseen)
            c = sum(experiment1_rule_answer(q.question, sc.events) == q.target for sc in unseen for q in sc.questions)
            c2 = sum(rule_answer(q.question, sc.events) == q.target for sc in unseen for q in sc.questions)
            print(f"Experiment 1 held-out, seed {seed}: original rule script {c / n:.1%} | "
                  f"this benchmark's rule_baseline {c2 / n:.1%}")
        return

    if args.show_examples:
        train, tests = build_splits(1, 1, 1)
        for name, stories in [("TRAIN", train)] + list(tests.items()):
            print(f"\n=== {name} ===")
            for e in stories[0].events:
                print("  ", e)
            for q in stories[0].questions:
                print(f"   Q [{q.category}] {q.question} -> {q.target}")
        return

    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    unknown = [name for name in systems if name not in NEURAL_SYSTEMS]
    if unknown:
        ap.error(f"unknown systems: {unknown}")
    seeds = [1000 + i for i in range(args.seeds)]
    runs = []
    for s in seeds:
        train, _ = build_splits(s, args.train_stories, args.test_stories)
        runs.append({"seed": s, "n_train_questions": sum(len(x.questions) for x in train),
                     "systems": run_baselines(s, args.train_stories, args.test_stories)})
    jobs = [(s, name, args.steps, args.train_stories, args.test_stories) for s in seeds for name in systems]
    by_seed = {r["seed"]: r for r in runs}
    partial = Path(args.output).with_suffix(".partial.json")
    partial.parent.mkdir(parents=True, exist_ok=True)

    def record(result) -> None:
        # Save after every finished model so an interrupted run keeps its results.
        s, name, entry = result
        by_seed[s]["systems"][name] = entry
        partial.write_text(json.dumps({"config": vars(args), "runs": runs}, indent=2))

    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for result in pool.imap_unordered(run_neural_job, jobs, chunksize=1):
                record(result)
    else:
        for j in jobs:
            record(run_neural_job(j))
    report = {
        "benchmark": "world_generalization_v2",
        "config": vars(args),
        "summary": summarize(runs),
        "runs": runs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    s = report["summary"]
    print(f"\n{'system':22s}" + "".join(f"{c[:26]:>28s}" for c in s["rule_baseline"]))
    for sysname, conds in s.items():
        print(f"{sysname:22s}" + "".join(f"{conds[c]['mean_accuracy']:>28.1%}" if c in conds else f"{'-':>28s}" for c in s["rule_baseline"]))


if __name__ == "__main__":
    main()
