"""sonara_story.py — A timestamped story to test whether Sonara learns what time means.

Synthetic, for testing only (see SONARA_PLAN.md, experiment 3). Each episode is a few
timestamped events; the task is the mood (happy or sad) at a query date and time. The time
gap is never written in the text: only the timestamps carry it. Training is 100 episodes
about Jake in time order; the tests are new Jake episodes, unfamiliar people and objects,
and identical events asked minutes vs days later.

    python3 sonara_story.py --dir learning_data/sonara_story
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sonara_memory
from sonara_memory import SonaraMemorySystem

MIN, HOUR, DAY = 60.0, 3600.0, 86400.0
START = dt.datetime(2026, 1, 5, 8, 0).timestamp()
PEOPLE = {"train": [("Jake", "his")], "unfamiliar": [("Maya", "her"), ("Sam", "his")]}
IMPORTANT = {"train": ["ball", "wallet", "keys"], "unfamiliar": ["phone", "bike"]}
UNIMPORTANT = {"train": ["pen", "sock", "rubber band"], "unfamiliar": ["paperclip", "napkin"]}
LOSE = ["{p} lost {pos} {o}.", "{p} can't find {pos} {o}.", "{p} has no idea where {pos} {o} went.", "{p} misplaced {pos} {o}."]
FIND = ["{p} found {pos} {o}.", "{p}'s {o} turned up under the couch.", "{p} got {pos} {o} back."]
HAPPY = ["{p} is happy.", "{p} is smiling.", "{p} seems cheerful.", "{p} is in a good mood."]
SAD = ["{p} is sad.", "{p} looks upset.", "{p} is down.", "{p} seems unhappy."]
OTHER_SAD = ["It rained and {p} had to stay inside.", "{p}'s best friend moved away.", "{p} has a bad cold.",
             "{p}'s team lost the big game."]
NEUTRAL = ["{p} had a snack.", "{p} went for a walk.", "{p} read a book.", "{p} called a friend.",
           "{p} cleaned {pos} room."]
# kind -> number of training episodes (the user's proportions, plus the days-later group)
TRAIN_KINDS = {"lose_important_soon": 30, "lose_important_days_later": 15, "lose_unimportant": 15,
               "sad_other_reason": 25, "lose_then_find": 15}


def say(templates: List[str], person: Tuple[str, str], obj: str = "") -> str:
    return random.choice(templates).format(p=person[0], pos=person[1], o=obj)


def episode(kind: str, t: float, group: str):
    """(events as (time, text), query time, mood at the query, time the episode ends, person)."""
    person = random.choice(PEOPLE[group])
    events = [(t, say(NEUTRAL, person))]
    t += random.uniform(10 * MIN, 2 * HOUR)
    soon = random.uniform(2 * MIN, 40 * MIN)
    obj = random.choice(IMPORTANT[group])
    if kind in ("lose_important_soon", "lose_important_days_later", "lose_then_find"):
        events.append((t, say(LOSE, person, obj)))
    if kind == "lose_important_soon":
        query, mood = t + soon, "sad"
    elif kind == "lose_important_days_later":
        query, mood = t + random.uniform(2 * DAY, 4 * DAY), "happy"
    elif kind == "lose_unimportant":
        events.append((t, say(LOSE, person, random.choice(UNIMPORTANT[group]))))
        query, mood = t + soon, "happy"
    elif kind == "sad_other_reason":
        events.append((t, say(OTHER_SAD, person)))
        query, mood = t + soon, "sad"
    else:  # lose_then_find: sad soon after, then finds it, then happy soon after
        events.append((t + soon, say(SAD, person)))
        t += soon + random.uniform(1 * HOUR, 4 * HOUR)
        events.append((t, say(FIND, person, obj)))
        query, mood = t + random.uniform(2 * MIN, 40 * MIN), "happy"
    return events, query, mood, query + MIN, person


def generate(memory_dir: str, seed: int = 0) -> List[Dict]:
    """Write every episode's events as timestamped memories and return the episode list."""
    random.seed(seed)
    clock = {"t": START}
    real_stamps = sonara_memory.get_timestamps
    sonara_memory.get_timestamps = lambda: (dt.datetime.fromtimestamp(clock["t"]).astimezone().isoformat(),
                                            clock["t"], dt.datetime.fromtimestamp(clock["t"]).strftime("%b %d, %Y at %I:%M %p"))
    try:
        return _generate(memory_dir, clock)
    finally:
        sonara_memory.get_timestamps = real_stamps


def _generate(memory_dir: str, clock: Dict[str, float]) -> List[Dict]:
    memory = SonaraMemorySystem(memory_dir)
    episodes: List[Dict] = []
    day = START

    def record(eid: str, split: str, kind: str, events, query: float, mood: str, person, store_mood: bool):
        ids = []
        for when, text in events:
            clock["t"] = when
            ids.append(memory.remember(text, source="user_note").id)
        if store_mood:  # training episodes include the mood itself, at the query time
            clock["t"] = query
            ids.append(memory.remember(say(HAPPY if mood == "happy" else SAD, person), source="user_note").id)
        episodes.append({"id": eid, "split": split, "kind": kind, "memory_ids": ids, "query_epoch": query,
                         "mood": mood, "person": person[0]})

    kinds = [k for k, n in TRAIN_KINDS.items() for _ in range(n)]
    random.shuffle(kinds)
    for i, kind in enumerate(kinds):
        start = day + random.uniform(0, 12 * HOUR)  # a random time in the day
        events, query, mood, end, person = episode(kind, start, "train")
        record(f"train{i}", "train", kind, events, query, mood, person, store_mood=True)
        day = max(day + DAY, end - (end - START) % DAY + DAY)  # next episode starts on a later day
    test_kinds = list(TRAIN_KINDS)
    for split, group, n in (("familiar", "train", 20), ("unfamiliar", "unfamiliar", 20)):
        for i in range(n):
            kind = test_kinds[i % len(test_kinds)]
            events, query, mood, end, person = episode(kind, day + random.uniform(0, 12 * HOUR), group)
            record(f"{split}{i}", split, kind, events, query, mood, person, store_mood=False)
            day = max(day + DAY, end - (end - START) % DAY + DAY)
    for i in range(10):  # the same events, asked soon (sad) and days later (happy)
        events, soon_query, _, _, person = episode("lose_important_soon", day + random.uniform(0, 12 * HOUR), "train")
        lost_at = events[-1][0]
        later = lost_at + random.uniform(2 * DAY, 4 * DAY)
        record(f"time{i}soon", "time", "lose_important_soon", events, soon_query, "sad", person, store_mood=False)
        episodes.append(dict(episodes[-1], id=f"time{i}later", kind="lose_important_days_later",
                             query_epoch=later, mood="happy"))
        day = max(day + DAY, later - (later - START) % DAY + DAY)
    Path(memory_dir, "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    return episodes


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="learning_data/sonara_story")
    args = parser.parse_args()
    if Path(args.dir, "memories.jsonl").exists():
        raise SystemExit(f"{args.dir} already has memories; choose a new --dir")
    episodes = generate(args.dir)
    memory = SonaraMemorySystem(args.dir)
    print(f"{len(episodes)} episodes, {len(memory.memories)} memories in {args.dir}")
    for e in episodes[:3] + [x for x in episodes if x["split"] != "train"][:2]:
        print(f"\n{e['id']} ({e['kind']}): mood at {dt.datetime.fromtimestamp(e['query_epoch']):%b %d %H:%M} = {e['mood']}")
        for mid in e["memory_ids"]:
            m = memory.memories[mid]
            print(f"  {dt.datetime.fromtimestamp(m.timestamp_epoch):%b %d %H:%M}  {m.content}")


if __name__ == "__main__":
    main()
