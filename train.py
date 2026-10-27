"""Train association strengths from synthetic *labeled* episodes and evaluate held-out identities.

The paragraph encoder in sonara_memory.py is still a bounded teaching grammar.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import tempfile

from sonara_memory import Episode, LinkLearner, Memory, encode_json

TRAIN_PEOPLE = ("Sarah", "Maya", "Noah", "Liam", "Nora", "Emma", "Ivan", "Ava")
TEST_PEOPLE = ("Zara", "Felix", "Owen", "Ruby", "Amir", "Lena", "Caleb", "Iris")
TRAIN_ITEMS = ("keys", "journal", "wallet", "notebook", "umbrella", "badge", "camera", "scarf")
TEST_ITEMS = ("compass", "lantern", "tablet", "map", "violin", "charger", "helmet", "thermos")
TRAIN_PLACES = ("kitchen table", "front door", "bedroom drawer", "office shelf", "hallway bench", "garage cabinet")
TEST_PLACES = ("attic shelf", "garden shed", "porch basket", "basement desk", "window ledge", "studio rack")


def build_examples(seed: int = 17, n: int = 500, test: bool = False):
    rng = random.Random(seed)
    people = TEST_PEOPLE if test else TRAIN_PEOPLE
    objects = TEST_ITEMS if test else TRAIN_ITEMS
    places = TEST_PLACES if test else TRAIN_PLACES
    samples: list[tuple[Episode, Episode, int]] = []
    paragraphs: list[tuple[str, dict]] = []
    for i in range(n):
        actor, other = rng.sample(people, 2)
        item, other_item = rng.sample(objects, 2)
        usual, before, after = rng.sample(places, 3)
        first = Episode(actor=actor, item=item, action="placed", location=before, kind="location")
        second = Episode(actor=actor, item=item, action="moved", location=after, kind="location")
        habit = Episode(actor=actor, item=item, action="usually", location=usual, kind="habit")
        samples.extend([(first, second, 1), (second, habit, 1),
                        (first, Episode(actor=other, item=item, location=before, kind="location"), 0),
                        (first, Episode(actor=actor, item=other_item, location=after, kind="location"), 0),
                        (first, Episode(actor=other, item=other_item, location=before, kind="habit"), 0)])
        first = rng.choice((
            f"At dawn, {actor} placed the {item} on the {before}.",
            f"On Monday morning, {actor} put the {item} on the {before}.",
            f"At sunrise, {actor} left the {item} at the {before}.",
            f"At 8 AM, {actor} stored the {item} in the {before}."))
        second = rng.choice((
            f"At noon, {actor} moved the {item} to the {after}.",
            f"Later that afternoon, {actor} moved the {item} into the {after}.",
            f"At dusk, {actor} carried the {item} to the {after}.",
            f"On Tuesday evening, {actor} placed the {item} in the {after}."))
        habitual = rng.choice((
            f"Usually, {actor} keeps the {item} by the {usual}.",
            f"Usually, {actor} stores the {item} at the {usual}.",
            f"Usually, {actor} puts the {item} near the {usual}."))
        text = f"{first} {second} {habitual}"
        paragraphs.append((text, {"actor": actor, "item": item, "before": before,
                                  "after": after, "usual": usual, "novel": True}))
    rng.shuffle(samples)
    return samples, paragraphs


def build_context_examples(seed: int = 29, n: int = 500, test: bool = False):
    """Teach contextual overlap separately from same-object identity.

    A shared object *label* across people is NOT evidence of one physical item.
    """
    rng = random.Random(seed)
    people = TEST_PEOPLE if test else TRAIN_PEOPLE
    objects = TEST_ITEMS if test else TRAIN_ITEMS
    places = TEST_PLACES if test else TRAIN_PLACES
    samples = []
    for _ in range(n):
        actor, other = rng.sample(people, 2)
        item, other_item = rng.sample(objects, 2)
        place, other_place = rng.sample(places, 2)
        a = Episode(actor=actor, item=item, action="moved", location=place,
                    stated_time="Monday morning", kind="location",
                    source=f"{actor} moved the {item} into the {place}.")
        samples.extend([
            (a, Episode(actor=actor, item=other_item, action="placed",
                        location=other_place, stated_time="Tuesday evening",
                        kind="location", source=f"{actor} placed the {other_item} at the {other_place}."), 1),
            (a, Episode(actor=other, item=other_item, action="placed", location=place,
                        stated_time="Tuesday evening", kind="location",
                        source=f"{other} placed the {other_item} on the {place}."), 1),
            (a, Episode(actor=other, item=item, action="moved", location=other_place,
                        stated_time="Tuesday evening", kind="location",
                        source=f"{other} moved the {item} to the {other_place}."), 0),
            (a, Episode(actor=other, item=other_item, action="placed", location=other_place,
                        stated_time="Tuesday evening", kind="location",
                        source=f"{other} placed the {other_item} into the {other_place}."), 0),
        ])
    rng.shuffle(samples)
    return samples

def build_signal_curriculum(seed: int = 103, n: int = 480, test: bool = False):
    """Varied pair supervision for the full cue vocabulary, beyond the grammar.

    Labels come from explicit synthetic actor/item/place relations, never from
    the model's own output. This teaches cues, NOT open-domain paragraph parsing.
    """
    rng = random.Random(seed)
    people = TEST_PEOPLE if test else TRAIN_PEOPLE
    objects = TEST_ITEMS if test else TRAIN_ITEMS
    places = ("garden shelf", "office shelf", "garden drawer", "kitchen table",
              "office cabinet", "front door") if not test else (
              "studio shelf", "attic shelf", "studio drawer", "porch table",
              "attic cabinet", "garden door")
    times = ("Monday morning at 8 AM", "Monday morning at 9 AM",
             "Tuesday evening at 9 PM", "Wednesday afternoon at 11 AM", "", "At noon")
    actions = ("moved", "placed", "usually", "carried", "stored")
    identity, context = [], []
    for i in range(n):
        actor, other_actor = rng.sample(people, 2)
        item, other_item = rng.sample(objects, 2)
        place, other_place = rng.sample(places, 2)
        kind_a = rng.choice(("location", "location", "habit"))
        act_a = rng.choice(actions)
        time_a = rng.choice(times)
        source_a = rng.choice(("Maybe", "No", "Reportedly", "")) + f" {actor} {act_a} {item} at {place}."
        a = Episode(actor=actor, item=item, location=place, action=act_a,
                    stated_time=time_a, kind=kind_a, novelty=bool(rng.randrange(3) == 0),
                    source=source_a.strip())
        pattern = i % 6
        # Identity / actor context / place context / same-label distractor / unconnected / changed identity.
        actor_b = actor if pattern in (0, 1, 5) else other_actor
        item_b = item if pattern in (0, 2, 3) else other_item
        place_b = place if pattern in (0, 2, 5) else other_place
        action_b = act_a if rng.randrange(3) == 0 else rng.choice(actions)
        time_b = time_a if rng.randrange(3) == 0 else rng.choice(times)
        source_b = rng.choice(("Maybe", "No", "Perhaps", "")) + f" {actor_b} {action_b} {item_b} at {place_b}."
        b = Episode(actor=actor_b, item=item_b, location=place_b, action=action_b,
                    stated_time=time_b, kind=rng.choice(("habit", "location")),
                    novelty=bool(rng.randrange(3) == 0), source=source_b.strip())
        identity.append((a, b, int(a.identity == b.identity)))
        if a.identity != b.identity:
            context.append((a, b, int(actor_b == actor or place_b == place)))
    # Explicit rare cues: temporal habit conflict and partially known location.
    exemplar = Episode(actor=people[0], item=objects[0], action="usually",
                       location=places[0], kind="habit", source="Usually stored at the first place.")
    changed = Episode(actor=people[0], item=objects[0], action="moved",
                      location=places[1], kind="location", source="Moved to a different place.")
    missing = Episode(actor=people[0], item=objects[0], action="moved",
                      location="", kind="note", source="Location was not specified.")
    identity.extend(((exemplar, changed, 1), (exemplar, missing, 1)))
    rng.shuffle(identity)
    rng.shuffle(context)
    return identity, context

def association_metrics(learner: LinkLearner, samples: list[tuple[Episode, Episode, int]], context: bool = False) -> dict:
    tp = fp = tn = fn = 0
    positive_scores, negative_scores = [], []
    for a, b, label in samples:
        score = (learner.context_probability(a, b) if context else learner.probability(a, b))
        if label:
            positive_scores.append(score)
        else:
            negative_scores.append(score)
        predicted = int(score >= 0.7)
        tp += int(label == 1 and predicted == 1)
        fn += int(label == 1 and predicted == 0)
        fp += int(label == 0 and predicted == 1)
        tn += int(label == 0 and predicted == 0)
    return {"accuracy": round((tp+tn)/len(samples), 4),
            "precision": round(tp/max(1, tp+fp), 4),
            "recall": round(tp/max(1, tp+fn), 4),
            "mean_positive_score": round(sum(positive_scores)/len(positive_scores), 4),
            "mean_negative_score": round(sum(negative_scores)/len(negative_scores), 4)}


def benchmark(learner: LinkLearner, stories: list[tuple[str, dict]], n: int = 100) -> dict:
    correct, attempts, visited, number_edges, parse_count = 0, 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as folder:
        with Memory(Path(folder) / "benchmark.db", learner) as memory:
            for paragraph, gold in stories[:n]:
                encoded = memory.ingest(paragraph)
                parse_count += sum(event.kind != "note" for event in encoded)
                subject = f"{gold['actor']}'s {gold['item']}"
                probes = [(f"Where are {subject} now?", gold["after"]),
                          (f"Where were {subject} before?", gold["before"]),
                          (f"What is unusual about {subject}?",
                           f"{gold['after']} differs from usual {gold['usual']}"),
                          (f"Why did {subject}?", None)]
                for q, expected in probes:
                    result = memory.recall(q)
                    attempts += 1
                    correct += int(result["answer"] == expected)
                    visited += len(result["activated"])
            counts = memory.counts()
            number_edges = counts["links"]
    return {"queries": attempts, "answer_accuracy": round(correct/max(1, attempts), 4),
            "extracted_events": parse_count, "mean_activated": round(visited/max(1, attempts), 3),
            "total_stored_events": 3 * min(len(stories), n), "learned_links": number_edges}


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach Sonara bounded associative memory")
    parser.add_argument("--model", default="model.json")
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train, _ = build_examples(seed=args.seed, n=args.examples)
    diverse_identity, diverse_context = build_signal_curriculum(
        seed=args.seed + 59, n=args.examples)
    heldout, stories = build_examples(seed=args.seed + 23, n=100, test=True)
    context_train = build_context_examples(seed=args.seed, n=args.examples)
    context_heldout = build_context_examples(seed=args.seed + 23, n=100, test=True)
    before = LinkLearner()
    learner = LinkLearner()
    learner.fit(train + diverse_identity)
    learner.fit_context(context_train + diverse_context)
    learner.save(args.model)
    report = {
        "training_pairs": len(train) + len(diverse_identity), "unseen_pairs": len(heldout),
        "context_training_pairs": len(context_train) + len(diverse_context),
        "context_unseen_pairs": len(context_heldout),
        "extra_association_signals": 50,
        "signals_observed_during_training": len({k for a, b, _ in diverse_identity
            for k, value in __import__('association_signals').brain_signals(a, b).items()
            if value > 0}),
        "before": association_metrics(before, heldout),
        "after": association_metrics(learner, heldout),
        "context_before": association_metrics(before, context_heldout, context=True),
        "context_after": association_metrics(learner, context_heldout, context=True),
        "memory_benchmark": benchmark(learner, stories),
        "limitations": ["Synthetic grammar only: paragraph extraction is not yet learned",
                        "Association scores are not validated calibrated probabilities",
                        "50 pairwise cues are inspired by memory computation, not 50 neural pathways",
                        "No autonomous code generation or LLM consultation yet"]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
