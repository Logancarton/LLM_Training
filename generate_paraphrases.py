"""generate_paraphrases.py - Reword the benchmark's TRAINING templates with a local LLM.

Diagnosis (EXPERIMENT_WORLD_LEARNED_LEXICON_REPORT.md): with 4 move phrasings, the place
is always last and the person always first, so models learn positions instead of roles.
This script asks the local Ollama chat model to reword each TRAINING template, then keeps
a rewording only if it passes three checks:

  1. placeholders: exactly the template's placeholders, each exactly once
  2. meaning: filled with example names, the local model answers a factual question
     about it correctly (a move puts the object in the place; a distraction does not;
     a correction leaves the object in the earlier place; a question asks the same thing)
  3. held-out: it is not identical to any HELD-OUT test template

Held-out templates are never shown to the model. Output: paraphrased_templates.json,
which `experiment_world_generalization.py --systems ..._paraphrased` reads.

Usage:
    python3 generate_paraphrases.py --rounds 3 --output paraphrased_templates.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from typing import Dict, List

from experiment_world_generalization import CORRECTION, DISTRACT, MOVE, NOISE, QUESTIONS

MODEL = "qwen2.5:1.5b-instruct"
ENDPOINT = "http://127.0.0.1:11434/api/generate"
EXAMPLE = {"a": "jake", "o": "ball", "p": "shed", "pbad": "pond", "pgood": "shed",
           "t": "10:15", "t1": "10:15", "t2": "11:40"}

ROLE_TEXT = {
    "a": "{a} (a person)", "o": "{o} (an object)", "p": "{p} (a place)",
    "pbad": "{pbad} (the wrong place)", "pgood": "{pgood} (the right place)",
    "t": "{t} (a clock time)", "t1": "{t1} (a clock time)", "t2": "{t2} (a clock time)",
}
EVENT_MEANING = {
    "move": "{a} moved {o} so that {o} is now in {p}.",
    "distract": "{a} only paid attention to {o}; {o} did NOT move.",
    "noise": "{a} did something unrelated to any object or place.",
    "correction": "{o} never ended up in {pbad}; it is still in {pgood}.",
}


def generate(prompt: str, seed: int, temperature: float = 0.9, as_json: bool = True) -> str:
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": 700}}
    if as_json:
        body["format"] = "json"
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        return json.load(response)["response"]


def ask_yes_no(question: str) -> bool:
    answer = generate(question + "\nAnswer with one word: yes or no.", seed=0, temperature=0.0, as_json=False)
    return answer.strip().lower().startswith("yes")


def placeholders(template: str) -> List[str]:
    # Any brace group counts, so malformed ones like "{a's}" fail the placeholder check.
    return re.findall(r"\{([^{}]*)\}", template)


def normalize(template: str) -> str:
    return re.sub(r"[^a-z{} ]", "", template.lower()).strip()


def clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = text[0].lower() + text[1:] if text else text
    return text


def reword(template: str, kind: str, meaning: str, seed: int, n: int = 12) -> List[str]:
    roles = ", ".join(ROLE_TEXT[p] for p in dict.fromkeys(placeholders(template)))
    if kind == "question":
        instructions = "Keep it a question that asks for exactly the same information."
    else:
        instructions = ("Vary the verb and the word order: some should start with the place or time, "
                        "some should use passive voice, some should add extra words.")
    prompt = (f"Rewrite this sentence template in {n} different ways. Keep exactly the placeholders "
              f"{roles}, each used once, and keep the meaning: {meaning}\n{instructions}\n"
              f'Template: "{template}"\nWrite the placeholders literally in curly braces; '
              f'never replace them with real names or places.\nAnswer with only a JSON list of strings.')
    raw = json.loads(generate(prompt, seed))
    items = raw if isinstance(raw, list) else list(raw.values()) if isinstance(raw, dict) else []
    flat: List[str] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(x for x in item if isinstance(x, str))
        elif isinstance(item, str):
            flat.append(item)
    return [clean(x) for x in flat]


YES_NO = {"event_order", "membership"}
# Function words may be shared with held-out templates; content words may not.
SHARED_OK = {"the", "a", "an", "to", "in", "at", "of", "and", "by", "was", "did", "not", "is", "it"}


def words(template: str) -> set:
    return set(re.findall(r"[a-z]+", re.sub(r"\{[^{}]*\}", "", template.lower())))


def heldout_only_words(splits: Dict[str, List[str]]) -> set:
    """Content words that appear in HELD-OUT templates but not in TRAINING ones."""
    held = set().union(*map(words, splits["heldout"]))
    return held - set().union(*map(words, splits["train"])) - SHARED_OK


def question_form_ok(category: str, original: str, candidate: str) -> bool:
    """Yes/no questions must stay yes/no; wh-questions must keep their question word."""
    first = candidate.split()[0] if candidate.split() else ""
    if category in YES_NO:
        return first in {"is", "was", "did", "does", "do", "are", "were", "has", "had"}
    return first == original.split()[0] or original.split()[0] in candidate.split()[:2]


def meaning_ok(kind: str, original: str, candidate: str) -> bool:
    try:
        filled = candidate.format(**EXAMPLE)
    except (KeyError, IndexError, ValueError):
        return False
    if kind == "move":
        return ask_yes_no(f'Sentence: "{filled}"\nAfter this, is the ball in the shed?')
    if kind == "distract":
        return not ask_yes_no(f'Sentence: "{filled}"\nDid the ball actually end up in the shed?')
    if kind == "correction":
        return (ask_yes_no(f'Sentence: "{filled}"\nIs the ball in the shed?')
                and not ask_yes_no(f'Sentence: "{filled}"\nIs the ball in the pond?'))
    if kind == "noise":
        return True
    return ask_yes_no(f'Question A: "{original.format(**EXAMPLE)}"\nQuestion B: "{filled}"\n'
                      "Do questions A and B ask for exactly the same information?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3, help="generation calls per template (different seeds)")
    ap.add_argument("--output", default="paraphrased_templates.json")
    ap.add_argument("--only", default="", help="comma-separated groups to (re)generate; others are kept from --output")
    args = ap.parse_args()

    groups: Dict[str, tuple] = {
        "move": ("event", MOVE), "distract": ("event", DISTRACT),
        "noise": ("event", NOISE), "correction": ("event", CORRECTION),
    }
    for cat, splits in QUESTIONS.items():
        groups[cat] = ("question", splits)
    heldout = {normalize(t) for _, splits in groups.values() for t in splits["heldout"]}
    only = [g for g in args.only.split(",") if g]
    previous = {"templates": {}, "stats": {}}
    if only:
        with open(args.output) as fh:
            previous = json.load(fh)
        groups = {k: v for k, v in groups.items() if k in only}

    result: Dict[str, List[str]] = dict(previous["templates"])
    stats: Dict[str, Dict[str, int]] = dict(previous["stats"])
    t0 = time.monotonic()
    for key, (kind, splits) in groups.items():
        kept: List[str] = []
        seen = {normalize(t) for t in splits["train"]}
        counts = {"generated": 0, "bad_placeholders": 0, "duplicate": 0, "heldout_match": 0,
                  "heldout_word": 0, "wrong_form": 0, "wrong_meaning": 0, "kept": 0}
        for template in splits["train"]:
            meaning = (EVENT_MEANING[key] if kind == "event"
                       else f"the question asks: {template}")
            want = sorted(placeholders(template))
            for r in range(args.rounds):
                for cand in reword(template, kind if kind == "question" else key, meaning, seed=r + 1):
                    counts["generated"] += 1
                    norm = normalize(cand)
                    if sorted(placeholders(cand)) != want:
                        counts["bad_placeholders"] += 1
                    elif norm in seen:
                        counts["duplicate"] += 1
                    elif norm in heldout:
                        counts["heldout_match"] += 1
                    elif words(cand) & heldout_only_words(splits):
                        counts["heldout_word"] += 1
                    elif kind == "question" and not question_form_ok(key, template, cand):
                        counts["wrong_form"] += 1
                    elif not meaning_ok(key if kind == "event" else "question", template, cand):
                        counts["wrong_meaning"] += 1
                    else:
                        if kind == "question" and not cand.endswith("?"):
                            cand = cand.rstrip(".") + "?"
                        if kind == "event" and not cand.endswith("."):
                            cand += "."
                        kept.append(cand)
                        counts["kept"] += 1
                    seen.add(norm)
        result[key] = kept
        stats[key] = counts
        print(f"{key:22s} " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    with open(args.output, "w") as fh:
        json.dump({"model": MODEL, "rounds": args.rounds, "stats": stats, "templates": result}, fh, indent=2)
    print(f"Wrote {args.output} in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
