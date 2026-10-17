"""experiment_llm_zero_shot.py - Score a local chat LLM on the generalization benchmark.

The model reads each test story and question and answers directly: no training, no
templates, no task-specific rules. This is the ceiling to compare the trained world
model against. Same test stories and exact-match scoring as experiment_world_generalization.py.

Usage:
    python3 experiment_llm_zero_shot.py --model qwen2.5:1.5b-instruct --seeds 3 \
        --output artifacts/llm_zero_shot_qwen.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from experiment_world_generalization import build_splits, score

ENDPOINT = "http://127.0.0.1:11434/api/generate"
INSTRUCTIONS = (
    "Read the story, then answer the question using only the story. Events are listed in "
    "time order; a line starting with 'correction:' fixes an earlier event.\n"
    "Answer with ONE word only: a name, object or place copied from the story, or yes, "
    "or no, or unknown if the story does not say."
)


def ask(model: str, events: List[str], question: str) -> str:
    prompt = f"{INSTRUCTIONS}\n\nStory:\n" + "\n".join(events) + f"\n\nQuestion: {question}\nAnswer:"
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_predict": 8, "num_ctx": 2048}}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)["response"]


def normalize(raw: str) -> str:
    words = re.findall(r"[a-z]+", raw.lower())
    if not words:
        return "unknown"
    if words[0] == "the" and len(words) > 1:
        return words[1]
    return words[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-stories", type=int, default=60, help="stories per condition (benchmark uses 60)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", default="artifacts/llm_zero_shot.json")
    args = ap.parse_args()

    report: Dict = {"model": args.model, "instructions": INSTRUCTIONS, "runs": []}
    t0 = time.monotonic()
    for seed in [1000 + i for i in range(args.seeds)]:
        _, tests = build_splits(seed, 1, 60)
        run: Dict = {"seed": seed}
        for cond, stories in tests.items():
            stories = stories[:args.test_stories]
            items = [(s.events, q.question) for s in stories for q in s.questions]
            with ThreadPoolExecutor(args.threads) as pool:
                answers = list(pool.map(lambda it: normalize(ask(args.model, *it)), items))
            lookup = {(tuple(e), q): a for (e, q), a in zip(items, answers)}
            run[cond] = score(lambda q, ev: lookup[(tuple(ev), q)], stories)
            print(f"seed {seed} {cond}: {run[cond]['accuracy']:.1%} (n={run[cond]['n']}, "
                  f"{time.monotonic() - t0:.0f}s)", flush=True)
        report["runs"].append(run)
    conds = [c for c in report["runs"][0] if c != "seed"]
    report["mean"] = {c: round(sum(r[c]["accuracy"] for r in report["runs"]) / len(report["runs"]), 4) for c in conds}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("mean:", {c[:1]: f"{v:.1%}" for c, v in report["mean"].items()})


if __name__ == "__main__":
    main()
