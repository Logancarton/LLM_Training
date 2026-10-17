"""Step 8: A/B evaluation. Does the state brief make answers better?

Each scripted conversation runs twice from a clean start:
  arm "loop":     the full loop (brief + action selection + learning)
  arm "baseline": the same generator with the same recent history, but no brief
Both arms see the same recent-history window, so the only difference is the brief.
Sessions are separated by a simulated day, so corrections must survive beyond the history.

Output: eval/results/<time>.md with the two replies per turn in random order (A/B), and a
separate key file. Rate the pairs blind, then open the key.

    python3 eval_ab.py

Needs a real generator (Ollama running locally) to be meaningful.
"""
from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path

import config
import llm
from brain import Brain
from evidence import Evidence
from generate import generate

HERE = Path(__file__).resolve().parent
DAY = 24 * 3600


def run_loop(script: dict, start: float) -> list:
    tmp = Path(tempfile.mkdtemp())   # isolated: eval turns never touch the real evidence store
    brain = Brain(tmp / "eval.db", evidence=Evidence(tmp / "mem"))
    out = []
    for s_i, session in enumerate(script["sessions"]):
        for t_i, text in enumerate(session):
            r = brain.handle_turn(text, now=start + s_i * DAY + t_i * 60)
            out.append({"text": text, "reply": r["response"], "action": r["action"]})
    return out


def run_baseline(script: dict) -> list:
    history, out = [], []
    for session in script["sessions"]:
        for text in session:
            try:
                reply = generate(text, "", "ANSWER", history[-2 * config.HISTORY_TURNS:])
            except llm.LLMError as exc:
                reply = f"[generator unavailable: {exc}]"
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            out.append({"text": text, "reply": reply})
    return out


def main() -> None:
    scripts = json.loads((HERE / "eval" / "scripts.json").read_text())
    provider = llm.resolve_provider(config.GENERATOR_PROVIDER)
    if provider == "mock":
        print("Note: generator is 'mock', so replies are placeholders. Start Ollama "
              "for a real evaluation.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = HERE / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    lines, key = [f"# Blind A/B evaluation {stamp}", "",
                  "For each turn, pick the better reply (A, B, or tie). "
                  "Open the key file only after rating.", ""], {}
    rng = random.Random(stamp)
    for script in scripts:
        loop = run_loop(script, time.time())
        base = run_baseline(script)
        lines += [f"## {script['name']}", ""]
        for i, (lt, bt) in enumerate(zip(loop, base), 1):
            loop_is_a = rng.random() < 0.5
            a, b = (lt["reply"], bt["reply"]) if loop_is_a else (bt["reply"], lt["reply"])
            item = f"{script['name']} / turn {i}"
            key[item] = {"A": "loop" if loop_is_a else "baseline", "loop_action": lt["action"]}
            lines += [f"### Turn {i}", f"**User:** {lt['text']}", "",
                      f"**A:** {a}", "", f"**B:** {b}", "", "Better: ___", ""]
    (results_dir / f"{stamp}.md").write_text("\n".join(lines))
    (results_dir / f"{stamp}-key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote eval/results/{stamp}.md (and the key file beside it).")
    print("v0 success bar: the loop wins at least 12 of 20 rated pairs.")


if __name__ == "__main__":
    main()
