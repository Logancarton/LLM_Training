# Sonara

A small learner that builds on experience over time. Qwen (run locally through Ollama) handles language; Sonara remembers what happened and when, and learns what matters. See [SONARA_PLAN.md](SONARA_PLAN.md) for the plan and how success is measured.

## Files

- `sonara_memory.py`: append-only, timestamped memory of events, with retrieval and a CLI.
- `sonara_text_encoder.py`: frozen Qwen embedding vectors from local Ollama (loopback only), cached.
- `sonara_recall.py`: chat with Qwen while logging experience, the fixed recall baseline, and recall@5 scoring.
- `sonara_story.py`: a timestamped story (synthetic) to test whether Sonara learns what time means (experiment 3).
- `sonara.py`: Sonara, a linear scorer trained on your feedback with day-clock inputs, plus the learning-curve report against the baseline and a fixed feedback rule.
- `test_sonara_memory.py`, `test_sonara_recall.py`, `test_sonara.py`: tests (no Ollama needed).
- `LAB_LOG.md`: one entry per run.

## Setup

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:1.5b-instruct
python3 test_sonara_memory.py && python3 test_sonara_recall.py && python3 test_sonara.py
python3 sonara_recall.py chat     # ask questions, mark useful memories; Sonara ranks in shadow
python3 sonara_recall.py score    # baseline recall@5 on the frozen eval questions
python3 sonara.py curve           # Sonara vs baseline, rule and control, in time order
```

Earlier world-model experiments and their reports are in git history (branch `research/world-model-learned-lexicon`).
