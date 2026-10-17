# Sonara

A small learner that builds on experience over time. Qwen (run locally through Ollama) handles language; Sonara remembers what happened and when, and learns what matters. See [SONARA_PLAN.md](SONARA_PLAN.md) for the plan and how success is measured.

## Files

- `sonara_memory.py`: append-only, timestamped memory of events, with retrieval and a CLI.
- `sonara_text_encoder.py`: frozen Qwen embedding vectors from local Ollama (loopback only), cached.
- `test_sonara_memory.py`: memory tests.

## Setup

```bash
ollama pull qwen3-embedding:0.6b
python3 test_sonara_memory.py
```

Earlier world-model experiments and their reports are in git history (branch `research/world-model-learned-lexicon`).
