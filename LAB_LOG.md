# Lab log

One entry per run: what changed, result, what the wrong answers showed.

## 2026-10-17 · Run 0: pipeline smoke test (fixed baseline)

- **Changed:** first run. Baseline = cosine(Qwen question vector, memory vector) + 0.1 × recency (7-day half-life), untuned.
- **Data:** 12 hand-written notes (one per day), 9 questions with hand-labeled correct notes. Not saved; it tests the code only.
- **Result:** recall@5 = 9/9. Eight questions had the correct note at rank 1.
- **Wrong answers:** none. With only 12 memories, the top 5 covers 42% of everything, so recall@5 can't tell methods apart yet. Experiment 1 needs hundreds of memories before this metric means anything.
