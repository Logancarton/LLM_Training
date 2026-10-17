# Sonara World Model — Experiment 2: Generalization Benchmark

**Date:** 2026-10-15 (America/Phoenix)
**Code:** `experiment_world_generalization.py` · **Data:** `artifacts/world_generalization_report.json`
**Scope:** 2 seeds (1000, 1001). The run was stopped before the third seed to save time; no seed was dropped based on its results.
**Model under test:** `sonara_world_model.py` as of this date, unmodified.

## Verdict

**Learned event understanding is not established.** On average the trained world model never beats a rule script that does no learning. Its accuracy falls to 28% when questions are reworded, and it loses another 6–14 points when the hand-computed features are switched off. The Experiment 1 result (100% held-out) came from the benchmark's design, not from the model's ability. The live default, where `/ask` answers `UNKNOWN`, remains the correct setting.

## Results (mean of 2 seeds; every test uses names never seen in training)

| System | A: new names | B: reworded events | C: reworded events + questions | D: 2× longer stories |
|---|---:|---:|---:|---:|
| Rule script, no learning | **92.4%** | 47.0% | **46.7%** | **89.1%** |
| Majority answer per category | 7.7% | 8.1% | 8.2% | 7.6% |
| **World model (trained)** | 80.2% | **48.0%** | 28.0% | 79.3% |
| World model, hand-computed features off | 74.2% | 34.1% | 17.4% | 69.0% |
| World model, event order shuffled | 61.5% | — | — | — |

Per seed, world model: A 85.4 / 74.9, B 62.4 / 33.6, C 28.9 / 27.0, D 84.8 / 73.9.
Per seed, features off: A 76.6 / 71.8, B 36.0 / 32.2, C 13.1 / 21.8, D 73.6 / 64.3.

For reference, on Experiment 1's own held-out set the original rule script scores **98.7%** (`--score-experiment1`).

## What the benchmark varies that Experiment 1 held fixed

- **Story structure:** 2–3 objects, 2–3 actors, places in random roles, interleaved moves, random timestamps, distractor events that mention an object or a place without moving anything (including negated moves: "wanted to take the ball to the garden but stopped"), and optional corrections.
- **Balanced answers:** yes/no questions are 50/50. "Unknown" questions ask about an object or a move that is genuinely absent from the story; there is no fixed "what did the cat chase" question.
- **Held-out phrasing:** event and question templates are split into training and held-out sets. Held-out events put the place first ("in the attic, emma set down the scarf"), use passive voice ("the journal was taken to the garage by mia"), or put the place mid-sentence ("dropped off the scarf at the attic and walked away").
- **Training procedure:** the model trains with its own `learn_step` (AdamW plus replay) for 900 steps. Every 150 steps it is scored on validation stories built from training names and phrasings, never test data, and the best checkpoint is kept. Without this, accuracy is unstable: in a trial run it peaked near 600 steps and fell by 900.

## Mechanism: what the numbers show

1. **New names (A, D): template matching beats the model.** When the sentence structure matches training, positional rules ("the place is the last word of the latest sentence mentioning the object") reach 89–92%. The model gets 79–80%. Experiment 1 tested only this condition, and in its single-template form, so it measured how well the model reproduces the rule.
2. **Reworded events (B): a tie on average, and an unstable one.** The rule script drops to 47% because "last word = place" breaks. The model averages 48%, but its two seeds were 62% and 34%. One seed found a representation that partly transfers to new phrasing and the other did not. A real but unreliable signal.
3. **Reworded questions (C): collapse.** The model falls to 28%, below the rule script (which is given both question phrasings). The cause is structural. Word meanings are learned only for `STRUCTURAL_WORDS`, a fixed list holding the original benchmark's vocabulary. New question words like "room", "currently" and "start" are reduced to spelling fragments and treated as candidate answers.
4. **The hand-computed features carry much of the load.** Zeroing `event_temporal_features`, `event_local_features` and `global_rel_features` costs 6 points on A and 10–14 points on B, C and D. These features compute "latest event mentioning X", "earliest event mentioning X" and "event just before timestamp T" in ordinary code, so the network is choosing among precomputed answers rather than deriving them.
5. **Event order matters a little.** Shuffling event order costs 19 points on A (80% to 62%). Order is used, but much of the score does not depend on it.

## Implication

The current design is a question reader that re-reads the whole event list for each question, with its vocabulary and reasoning steps fitted to one template. It is not yet a model that keeps track of the world's state as events arrive. Building further Sonara layers on top of it would rest on a capability that has not been demonstrated.

## Recommended next steps (model side)

1. **Learn the vocabulary instead of hard-coding it.** Give every training word a learned embedding and remove the `STRUCTURAL_WORDS` gate. This is the most likely fix for condition C.
2. **Remove the hand-computed progression features,** or keep them only as a labeled ablation, and let the network's recurrent state tracker earn those answers itself.
3. **Train on more phrasing variety than you test on,** and keep B/C as the main score. A change counts as progress only if it beats the rule script on B and C across at least 3 seeds without the features.
4. **Stabilize training** with learning-rate decay and checkpoint selection on validation data; the seed-to-seed spread (62% vs 34% on B) is as large as most effects of interest.

## Reproduce

```bash
python3 experiment_world_generalization.py --show-examples        # print one story per condition
python3 experiment_world_generalization.py --score-experiment1    # rule script on Experiment 1: 98.7%
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 2 \
    --output artifacts/world_generalization_report.json
```

Runtime: about 17 minutes per model on one CPU core; 9 models for 3 seeds. Results are saved after each model to `*.partial.json`. Expect it to run faster on the MacBook.

## Limitations

- 2 seeds only; per-category accuracies for the neural systems were not saved in this run (they will be in future runs). Rule-baseline category scores are in the JSON.
- Still synthetic: a richer set of templates, but researcher-written ones. Passing this benchmark would be necessary, not sufficient, for enabling live world reasoning.
- One model size and learning rate (the defaults); no hyperparameter search was done for any system.
