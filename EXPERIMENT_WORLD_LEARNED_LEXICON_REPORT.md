# Sonara World Model — Experiment 3: Learned Vocabulary, No Hand-Computed Features

**Date:** 2026-10-17 · **Code:** `experiment_world_generalization.py` (v2), `sonara_world_model.py`
**Data:** `artifacts/world_generalization_v2_report.json` · 3 seeds (1000–1002), 900 steps, 240 training stories

## Verdict

**The fixes made the model worse.** The new design loses to the original on every test, by 14–23 points on three of the four. Extra training phrasings did not help the original design either. The original model still loses to the rule script on everything except reworded events. The live default stays as it was: `/ask` answers `UNKNOWN`, and the new design remains opt-in.

## Results (mean of 3 seeds; identical test stories for every system)

| System | A: new names | B: reworded events | C: reworded events + questions | D: longer stories |
|---|---:|---:|---:|---:|
| Rule script, no learning | **90.9%** | 48.1% | **46.9%** | **88.7%** |
| Original model | 84.5% | **59.7%** | 33.3% | 82.4% |
| Original + extra training phrasings | 77.4% | 59.7% | 27.9% | 75.9% |
| New design (all four fixes) | 66.9% | 36.7% | 22.9% | 63.4% |

Per seed, B: original 69.3 / 54.4 / 55.3; rule script 49.3 / 45.7 / 49.2.

## What the numbers show

1. **One real gain from the original model:** on reworded events it beats the rule script on all 3 seeds (by 5–20 points). In Experiment 2 the two had tied on average.
2. **The new design is under-trained or under-powered.** It picked its final step (900) on every seed, so it was still improving when training stopped. Without the hand-computed features, 900 steps on 240 stories is not enough to learn "latest mention" or "just before a time" from scratch.
3. **Extra phrasings cost accuracy elsewhere.** They neither changed reworded-event accuracy nor helped reworded questions, and they lowered new-name accuracy by 7 points.
4. **Reworded questions remain the failure point.** Every model scores below the rule script there (23–33% vs 47%).

## What was built (opt-in, default unchanged)

`SonaraWorldModel(lexicon="learned", engineered_features=False)`: every training word gets its own learned meaning; a learned gate decides which words name things (replacing `STRUCTURAL_WORDS`); 20% of words per story are randomly hidden during training; and words never trained on are read as unknown. Hand-computed features are replaced by event scoring from the model's running memory. Checkpoints record the design, and a mismatched one is refused. Four new unit tests cover these paths.

## Next step

Before adding more architecture, test whether the new design is simply under-trained: run it for 3,000 steps on more stories. If it still trails the original, the hand-computed features are doing work that this network cannot learn at this size, and a pretrained text encoder (e.g. the local Qwen model) is the more realistic route to handling reworded questions.

## Reproduce

```bash
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 5 \
    --output artifacts/world_generalization_v2_report.json
```

About 25 minutes on 8 cores.
