# Sonara World Model — Experiment 3: Learned Vocabulary, No Hand-Computed Features

**Date:** 2026-10-17 · **Code:** `experiment_world_generalization.py` (v2), `sonara_world_model.py`, `sonara_text_encoder.py`
**Data:** `artifacts/world_generalization_v2_report.json` · 3 seeds (1000–1002), 900 steps, 240 training stories

## Verdict

**The fixes made the model worse.** The new design loses to the original on every test, by 14–23 points on three of the four. Extra training phrasings did not help the original design either. The original model still loses to the rule script on everything except reworded events. The live default stays as it was: `/ask` answers `UNKNOWN`, and the new design remains opt-in.

**Follow-ups (below):** 3,000 steps did not close the gap, and a frozen Qwen encoder did worse on reworded events. Both new designs memorise which training words are not answers; the root cause is training data with too few phrasings.

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

## Follow-up 1: 3,000 training steps (same seeds and tests)

| System | Training stories | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Rule script | 91.2% | **90.9%** | 48.1% | **46.9%** | **88.7%** |
| Original | 90.4% | 90.2% | **57.2%** | 29.0% | 88.1% |
| New design | **93.1%** | 85.7% | 40.9% | 24.6% | 81.0% |

**Not under-training: memorising.** With more steps the new design fits its training stories better than the original and catches up on new names (67% → 86%), but reworded events barely move (37% → 41%). The original roughly ties the rule script on A and D at 3,000 steps; reworded text does not improve.

## Follow-up 2: frozen Qwen encoder (`lexicon="pretrained"`)

The original design with only word and sentence reading replaced: each word and each sentence gets a frozen vector from `qwen3-embedding:0.6b` (local Ollama, `sonara_text_encoder.py`) through a trainable projection. Features, training and tests are unchanged.

| System (900 steps) | A | B | C | D |
|---|---:|---:|---:|---:|
| Original | **84.5%** | **59.7%** | **33.3%** | **82.4%** |
| Qwen encoder | 76.5% | 30.9% | 27.9% | 76.2% |

Qwen did help reworded *questions* by type: "was X earlier than Y?" 8% → 80%, the conditional state question 0% → 23%. It hurt reading reworded *events*: "where was it first?" fell from 80% to 27% on B.

## Diagnosis: why the new designs fail

Error breakdown on B for retrained seed-1000 models:

| Wrong answer type | Original | Qwen encoder |
|---|---:|---:|
| A filler word ("away", "down", "set", "put") | 1% | **34%** |
| Wrong place | 7% | 4% |
| Wrong object or person | 15% | 22% |
| Correct | 69% | 29% |

1. **The Qwen model tracks the world but copies the wrong word.** It rarely picks the wrong place; it picks filler words. "away" (89 times) is the last word of the held-out template "dropped off the {o} at the {p} and walked away".
2. **The learned entity gate learned nothing.** It scores every token 0.84–0.91, including "the", "set" and "down". Nothing in training forces it to: every training phrasing has a fixed small set of non-answer words.
3. **So the model learned a shortcut: "the answer is near the end, unless it's one of these training words."** The new designs memorise which specific words aren't answers (training fit 93%), and unseen filler words pass. The original's fixed `STRUCTURAL_WORDS` list does that job by hand, and collapses all unfamiliar words to the same representation, which limits the damage.
4. **Single-word Qwen vectors are weak.** In raw form "off" and "away" are more similar to the training places than "studio" is; after removing the shared direction, places separate only slightly (0.15 vs about 0). Whole-sentence vectors are strong (reworded twin 0.92 vs different facts 0.71), but this model's copy step works word by word.

**Root cause: the training data, not the network.** 240 stories from 4 move templates cannot teach "what counts as a place" in general; any learner extracts the template's positional shortcut, which is what the rule script encodes by hand.

## Next step

Make the shortcut stop working in training: generate dozens of varied phrasings per event type (for example by paraphrasing training events with the local Qwen chat model), with answers in different positions and many different filler words, while keeping B and C held out. Retest the original and Qwen designs on that data before changing the architecture again.

## Reproduce

```bash
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 5 \
    --output artifacts/world_generalization_v2_report.json
python3 experiment_world_generalization.py --seeds 3 --steps 3000 --workers 6 \
    --systems world_model,learned --output artifacts/world_generalization_v2_3000steps.json
ollama pull qwen3-embedding:0.6b   # once; embeds ~18k texts on first run (~15 min)
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 2 \
    --systems pretrained --output artifacts/world_generalization_pretrained.json
```
