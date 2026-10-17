# Sonara World Model — Experiment 3: Learned Vocabulary, No Hand-Computed Features

**Date:** 2026-10-17 · **Code:** `experiment_world_generalization.py` (v2), `sonara_world_model.py`, `sonara_text_encoder.py`, `generate_paraphrases.py`, `experiment_llm_zero_shot.py`
**Data:** `artifacts/world_generalization_v2_report.json` · 3 seeds (1000–1002), 900 steps, 240 training stories

## Verdict (corrected)

**Correction.** Earlier versions of this report (and Experiment 2) said the models lose to the rule script on reworded questions (C). That comparison was unfair: the rule script hard-codes the *held-out* question wordings, so on C it knew the test questions in advance. Restricted to the training question wordings, which is all a model sees, it scores **7.4%** on C, not 46.9%. On A and D it is hand-written for the exact training phrasings, so beating it there was never a meaningful bar.

| System (mean of 3 seeds) | A: new names | B: reworded events | C: reworded events + questions | D: longer stories |
|---|---:|---:|---:|---:|
| Rule script, training wording only (fair) | **90.9%** | 48.1% | 7.4% | **88.7%** |
| Rule script, knows held-out questions | 90.9% | 48.1% | 46.9% | 88.7% |
| Original world model | 84.5% | 59.7% | 33.3% | 82.4% |
| Original + LLM-reworded training | 74.1% | 60.2% | 40.4% | 71.2% |
| Qwen encoder + gate + reworded training | 59.6% | 49.5% | 39.9% | 56.2% |
| **Qwen 2.5 1.5B answering directly, no training** | 64.5% | **65.2%** | **56.2%** | 59.7% |

1. **On fair comparisons of reworded text, the original world model beats the rule script:** B 59.7% vs 48.1%, C 33.3% vs 7.4%. Every trained design beats it on C (23–40%); on B the weaker new designs do not (31–49%).
2. **A small general-purpose LLM, untrained on this task, beats every trained model on reworded text** (B 65%, C 56%). Its accuracy barely changes with phrasing (A 64.5% vs B 65.2%), which is what reading rather than template matching looks like. The trained models are template specialists: best on the original phrasings, weaker as wording changes.
3. **None of the architecture changes beat the original world model.** The learned-vocabulary design, the frozen Qwen encoder and the answer gate all scored below it. LLM-reworded training data helped reworded questions (C +7 to +11 points) but cost 10+ points on the original phrasings.

The live default stays as it was: `/ask` answers `UNKNOWN`, and all new designs remain opt-in.

## Zero-shot LLM baseline

`experiment_llm_zero_shot.py` sends each test story and question to the local `qwen2.5:1.5b-instruct` with a short generic instruction (answer with one word from the story, or yes/no/unknown) and scores it with the same exact-match check. No training, examples or template knowledge. About 9,300 questions in 17 minutes.

Per seed: A 63.4 / 61.1 / 68.9, B 65.9 / 63.6 / 66.1, C 56.0 / 55.8 / 56.9, D 59.2 / 59.9 / 59.8. Weak spots: it rarely says "unknown" when information is missing (C missing info 20%), and it is weaker on state tracking than on lookup ("where is it now" 47–56%, "where was it before 10:15" 43–50%, "what did X put in the Y" 68–85%). Exact-match scoring is strict for a chat model, so these are lower bounds.

## Experiment 3 results (mean of 3 seeds; identical test stories for every system)

| System | A: new names | B: reworded events | C: reworded events + questions | D: longer stories |
|---|---:|---:|---:|---:|
| Rule script, knows held-out questions | **90.9%** | 48.1% | **46.9%** | **88.7%** |
| Original model | 84.5% | **59.7%** | 33.3% | 82.4% |
| Original + extra training phrasings | 77.4% | 59.7% | 27.9% | 75.9% |
| New design (all four fixes) | 66.9% | 36.7% | 22.9% | 63.4% |

Per seed, B: original 69.3 / 54.4 / 55.3; rule script 49.3 / 45.7 / 49.2.

## What the numbers show

1. **One real gain from the original model:** on reworded events it beats the rule script on all 3 seeds (by 5–20 points). In Experiment 2 the two had tied on average.
2. **The new design is under-trained or under-powered.** It picked its final step (900) on every seed, so it was still improving when training stopped. Without the hand-computed features, 900 steps on 240 stories is not enough to learn "latest mention" or "just before a time" from scratch.
3. **Extra phrasings cost accuracy elsewhere.** They neither changed reworded-event accuracy nor helped reworded questions, and they lowered new-name accuracy by 7 points.
4. **Reworded questions remain the weakest condition** (23–33%). These numbers sit below the rule script's 47% only because that script knew the held-out questions; see the correction above.

## What was built (opt-in, default unchanged)

`SonaraWorldModel(lexicon="learned", engineered_features=False)`: every training word gets its own learned meaning; a learned gate decides which words name things (replacing `STRUCTURAL_WORDS`); 20% of words per story are randomly hidden during training; and words never trained on are read as unknown. Hand-computed features are replaced by event scoring from the model's running memory. Checkpoints record the design, and a mismatched one is refused. Four new unit tests cover these paths.

## Follow-up 1: 3,000 training steps (same seeds and tests)

| System | Training stories | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Rule script, knows held-out questions | 91.2% | **90.9%** | 48.1% | **46.9%** | **88.7%** |
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

## Follow-up 3: answer gate for the Qwen design

A linear probe on each candidate word's frozen Qwen vector predicts whether the word can be an answer; its log-probability is added to the copy step's scores. It is trained only from words that were verified answers (`answer_words`, saved in checkpoints), with no word list.

| System (900 steps) | A | B | C | D |
|---|---:|---:|---:|---:|
| Original | **84.5%** | **59.7%** | **33.3%** | **82.4%** |
| Qwen encoder | 76.5% | 30.9% | 27.9% | 76.2% |
| Qwen encoder + answer gate | 71.7% | 35.2% | 28.8% | 66.2% |

**The gate fixed what it targeted, but not the score.** On B (seed 1000), filler-word answers fell from 34% to 13%. Those answers mostly became the *wrong entity* instead (wrong object 16%, wrong person 13%), not correct ones. Actor and object questions improved (B actor 16% → 31%, object 19% → 49%); location questions did not (current state 19% → 19%). A and D got worse.

**The remaining failure is role confusion.** In "in the attic, emma set down the scarf", all three of attic, emma and scarf are valid answers to *some* question; the model has to know which one is the place. In training, roles always sit in the same positions (person first, object after the verb, place last), so it learned positions, not roles. That is the same training-data limit as above, and no gate can fix it.

## Follow-up 4: training on LLM-reworded templates

`generate_paraphrases.py` asked the local `qwen2.5:1.5b-instruct` to reword every TRAINING template, and kept a rewording only if (1) its placeholders were exactly right, (2) filled with example names, the same model answered a factual check correctly (a move leaves the ball in the shed; a distraction does not; a reworded question asks the same thing), (3) yes/no questions stayed yes/no, and (4) it neither matched a held-out template nor used a content word found only in held-out templates. Held-out templates were never shown to the model. 171 rewordings were kept (`paraphrased_templates.json`): 65 moves, 36 noise, 22 distractions, 2 corrections, and 2–9 per question type. Tests are unchanged.

| System (900 steps) | A | B | C | D |
|---|---:|---:|---:|---:|
| Rule script, knows held-out questions | **90.9%** | 48.1% | **46.9%** | **88.7%** |
| Original | 84.5% | 59.7% | 33.3% | 82.4% |
| Original + reworded training | 74.1% | **60.2%** | 40.4% | 71.2% |
| Qwen + gate | 71.7% | 35.2% | 28.8% | 66.2% |
| Qwen + gate + reworded training | 59.6% | 49.5% | 39.9% | 56.2% |

1. **Reworded training is the first change that helps reworded questions.** C rose 7 points for the original and 11 for Qwen; "was X before Y?" went from 8% to 68% for the original.
2. **It fixed most of the Qwen design's reworded-event gap** (B 35% → 50%), but Qwen still trails the original everywhere.
3. **The original-phrasing tests got worse for both** (A down 10–12 points, D down 10–11). The four original move phrasings are now 4 of 69 in training, so the A and D phrasing is seen far less often.
4. **Caveats.** About half the kept move rewordings use passive voice ("was moved ... by"), which one held-out B template also uses, so B is easier than before. Several runs selected their final step (900), so these models may still be improving. Per-seed spread remains wide (original B: 72 / 55 / 54).

**Conclusion:** varied training text helps; the frozen Qwen encoder still does not. The original design trained on reworded data is the best trained model on reworded questions.

## Next step

The zero-shot result changes the question. A 1.5B general LLM already reads reworded stories better than any model trained here, so further tuning of this small network is unlikely to be the best use of time. Options, in order of expected value:

1. **Use the LLM as the reader** over Sonara's verified evidence store, and score it on this benchmark with an abstention check, since it rarely answers "unknown" on its own.
2. **Try a larger local model zero-shot** (for example `llama3.1:8b`, already installed) to see how much headroom there is.
3. If the trained world model stays in scope: mix original and reworded templates 50/50 so A and D do not regress, generate more question rewordings, and train longer.

## Reproduce

```bash
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 5 \
    --output artifacts/world_generalization_v2_report.json
python3 experiment_world_generalization.py --seeds 3 --steps 3000 --workers 6 \
    --systems world_model,learned --output artifacts/world_generalization_v2_3000steps.json
ollama pull qwen3-embedding:0.6b   # once; embeds ~18k texts on first run (~15 min)
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 2 \
    --systems pretrained --output artifacts/world_generalization_pretrained.json
# (that run predates the answer gate; the same command now includes it)
python3 generate_paraphrases.py --rounds 3          # local qwen2.5:1.5b-instruct, ~4 min
python3 experiment_world_generalization.py --seeds 3 --steps 900 --workers 6 \
    --systems world_model_paraphrased,pretrained_paraphrased \
    --output artifacts/world_generalization_paraphrased.json
python3 experiment_llm_zero_shot.py --model qwen2.5:1.5b-instruct --seeds 3 \
    --output artifacts/llm_zero_shot_qwen.json   # ~17 min
```
The fair rule baseline (`rule_baseline_train_wording`) is reported by every benchmark run.
LLM generation is not bit-for-bit reproducible across Ollama versions; the committed `paraphrased_templates.json` is the exact set used.
