# Sonara: Experience Learner — Plan

**Date:** 2026-10-17 · **Status:** plan only, nothing built yet

## Goal

Qwen handles language: it reads text, turns it into vectors, and writes answers. Qwen is fixed and knows nothing about your timeline. **Sonara is a small network that learns from experience:** what happened when, which memories matter, and how things relate over time. It gets better the more it is used.

| | Qwen (fixed) | Sonara (learns) |
|---|---|---|
| Job | Read, embed, answer | Remember, relate, recall |
| Changes over time | No | Every verified interaction |
| Cost per question | One embedding + one answer | A tiny network over stored vectors |

## The one question for experiment 1

**Does Sonara recall the right memories better *because of experience* than a fixed method that never learns?**

- **Memory:** every event is stored once in `sonara_memory.py` (append-only, timestamped) with its Qwen vector from `sonara_text_encoder.py`. Text is never re-read.
- **Fixed baseline:** rank memories by Qwen vector similarity to the question, plus recency. No learning.
- **Sonara:** a small network scores each memory from the question vector, the memory vector, the time gap, and what past feedback said about similar memories. It is trained only from verified feedback (a recalled memory was confirmed useful, or not).
- **Qwen answers** from the top memories Sonara returns. Answer quality is measured, but recall is the target.

## How it is measured (decided before building)

1. **Frozen evaluation set:** questions with known correct memories, written before any training and never used for training.
2. **Time split:** train on earlier experience, test on later questions. Repeat on 3 different split points.
3. **Metric:** recall@5 (is a correct memory in the top 5?), plus cost per question.
4. **Success:** Sonara beats the fixed baseline by at least 5 points recall@5 on all 3 splits, and the gap grows as experience grows. Anything less is reported as no effect.
5. **Every run:** change one thing; read 20 wrong answers before drawing conclusions; add one entry to a lab log.

## Open decision before building

Where does experience come from? Real logged experience is slow to accumulate but honest; synthetic stories are fast but taught us today that models learn the generator's shortcuts. Recommended: log real notes for 2–4 weeks while building the pipeline, and use synthetic data only to test that the code runs.

## What the earlier experiments taught us

1. A fair baseline gets exactly the information the model gets. The old rule script knew the test questions and looked 6× better than it was.
2. A small LLM with no training (65% on reworded text) beat every model trained here. Don't rebuild language understanding; use it.
3. Small networks trained on a few templates learn word positions, not meaning.
4. Whole-sentence vectors carry meaning well; single-word vectors are weaker.
5. Reading actual wrong answers found the real cause faster than any aggregate number.

The full experiment history is in git (branch `research/world-model-learned-lexicon`, commit `dbaca76`).
