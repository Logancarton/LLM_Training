# Sonara: Experience Learner — Plan

**Date:** 2026-10-17 · **Status:** below. Nothing has been shown on real data yet (1 memory, 0 questions).

| Feature | Built | Integrated | Enabled | Validated on real data |
|---|---|---|---|---|
| Memory store + Qwen vectors | yes | `chat` | yes | no |
| Fixed baseline (tuned on train) | yes | `chat` answers from it | yes | no |
| Sonara learner (online; clock + experience inputs) | yes | `curve` only | in `curve` | no (synthetic only) |
| Top-1 target metric, recall@5 and `answer_ok` alongside | yes | `curve` | yes | no |
| Incremental `newer_similar` (saved, linear per memory) | yes | Sonara inputs | yes | matches N x N in tests |
| Saved learner state, replay on feature/code change | yes | `chat` | yes | saved == replayed in tests |
| Bounded candidate set (top 50 cosine + 10 newest), shared | yes | baseline, Sonara, `chat` | yes | no |
| Shadow ranking in `chat` (both rankings logged; learns live from train only) | yes | `chat` | yes | no |
| Union labeling pool (baseline top 5 + Sonara top 5 + 5 newest, proposers recorded) | yes | `chat` | yes | no |

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

## Sonara v1 design (proposed 2026-10-17, not built)

**Feedback signal.** Each train question you label gives positives (memories you marked useful) and negatives (memories shown in the 15-candidate pool but not marked). Memories that were never shown are unknown, not negative. `answer_ok` is recorded but not trained on, since it mostly measures Qwen.

**Scorer.** A linear layer (the smallest network that can be trained) scores each (question, memory) pair from 9 inputs:

| Input | Baseline has it? |
|---|---|
| cosine(question, memory) | yes |
| age at ask time, log days | yes |
| source, 4 one-hot values | yes |
| times this memory was marked useful before now | no: experience |
| times it was shown but not marked before now | no: experience |
| neighbor vote: max over earlier train questions q′ of cosine(question, q′), counting only q′ where this memory was marked useful | no: experience |

The last three are the "what past feedback said" part. They count only train-split feedback given before the question was asked. Eval labels never become inputs.

**Training (changed 2026-10-17: online).** A softmax over each train question's pool, with cross-entropy on its positives. Sonara learns one verified question at a time, in the order they were asked (Adam lr 0.01, 10 steps per question), and carries its weights forward. It is never shuffled and never retrained from scratch, so its progression lives in its weights. The question and memory vectors enter only through cosine similarity, because a learned 1024-dimension projection would overfit a few hundred labels. Learning on top of the raw vectors would be a later change, one at a time.

**Learning curve.** Every eval question is scored by each method as it stood at the second the question was asked. The baseline and rule are re-tuned on all earlier train questions every 10 labeled ones. Results are reported in 4 consecutive time periods. "The gap grows with experience" means Sonara's period results rise while the baseline's stay flat.

**No early verdict.** Sonara isn't expected to beat the baseline early on; improvement should be slow. Until there are about 150 eval questions, each run reports the curve (Sonara minus baseline top-1 vs. number of train questions) and makes no success or failure call. A fixed feedback-bonus rule (memories marked useful for similar past questions get a bonus whose size is tuned on train) is reported next to it as a reference. It shows how much of any gain comes from storing feedback rather than learning from it, but it isn't a success gate. A no-experience control (the same learner, with the three experience inputs always 0) is also reported. Sonara minus this control is the part of the gain that comes from experience, not from a stronger optimizer.

**Cost.** One matrix multiply over memories plus one over past questions, about 10 weights. Negligible next to one Qwen embedding.

## v2 direction: a day clock (decided 2026-10-17; clock and progression inputs built)

Time is one continuous progression, not just stamps on events. Every time Sonara runs, it reads "now" to the second, and every memory is a point on that same timeline. Nothing runs in the background; elapsed time and quiet periods are computed from the clock when Sonara is used. Event dates are not pulled out of the text.

- **Clock inputs:** where now sits and where each memory sits (time of day, day of week, time of year, as cycles), plus the elapsed time between them in log-seconds. Also each memory's absolute position: seconds since Sonara's first memory, in log scale.
- **Progression inputs:** how many sessions have started between the memory and now (a quiet hour or more starts a new session, judged from memory timestamps), and how similar the most similar newer memory is.
- **Growth:** add the learner's capacity only as labels grow, and keep the no-experience control at every step, so the curve shows whether time is actually learned.

## How it is measured (decided before building)

1. **Frozen evaluation set:** questions with known correct memories, written before any training and never used for training.
2. **Time order:** every eval question is tested only with what was learned from earlier train questions, as described in the learning curve above. Results are reported in 4 consecutive periods.
3. **Metric (changed 2026-10-17):** top-1 correctness (is the first-ranked memory a correct one?). Recall@5 and `answer_ok` are reported alongside, plus cost per question. Top-1 replaced recall@5 because recall@5 can't see whether the current fact beats its older version (Runs 1, 5). `answer_ok` only measures the system Qwen answered from (the baseline), so it is not a Sonara-vs-baseline comparison.
4. **Success:** Sonara beats the fixed baseline by at least 5 points top-1 in each of the last 3 periods, and the gap grows as experience grows. Anything less is reported as no effect.
5. **Every run:** change one thing; read 20 wrong answers before drawing conclusions; add one entry to a lab log.

## Where experience comes from (decided 2026-10-17)

Real use. You ask Qwen questions in `sonara_recall.py chat`. The baseline recalls memories, Qwen answers from them, and you mark which memories were actually useful. After every answer Qwen is asked: what makes this true, what would make it not true, and what are the most relevant variables and how they relate to the question and answer. The question, the answer, that follow-up and any notes are stored as timestamped memories. Each verified question is assigned to train or eval when it is written (1 in 3 eval, by id hash), so the eval set is frozen and never trained on. Synthetic data is used only in tests.

- **Labeling pool:** (changed 2026-10-17) you label from up to 15 shuffled candidates: the baseline's top 5, Sonara's top 5 and the 5 newest, shown without scores, or by typing a memory id. Each question row records which system proposed each candidate, so memories only Sonara finds can be credited. Marking a memory useful (including a Qwen answer) means "useful for that question", not "verified fact". Labels still favor what the two systems and recency already surface; a correct memory neither ranks in its top 5 is found only by typing its id. This is a known limit.
- **Volume:** a 5-point gap needs roughly 150–300 eval questions to tell apart from noise, i.e. about 10 verified questions a day for several weeks.
- **Fair baseline:** it gets the same inputs Sonara gets, including each memory's source (note, question, Qwen answer, follow-up). Its recency weight, half-life and per-source bias are tuned on the same train questions Sonara learns from, never on eval. Qwen-generated memories stay in recall; whether they help is for the feedback to show.

## What the earlier experiments taught us

1. A fair baseline gets exactly the information the model gets. The old rule script knew the test questions and looked 6× better than it was.
2. A small LLM with no training (65% on reworded text) beat every model trained here. Don't rebuild language understanding; use it.
3. Small networks trained on a few templates learn word positions, not meaning.
4. Whole-sentence vectors carry meaning well; single-word vectors are weaker.
5. Reading actual wrong answers found the real cause faster than any aggregate number.

The full experiment history is in git (branch `research/world-model-learned-lexicon`, commit `dbaca76`).
