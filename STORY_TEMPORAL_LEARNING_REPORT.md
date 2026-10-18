# Sonara story: incremental timestamp learning (exploratory)

Reference story commit: https://github.com/Logancarton/LLM_Training/commit/67a5e7c52b74bdeb39e2d1d4d63525e9d3f6d515

## Question
With 100 sequential labeled experiences, do learned timestamp representations improve held-out prediction versus the same model with erased timestamps? Can temporal prediction be learned when event meaning is available?

## Run
python3 sonara_story_learning.py --seeds 5 --steps 12 --json learning_data/story_curve.json

## Controls
- 100 chronological training episodes and 60 held-out questions (20 familiar, 20 unfamiliar, 20 time-sensitive), per seed.
- After each new episode, train with that episode and three randomly selected previously observed episodes (12 gradient steps). Both arms use identical initial weights, model size, and examples.
- Score after 0, 10, 30, 60, and 100 experiences. Evaluation labels and observations are never used for updates.
- Inputs contain only this episode's events strictly before the question time. The training answer memory at the query time is excluded.
- Timed arm uses a learned time-vector encoder with elapsed-time and clock coordinates. Control erases these inputs without changing the model size. No half-life or hard-coded relevance.
- Paired time test asks two opposite-outcome questions with IDENTICAL events but different query times, tracking both-correct accuracy.
- A SEPARATE oracle diagnostic supplies the correct semantic event category from synthetic generator metadata. Important-loss-soon and important-loss-days-later have precisely the same semantic category, so only time distinguishes them. This diagnostic does NOT measure Sonara independently understanding sentences.

## Local exploratory results

Validated with a reproduced generator for commit 67a5e7c and a compatible minimal memory-store surrogate. Values are averages across five seeds and 60 test examples per seed. Re-run in the repository for canonical results; CI runs leakage and smoke checks on actual repository classes.

| Episodes | Raw words + time | Raw words no time | Oracle meaning + time | Oracle meaning no time |
|---:|---:|---:|---:|---:|
| 0 | 48.3% | 52.3% | 42.3% | 48.7% |
| 10 | 52.3% | 51.3% | 60.3% | 56.7% |
| 30 | 53.0% | 52.3% | 95.7% | 70.0% |
| 60 | 53.7% | 49.7% | 100.0% | 70.0% |
| 100 | 50.3% | 53.7% | 100.0% | 70.0% |

At 100 episodes, oracle+time answered both sides of all near/far pairs correctly (100% pair success), whereas oracle without time achieved 0% paired success.

## Interpretation and limitations

Time is learnable in this synthetic system WHEN correct event semantics are already provided. The raw-word neural model remained near chance after 100 exposures and did NOT consistently benefit from time. Thus the end-to-end learning hypothesis remains unproven; event understanding is a plausible limiting factor.

This benchmark does not test causal intervention, real-world reasoning, pretrained Qwen embeddings, or Sonara's deployed retrieval stack. The oracle receives privileged synthetic event-class metadata; those values must NEVER be confused with autonomously learned meaning. Scenario generation itself encodes deterministic outcomes. Five inspected seeds are exploratory, not independent confirmation. Vocabulary IDs are allocated from training text before sequential gradient updates, but never from evaluation text or labels.

## Next step
Use frozen Qwen sentence embeddings (existing project infrastructure) to improve event meaning without supplying an oracle; repeat time/no-time with new evaluation seeds and do not tune to the present synthetic test split. Only after temporal prediction works on novel language should causal generalization be tested using intervention or counterfactual controls.
