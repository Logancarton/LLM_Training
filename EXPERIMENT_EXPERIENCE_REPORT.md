# Sonara Experience-Learning Experiment — Initial Research Report

**Repository:** Logancarton/LLM_Training
**Evaluation:** 2026-10-13 (America/Phoenix) / 2026-10-14 (UTC)
**Source:** GitHub Actions run [38499704311](https://github.com/Logancarton/LLM_Training/actions/runs/38499704311), artifact "experience-learning-results".

## Verdict

Sonara has real adjustable neural parameters and an operational approved-Q/A fine-tuning path. **This experiment did not establish that Sonara learns entity continuity, temporal state changes, or multi-event relationships.** The strongest observed improvement is consistent with learning question-format/answer-frequency patterns. The guarded learning gate limited large-scale regression but did not make the model reliably use the supplied experiences.

No production cognitive owner was modified. The synthetic experiment and CI gate are **Built and verified**, not evidence of a newly Integrated cognitive capability.

## Baseline architecture

- The `IdeaReader` GRU encodes one input into a compact vector and predicts semantic labels.
- The conditioned transformer predicts detailed output tokens; the existing personal path updates both reader and writer through supervised Q/A and uses public-example replay with a retention acceptance criterion.
- `SonaraMemorySystem.recall` is deterministic lexical relevance/recency/importance ranking; `reflect` groups recurring terms.
- The live session appends a recalled snippet to neural output; it does not feed retrieved multi-event context into neural reasoning.
- Session memory timestamps represent ingestion time; independent event time is not a first-class field in the production memory entry.

## Dataset, method, and isolation

Three seeded splits of 72 synthetic five-observation scenarios: **54 training scenarios, 18 held-out scenarios, 180 held-out questions per seed** (540 held-out questions in total). Entire actor–object combinations were withheld. The vocabulary for current-observation evaluation had no missing held-out tokens; unique held-out scene IDs were not learned. Tests cover entity attribution, temporal questions, correction, multi-event state, and warranted "unknown" responses.

The pre-existing repository checkpoint was loaded for each seed; experiments used in-memory weight updates. The experiment compared the original checkpoint, unguarded fine-tuning of the actual reader/writer, deterministic memory evidence retrieval, retrieval-fed neural prompts, and `learn_approved` with synthetic approved Q/A labels and its existing replay/retention guard. The label provider is a **synthetic oracle**, not Sonara or a real user's human approval. Evaluation facts were not trained, but question templates recur across families; this is a limited compositional benchmark.

Scores below are **single-token first-answer accuracy**, not conversational-answer correctness.

## Observed results

| Measurement | Seed 17 | Seed 18 | Seed 19 | Mean |
|---|---:|---:|---:|---:|
| Untrained checkpoint, held-out | 0% | 0% | 0% | 0% |
| Unguarded fine-tuned, held-out | 15.0% | 10.0% | 27.8% | 17.6% |
| Unguarded, novel observations supplied | 15.0% | 10.0% | 20.0% | 15.0% |
| Unguarded, shuffled observations | 15.0% | 10.0% | 17.8% | 14.3% |
| Existing guarded `learn_approved`, held-out | 35.0% | 30.0% | 30.0% | 31.7% |
| Guarded learner, novel observations supplied | 25.0% | 10.0% | 8.3% | 14.4% |
| Memory retrieval **evidence coverage** | 85.2% | 86.1% | 84.3% | 85.2% |
| Constant majority-class answer baseline | 20.0% | 20.0% | 20.0% | 20.0% |

**Do not compare retrieval evidence coverage directly to neural answer accuracy.** It only measures whether an expected direct-answer token appears in same-scene retrieved snippets. Retrieval also included at least one **wrong-scene** passage in 90/180 questions on each seed.

The unguarded experiment's earlier original-task loss increased from 0.145 to 3.303 / 2.979 / 2.874 (sampled validation). That path deliberately bypassed replay and is **not** a production forgetting result. With the production `learn_approved` replay and retention safeguard, the update was **accepted on all three seeds**; sampled original-task loss became 0.307 / 0.401 / 0.315, and full internal retention-gate loss rose from 0.135 to 0.315 / 0.378 / 0.320.

The guarded learner's correct-question training loss was lower than swapped-question loss (about 1.38–1.45 vs 2.09–2.26), indicating **some question conditioning**. Yet held-out **entity-category accuracy was 0% in all three seeds**. Its held-out "unknown" answers largely explain better weights-only scores; this does not demonstrate remembering unseen episodes.

## Failure mechanisms / confounds

1. **Output mismatch:** The original checkpoint is a sentence generator; 0% on newly formatted one-token answers is not a general intelligence baseline.
2. **Frequency learning:** Unguarded fine-tuning often predicts one word for nearly every question. Its improvement barely responds to shuffled observations, and average held-out accuracy is below the constant 20% "unknown" baseline.
3. **Lack of multi-event objective:** Approved Q/A token loss can improve without requiring ordered evidence integration or tracking entities through changing states.
4. **Context does not transfer reliably:** The guarded model's performance on novel questions with episode observations provided (14.4% mean) is lower than its question-only result (31.7%). The current pipeline is not proven to interpret event history.
5. **Retrieved evidence is not synthesis:** 85.2% direct-token coverage is a ceiling on evidence availability, not an answer generated by Sonara, and wrong-scene retrieved passages cause contamination.
6. **Evaluation limits:** The same small language templates are reused, and three split seeds on one scenario generator do not constitute independent real-world task families. The benchmark uses synthetic approvals and does not establish real-session continual learning from natural experience, restart stability, autonomous inference, or calibrated abstention.

## Implementation / verification

- New: `experiment_experience_learning.py`
- New: `.github/workflows/experience-learning.yml`
- Modified: `.gitignore` (future private data and checkpoint exclusions) and `README.md` (reproduction and interpretation).
- GitHub Actions run [38499704311](https://github.com/Logancarton/LLM_Training/actions/runs/38499704311): **passed**. Compilation, memory/neural self-tests, synthetic CLI session, scoped whitespace diff, three-seed benchmark, and artifact publication passed.
- Additional legacy `test_deliberation.py` and `test_without_ideas.py` smoke scripts both passed in earlier run 38499300826. That run's diff step failed only because shallow checkout lacked `origin/main`; checkout was corrected and its replacement gate passed.
- The full expensive default training/experiment suite has not been exhaustively rerun; no R4 gate applies to this repository.
- No new production neural behavior was implemented or claimed.

**Security:** The public repository already tracked a personal `sonara_memories/memories.jsonl` record. The new `.gitignore` prevents new untracked memory and checkpoints from accidental inclusion but does not untrack past content or erase Git history. Review the exposure and repository visibility separately; no history rewrite was performed.

## Research conclusion and next experiment

**Neural weights change and supervised answers improve, but successful learning of useful *temporal and entity relations* from event sequences has NOT been established.** Both the learned answer distribution and the inability to use current observations explain the gap. A focused next experiment should train and test **state transitions across independently timestamped events** using the correct existing interpretation and supervised-learning owners, with explicit correct-vs-shuffled sequence controls and old-task replay. Do not add a second planner, memory store, or runtime, and do not claim success unless live end-to-end temporal/entity questions improve beyond balanced baselines without damaging retained knowledge.
