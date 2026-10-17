# Learned sparse cognitive memory — controlled experiment

**Stage:** Built, integrated opt-in and tested through GitHub Actions; cognitive improvement **NOT demonstrated**. Experimental and disabled by default.

## Cognitive ownership and live signal path

Original user observation → canonical SonaraMemorySystem → transient SonaraWorldModel session view → lightweight learned ranker → chronological selected original events → the existing temporal world reasoner → prediction → verified outcome → the existing world-model learning owner → selector updates from event ablation.

The ranker is a head within NeuralWorldModel, not a parallel model or a second memory authority. Neural event representations use the existing trainable character/subword embeddings. Their rank scores are trainable through the existing optimizer, but selector training detaches the world-backbone features to avoid corrupting full-context world knowledge. Ranker training labels come from how removing an original event changes loss against an explicitly verified answer. The target is NOT the teacher's own prediction. The loss-ablation teacher is bounded to 12 events per verified training example; long cases skip attention learning rather than silently claim it happened.

## Recovery and limits

The ranker activates configurable top-k original observations and restores their order before recurrent state tracking. If confidence is low or answer is "unknown", it expands once to at most 32 selected events and abstains if still insufficient. High-confidence incorrect retrieval cannot currently be detected reliably; no recovery guarantee exists for arbitrary hidden causal dependencies. Selection never prunes stored evidence, and evaluation reports activated counts separately from end-to-end runtime.

## Benchmarks and measurement protocol

The new experiment runs seed-matched full context, recency, random, fixed lexical overlap, learned ranker, and learned ranker + bounded recovery on distinct held-out actors, objects, corrected reports, paraphrases and varying distractor counts. Verified answer accuracy is separate from agreement with full-context predictions. CPU process time and event count include selector overhead; process peak RSS is not a valid per-mode tensor-memory measurement.

The existing 3-seed historical comparison from PR #9/#10 is unchanged:

| Model | Held-out Q/A | Previous-state | Two-action decision success |
|---|---:|---:|---:|
| main | 35.56% | 0% | **100%** |
| PR #9 / PR #10 temporal | 93.34% | 100% | **77.78%** |

Those are results of previous synthetic tests, not the new sparse-memory experiment. A smaller activated event count by itself is not proof of lower CPU use, and action-selection regression blocks any merge to main.

## Measured results (October 17, 2026)

**Isolated research gate:** [run 39042774403](https://github.com/Logancarton/LLM_Training/actions/runs/39042774403) — SUCCESS (compile, focused tests, held-out experiment, artifact). **Original cumulative gate:** [run 39042422334](https://github.com/Logancarton/LLM_Training/actions/runs/39042422334) — SUCCESS on implementation commit da910a9 (compile, focused/integration, live CLI, two-action causal harness, full unit suite, world benchmark, processing/replay and new sparse benchmark). Passing a software gate does not mean cognitive outcomes improved.

3 seeds × 5 materially different synthetic held-out cases; retrieval budget 4:

| Mode | Correct verified answer | Full-model agreement | Mean activated events | CPU ms per prediction | Unresolved recoveries |
|---|---:|---:|---:|---:|---:|
| Full-context | **0.0%** | 100% | 19.2 | 23.578 | — |
| Recency | 0.0% | 33.3% | 4 | 6.795 | — |
| Random | 0.0% | 53.3% | 4 | 6.433 | — |
| Fixed lexical overlap | **40.0%** | 20.0% | 4 | 6.555 | — |
| Learned sparse | 33.3% | 26.7% | 4 | 8.575 | — |
| Learned + bounded recovery | 0.0% | 0.0% | 8 | 19.730 | **15/15** |

Learned sparse processing was approximately 2.75× faster than full-context reasoning (**63.6% less measured CPU** in this tiny workload), but **31% slower than fixed lexical retrieval** and less accurate. These CPU timings are within one CI runner and are not a demonstrated end-to-end deployment speedup. The selector must still inspect all lightweight event representations; only expensive recurrent processing is bounded. Native tensor peak-memory savings were not established.

**Why the experiment failed:** the full-context comparator saw only *three* supervised episodes per seed, whereas the independently reported PR #9 temporal model was trained on a much larger 40-step benchmark. A 0% full-context result therefore means this comparison lacks a competent teacher. It does **not** invalidate prior 93.34% results, but it cannot be used to prove learned attention preserves those results. Novel syntax and long distractor sequences are additional confounders. Independently, the uncertainty/recovery gate recovered more events in every case, but then abstained in every case because confidence remained below its conservative threshold. This is safe non-fabrication, **not successful evidence recovery**.

**Experimental verdict:** The model can train attention parameters on verified event-ablation signals, rank from original memory and restrict expensive activation; the ranker did **not** establish superior attention learning, grounded temporal reasoning, good decisions, or effective recovery. Do not merge or enable by default.

**Incomplete scientific gates:** An explicit hidden multi-hop causal dependency test, matched-weight attention-before/after ablation, full-context teacher with validated competence on this distribution, sparse action-choice success against the main baseline, calibrated recovery accept/reject, long-history 10k-event scaling, and per-mode native tensor memory are **not yet proved**. Current behavioral tests validate wiring, abstention, checkpoint and memory integrity rather than those broader claims.

## Completion criteria

- **Built:** trainable vector ranker, verified loss-influence training, bounded recovery, original chronology preservation, checkpoint compatibility.
- **Integrated:** opt-in live CLI, canonical-memory hydration and focus/recovery tests passed.
- **Enabled:** no (research flag only).
- **Restart verified:** passed on the focused synthetic evidence and checkpoint test; older compatible world weights preserved.
- **Full gate passed:** implementation commit passed cumulative GitHub Actions, but **cognitive non-regression/quality gate remains blocked** by temporal branch action-choice regression and the unsuccessful new sparse experiments.

**Merge policy:** keep the branch in draft status. No synthetic accuracy claims should be generalized to autonomous planning, lasting identity, or human-level causal understanding.
