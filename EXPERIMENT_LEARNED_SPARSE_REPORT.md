# Learned sparse cognitive memory — controlled experiment

**Stage:** Built + connected to the existing optional live session, unverified on CI. Experimental and disabled by default.

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

## New measurement results

No new sparse benchmark numbers have yet been verified. The CI job exports artifacts/learned_sparse_report.json; replace this statement only after inspecting the actual artifact, focused tests, integration, and full regression gate.

## Completion criteria

- **Built:** trainable vector ranker, verified loss-influence training, bounded recovery, original chronology preservation, checkpoint compatibility.
- **Integrated:** optional Sonara CLI flag and canonical-memory rehydration are wired; live verification pending.
- **Enabled:** no (research flag only).
- **Restart verified:** pending CI.
- **Full gate passed:** pending CI and action-decision non-regression.

**Merge policy:** keep the branch in draft status. No synthetic accuracy claims should be generalized to autonomous planning, lasting identity, or human-level causal understanding.
