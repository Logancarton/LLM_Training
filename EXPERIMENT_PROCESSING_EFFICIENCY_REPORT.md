# Sonara temporal world-model efficiency — controlled research

**Repository:** `Logancarton/LLM_Training`  
**Experimental base:** draft PR #9 (`research/temporal-action-integration`)  
**Implementation:** draft PR #10 (`research/efficient-temporal-world-model`)  
**Status:** Built and tested in the synthetic research path; **not merged or enabled by default**.

## Ownership and active path

- `SonaraWorldModel` remains the single neural world-model learning and prediction owner. `sonara_memory.py` remains the canonical episodic evidence owner. The interactive `LLM_Training.py` runtime is unchanged.
- The integrated temporal model conditions event encodings on the current question, uses bidirectional temporal state and hierarchical pointers, and trains on verified feedback. The efficiency branch does not add a second model, selector, or independent training store.
- A **16-entry bounded event-tokenization/candidate cache** and **4,096-entry bounded n-gram index cache** store only weight-independent preprocessing. Forward activations and gradients are never cached. Derived caches are cleared when the live session is explicitly cleared.
- During no-grad prediction, character pooling and pointer scoring use batched kernels. When building the training graph, the original paths are retained to avoid the initial batch-backward slowdown. This is an execution optimization, not a new cognitive capability.
- For sessions longer than 24 observations, the existing owner can select cue-matching events, adjacent events, and recent context; if the dependency footprint exceeds 24 events or evidence cannot be matched, it falls back to full history rather than silently cutting that chain. This heuristic still lacks an exhaustive proof of multi-hop causal coverage.
- Verified training replay stays bounded in storage and is saved/restored with checkpoints; checkpoint state shapes are validated before loading to reject incompatible architectures without partial mutation. Old optimizer moments are not restored; restarting an optimizer is not proof of exact continued-training trajectory.

## Measured baselines (same 3-seed synthetic evaluations)

| Condition | Held-out world Q/A | Previous-state | Two-action success |
|---|---:|---:|---:|
| Merged main, commit a8b5e46 | 35.56% | 0% | **100%** |
| PR #9 | **93.34%** | **100%** | 77.78% |
| Efficiency branch before final cache-bound cleanup | **93.34%** | **100%** | 77.78% |

The accuracy gains in PR #9 are genuine measurements on generated tasks with recurring wording and structures; they do not establish general-world causal understanding. The action-success regression against `main` is real and **blocks merging either draft PR**.

## Same-weight inference ablations (3 seeds; CPU, 8 repeats)

| Processing mode | Mean CPU ms / prediction | Traced peak Python KiB | Max probability difference |
|---|---:|---:|---:|
| PR #9-equivalent unoptimized | 26.3732 | 70.30 | 0 |
| Event preprocessing cache only | 25.3446 | 50.69 | 0 |
| Character batch only | 25.0845 | 59.56 | 2e-8 |
| Token scoring batch only | 25.0795 | 57.52 | 0 |
| All optimizations | **23.6502** | **47.72** | 1e-8 |

Combined speedup in this microbenchmark is **1.115×** (~10.3% less inference CPU per prediction) and ~32% lower **traced Python allocation peak**. Tracemalloc does not measure all PyTorch/native memory. Repeated short synthetic questions are not a representative whole-session workload.

The initial all-mode version slowed the 40-update world training benchmark on a different CI run. Batching is therefore now limited to no-grad inference. Cross-run training wall-times are not controlled enough to claim a confirmed training speedup.

## Verified-replay budget ablation

Identical models, seeds, 40 actual gradient updates per condition, same train/held-out examples, changing only how many **previously verified** category exemplars are included per update. The seeded initialization in this standalone replay experiment differs from the broader world benchmark, so the reference mean is 94.22% rather than 93.34%.

| Replay per update | Mean train CPU sec | Held-out Q/A | Previous-state |
|---|---:|---:|---:|
| Full verified replay (~6.96) | 18.65 | **94.22%** | 100% |
| 1 example | 5.01 | 74.22% | 83.33% |
| 2 examples | 7.07 | 81.34% | 100% |
| 4 examples | 11.05 | 88.44% | 100% |

**Decision:** retain full verified replay. The reduced-compute alternatives lose unacceptable world-question accuracy. Times in this ablation include Python allocation tracing, so use them only for within-run comparison, not against the world benchmark wall-times.

## Checks and remaining risks

The CI branch includes the full `unittest` suite, focused gradient/parity/sparse-activation/eviction/restart tests, the real CLI smoke test, compilation, scoped whitespace check, world benchmark, causal action gate, and both efficiency/replay ablations. The initial efficiency commit passed all checks ([CI run](https://github.com/Logancarton/LLM_Training/actions/runs/38998961089)); cache-bound cleanup must be retested on its own commit.

**Blocking risk:** action discrimination remains 77.78%, below main's 100%; the smarter temporal representation may weaken distinctions between interventions. The next cognitive experiment should trace action-token representations and gradients and fix this in the existing world-model owner. No merge to main until main-level action performance and temporal gains both pass, followed by checkpoint compatibility verification on saved local artifacts.

**Also unproved:** generalization to natural language beyond generated templates, robust sparse event-chain recall across pronouns/corrections, meaningful continual learning from live unscripted outcomes, total process/native peak memory, and exact optimizer-history persistence after restart.
