# Sonara — Local Neural Learning Research (Honest Baseline)

**Status: Neural World Model Built & Integrated (Not Enabled by default).** This repository
contains real PyTorch language-model training experiments, a durable local evidence
log, an optional local Qwen/Ollama research assistant, and a dedicated trainable
Neural World Model (`sonara_world_model.py`).

**Activation Rule:** In live sessions, world-question predictions are **Integrated but not Enabled by default**;
the live session continues to honestly answer `UNKNOWN` unless explicitly started with
`--enable-world-model`. Pretrained or legacy mock checkpoints (`world_model.pt`) are never blindly
trusted.

## Live session: what it actually does

```text
User observation  -> append-only user-sourced event and episodic memory
                  -> passed to Neural World Model (when --enable-world-model is set)
                  -> pre-outcome prediction logged; verified on outcome arrival
                  -> optional pending Q/A labeling candidate (no gradients)

User /recall       -> lexical relevance + recency + importance retrieval
                     (quotes with provenance, not inferred knowledge)
User /timeline     -> chronological observed records
User /ask          -> UNKNOWN by default (honest abstention)
                     OR neural pointer-generator prediction (when --enable-world-model)
User /research     -> explicit localhost-only Ollama generation (unverified)
User /rate-research-> real gradient fit of tiny utility head for the given question
```

The live session does **not** load the synthetic language checkpoint, answer
questions from hardcoded temporal/location patterns, guess semantic frames, or
generate automatic "reflections" from repeated names. Existing saved memory files
and pretrained language checkpoints are not deleted or silently converted.

The retired commands `/train-world`, `/reflect`, and `/feedback` give an
explicit disabled/unavailable explanation. Prior obsolete `world_model.pt` mock
files are safely isolated; live world reasoning is governed exclusively by the
tested `SonaraWorldModel` in `sonara_world_model.py`.

## Try on the MacBook (no API key)

Install [Ollama](https://ollama.com/download), then once:

```bash
ollama pull qwen2.5:1.5b-instruct
```

In the repository:

```bash
python3 LLM_Training.py --mode session --enable-research --data-dir learning_data/local_test
```

Try:

```text
Jake moved the ball into the pond.
Jake moved the ball into the shed.
/recall ball
/ask Where was the ball before it was at the shed?
/research Why do objects float?
/rate-research 0.8
/quit
```

Expected: two observations are persisted and retrievable verbatim; the
historical question honestly answers `UNKNOWN`; Qwen can offer an explicitly
unverified local explanation, and the usefulness rating updates a separate
small neural head. No unattended internet research or paid service is enabled.
Ollama can operate offline after downloading the model.

## Real neural learning, distinct from live cognition

`LLM_Training.py` retains experimental training of a small Transformer for
next-token prediction, a GRU for **synthetic role-label classification**, and
an idea-conditioned text generator. This is *real gradient descent* but the
dataset expands 287 short examples into template-derived variants; accuracy
on related templates is **not** demonstrated real-world understanding.

```bash
# Non-live synthetic language experiment
python3 LLM_Training.py --mode train --no-save --steps 0
# Explicit review of evidence-supported Q/A candidates
python3 LLM_Training.py --mode review
# Optional supervised fine-tuning on explicitly approved Q/A
python3 LLM_Training.py --mode learn
# Weights-only language experiment: no access to personal memory
python3 LLM_Training.py --mode evaluate --weights-only --question "Who was Jake?"
```

`--mode learn` performs real gradients on approved labels, with replay,
holdouts, and a retention guard. It does not learn automatically when an
observation is entered, and improved training loss is **not proof** of
generalization or causality. The supplied `LLM_Training_checkpoint_v2.pt`
is an existing experimental language checkpoint, not a trained World Model.

`experiment_experience_learning.py` and `experiment_world_learning.py` are **synthetic
controlled benchmarks** using researcher-created scenarios. They evaluate parameter
updates, ablations, and failure modes across multiple seeds. They must **never** be presented
as Sonara having personally experienced those situations or having solved general human intelligence.

## Neural World Model Benchmark (Experiment 1)

`experiment_world_learning.py` rigorously evaluates whether Sonara's dedicated `SonaraWorldModel`
learns from temporal event sequences:
- **Held-Out Generalization (Unfamiliar Entities):** Trained neural accuracy is **52.0%** (vs **0.4%** untrained baseline, **3.6%** frozen weights, and **34.7%** memory-only retrieval).
- **Temporal Sequence Understanding:** Shuffling observation order causes accuracy to plummet from **52.0% to 29.4%** (a 22.6% drop), proving dependence on event chronology.
- **Cognitive Capabilities:** Demonstrates strong causal intervention (100%), event order (83.3%), and state transitions (66.7%), while revealing genuine failure modes in previous-state tracking (0%) and multi-object interference.
- **Restart Persistence:** 100% verified across repeated reloads.

See [EXPERIMENT_WORLD_REPORT.md](file:///Users/logancarton/Desktop/LLM_Training/EXPERIMENT_WORLD_REPORT.md) for full data and analysis.

## Test and audit commands

```bash
# Compilation check
python3 -m py_compile LLM_Training.py sonara_memory.py sonara_research.py sonara_world_model.py test_sonara_world_model.py test_live_boundaries.py experiment_experience_learning.py experiment_world_learning.py

# Existing regression suite
python3 LLM_Training.py --mode self-test
python3 test_sonara_memory.py
python3 -m unittest -q test_live_boundaries

# Neural World Model unit tests
python3 -m unittest -v test_sonara_world_model

# Controlled benchmarks
python3 experiment_experience_learning.py --seeds 3 --steps 60 --output artifacts/experience_learning_report.json
python3 experiment_world_learning.py --seeds 3 --steps 40 --output artifacts/world_learning_report.json

# Scoped Git whitespace check
git diff --check origin/main...HEAD
```

The CI gate also checks the actual session: raw observations remain available;
unsupported world questions abstain; there is no implicit training or trusted
promotion of local model outputs. CI uses a fake Ollama response; successful
tests do **not** establish physical laptop speed, cognitive generalization,
or safe autonomous external research.

## Files

- `LLM_Training.py`: lightweight evidence/research CLI, session entrypoint with safe `--enable-world-model` gate
- `sonara_world_model.py`: dedicated neural World Model owner (pointer-generator, BiGRU event encoder, recurrent state tracker)
- `test_sonara_world_model.py`: unit tests for gradients, weight deltas, pointer generalization, uncertainty, and persistence
- `experiment_world_learning.py`, `EXPERIMENT_WORLD_REPORT.md`: controlled multi-seed benchmark and comprehensive empirical research report
- `sonara_memory.py`: timestamped, append-only user evidence; lexical indexing/retrieval
- `sonara_research.py`: loopback-only local LLM helper plus trainable usefulness estimator
- `test_live_boundaries.py`, `test_sonara_memory.py`: targeted behavioral tests
- `experiment_experience_learning.py`, `EXPERIMENT_EXPERIENCE_REPORT.md`: baseline experience learning experiment
- `ideas.txt`, `details.txt`: synthetic-compatible training dataset
- `test_without_ideas.py`: language conditioning ablation, not causal reasoning

## Next research milestone

Address **previous-state tracking (overwritten history)** and **multi-object interference** in the
recurrent state tracker (e.g. via decoupled slot-based entity memory or explicit temporal state stacks)
before enabling live world reasoning by default.

Security: logs and checkpoints are local but not encrypted. Do not place
patient health information or other secrets in this prototype.
