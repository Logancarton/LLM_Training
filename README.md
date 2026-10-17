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

## Optional feedback-calibrated adaptive loop (uploaded `sonara_loop` transplant)

The uploaded standalone loop was evaluated as *donor mechanism evidence*, not
copied into a second runtime. Its pre/post-turn feedback concept, cautious
action gating, and retrieval fallback are now integrated into the existing
`LLM_Training.py --mode session` path. The **existing** world model remains
the only neural prediction owner, `SonaraMemorySystem` remains the only episodic
memory owner, and explicit user feedback stays in `events.jsonl`.

```bash
python3 LLM_Training.py --mode session --enable-adaptive-loop --data-dir learning_data/local_test
# Add --enable-world-model --world-checkpoint path/to/world_model.pt
# only when you have deliberately trained and validated that checkpoint.
```

With the adaptive flag, asking a question first considers bounded matching
user observations (shown **as verbatim evidence, not as an inferred answer**)
and an explicitly enabled neural prediction. A neural answer still needs to
pass the confidence/uncertainty gate. After a neural answer, enter
`/rate-answer 0` (bad) through `/rate-answer 1` (good). That explicit
feedback calibrates caution for related question cues. The action decision,
supporting ratings and threshold are appended to the normal event log.
Caution survives restart, but **does not train or validate world-model weights**.
Implicit user replies, LLM outputs, and unverified reflections never become
learning labels. Without the flag, existing session behavior is unchanged.

**Scope:** Integrated and opt-in, not enabled by default. This is a bounded
adaptive action/feedback loop, **not yet** the ZIP's unattended heartbeat,
macOS launch agent, web panel, automatic sensor, emotion model, or separate
SQLite graph. Those are intentionally not introduced: parallel memory/runtime
authority would compromise the existing cognition boundary. It does not
establish improved factual prediction accuracy or general autonomous cognition.

Tests: `python3 -m unittest -v test_sonara_adaptive test_live_boundaries`.

## Experience-to-causal-learning experiment (opt-in; synthetic only)

Measured results and limitations: [EXPERIMENT_CAUSAL_LOOP_REPORT.md](EXPERIMENT_CAUSAL_LOOP_REPORT.md).

The `experience` mode tests a full **prediction → chosen action → verified
environmental consequence → one neural gradient update → later prediction
and decision** cycle. It uses the existing `SonaraWorldModel.learn_step`
and `SonaraMemorySystem.remember` owners; there is no separate learner,
memory graph, action runtime, or language-checkpoint authority.

```bash
python3 LLM_Training.py --mode experience --experience-seeds 3 \
  --experience-dim 24 --experience-heldout 3 \
  --data-dir learning_data --experience-output artifacts/causal_learning_report.json
python3 -m unittest -v test_causal_loop
```

In each randomly initialized, counterbalanced two-route micro-world, Sonara
receives a scene naming two controls and two possible destinations but **not
their effects**. The neural model predicts the effects of possible actions.
A fixed experimental score rule selects a control for a goal; only the
simulator can reveal the true destination after execution. Exactly **one**
verified outcome updates the existing world-model weights. The canonical
memory owner stores both the action and the observed outcome with separate
provenance. A holdout uses unfamiliar actor/object names and never places
outcome text into the model's prediction input.

The JSON report records goal-success changes, correct-destination
probabilities and rankings, model parameter changes, sample choices, frozen
no-gradient controls, deliberately **incorrect-label** controls, checkpoint
restart equivalence, and reloaded canonical memories. All conditions start
with identical weights. The test *reports* whether decisions improved, it
does not assert that one gradient step necessarily succeeds. Restart traces
and checkpoints are stored under `learning_data/causal_runs`; the report
under `artifacts`. Both directories are Git-ignored.

**Evidence boundary:** Passing integration tests demonstrates a real causal
feedback *signal path*. Improvements, if observed, are limited to a tiny
synthetic environment and novel entity names—not previously unseen physical
rules, self-supervised causal discovery, autonomous computer interaction, or
reliable single-trial generalization. An unchanged or worse held-out result
is a valid negative experimental finding. This experimental mode does **not**
enable world-model training in ordinary conversation.


## Action-conditioned learning after a second verified experience (research)

The original one-experience control remains unchanged. The `experience`
benchmark now **also** executes the alternative action, records its *verified*
outcome in the canonical memory, and trains a paired gradient objective in
the **existing** `SonaraWorldModel` owner. Training first rewards each
observed action's actual destination and then contrasts each destination's
probability under the two action queries. Nothing treats an untried action's
outcome as a verified fact.

Matched controls use exactly the same first-action checkpoint and second
observation: frozen/no-gradient, two supervised labels without contrast, and
deliberately swapped labels (kept outside canonical memory). The report
measures held-out goal success, destination ranking, signed correct-action
probability margins, separation between action predictions, and equivalence
after checkpoint/memory reload. Replay steps are *optimizer updates on two
observations*, not additional experiences.

Use:

```bash
python3 LLM_Training.py --mode experience --experience-seeds 3 --experience-dim 24 --experience-heldout 3
python3 -m unittest -v test_causal_loop
```

This is **Integrated only into the opt-in synthetic experiment**, not the
ordinary conversation learning loop. Contrastive gradient updates alone
do not establish successful decisions; read the generated JSON metrics before
claiming action-dependent transfer.
