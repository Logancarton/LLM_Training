# Sonara — Local Neural Learning Research (Honest Baseline)

**Status: no validated neural world-reasoning model is integrated.** This repository
contains real PyTorch language-model training experiments, a durable local evidence
log, and an optional local Qwen/Ollama research assistant. They are *not* the same
thing as an autonomous intelligence. Do not call retrieval, regexes, language
generation, synthetic oracle labels, or feedback fitting "world understanding."

## Live session: what it actually does

```text
User observation  -> append-only user-sourced event and episodic memory
                  -> optional pending Q/A *labeling* candidate (no gradients)

User /recall       -> lexical relevance + recency + importance retrieval
                     (quotes with provenance, not inferred knowledge)
User /timeline     -> chronological observed records
User /ask          -> UNKNOWN (world-question answering not validated)
User /research     -> explicit localhost-only Ollama generation (unverified)
User /rate-research-> real gradient fit of tiny *utility* head for the given
                     question (not LLM fine-tuning or proven transfer)
```

The live session does **not** load the synthetic language checkpoint, answer
questions from hardcoded temporal/location patterns, guess semantic frames, or
generate automatic "reflections" from repeated names. Previously written
heuristic-reflection records remain in the user's history but are not used for
evidence recall. Existing saved memory files and pretrained checkpoints are
not deleted or silently converted.

The retired commands `/train-world`, `/reflect`, and `/feedback` give an
explicit disabled/unavailable explanation. Prior `world_model.pt` checkpoints
(if any) are obsolete and **not loaded**. Only a separately trained and
validated successor could become a new neural world owner.

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

`experiment_experience_learning.py` is a **synthetic negative/control
benchmark** using researcher-created answers. It is useful for testing
parameters, ablations, and failure modes. It must **never** be presented as
Sonara having experienced those situations or learned their causes.

## Test and audit commands

```bash
python3 LLM_Training.py --mode self-test
python3 test_sonara_memory.py
python3 -m unittest -q test_live_boundaries
python3 experiment_experience_learning.py --seeds 3 --steps 60 --output artifacts/experience_learning_report.json
python3 -m py_compile LLM_Training.py sonara_memory.py sonara_research.py test_live_boundaries.py experiment_experience_learning.py
git diff --check
```

The CI gate also checks the actual session: raw observations remain available;
unsupported world questions abstain; there is no implicit training or trusted
promotion of local model outputs. CI uses a fake Ollama response; successful
tests do **not** establish physical laptop speed, cognitive generalization,
or safe autonomous external research.

## Files

- `LLM_Training.py`: real but experimental supervised neural training; lightweight evidence/research CLI
- `sonara_memory.py`: timestamped, append-only user evidence; lexical indexing/retrieval
- `sonara_research.py`: loopback-only local LLM helper plus trainable usefulness estimator
- `test_live_boundaries.py`, `test_sonara_memory.py`: targeted behavioral tests
- `experiment_experience_learning.py`, `EXPERIMENT_EXPERIENCE_REPORT.md`: clearly synthetic research and measured limitations
- `ideas.txt`, `details.txt`: synthetic-compatible training dataset
- `test_without_ideas.py`: language conditioning ablation, not causal reasoning

## Next research milestone

Build an **actual neural experience-transition predictor** in the correct
world owner, with pre-training predictions, supervised observation outcomes,
independent held-out situations, frozen/untrained/shuffled-order baselines,
restart persistence, and a measurable improvement over retrieval-only
controls. Do not add hardcoded answer rules to make tests pass.

Security: logs and checkpoints are local but not encrypted. Do not place
patient health information or other secrets in this prototype.
