# Miniature LLM — Historical Experiment Notes

This document supersedes the earlier starter-package walkthrough, which
referenced a `starter_general.pt` file and a live Q/A generation workflow
that are no longer the supported live Sonara session.

The repository still includes **real neural learning experiments**:

- Stage 1 trains causal next-token predictions.
- Stage 2 trains a GRU on synthetic situation-role labels.
- Stage 3 trains idea-conditioned next-token generation.
- Optional `--mode learn` applies gradients only to user-approved and
  evidence-supported Q/A targets, with retained checkpoints, replay, and
  holdouts. It has not demonstrated broad transfer or causal understanding.
- `--mode evaluate --weights-only` can test a trained language checkpoint
  without reading the user's local events or memories.

It does **not** follow that the live Sonara session knows how to reason
about experiences. The live route intentionally abstains from world QA while
that ability remains unproved. Older results or experiments reporting reduced
training loss, including earlier in-sample Q/A improvements, should be
read as measurements of *supervised fitting*, not general cognition.

For current setup, architecture boundaries, and test commands, use
[README.md](README.md). Controlled synthetic results and limitations remain
available in [EXPERIMENT_EXPERIENCE_REPORT.md](EXPERIMENT_EXPERIENCE_REPORT.md).
