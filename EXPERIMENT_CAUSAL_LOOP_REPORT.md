# Sonara — Single-Experience Causal Transfer Experiment

**Status (2026-10-16):** Experiment built, connected through `LLM_Training.py --mode experience`, and evaluated in GitHub Actions. One verified outcome changes neural weights and survives restart. **Held-out decision improvement was not demonstrated.**

## Question and mechanism

Does one verified interaction alter the existing world model's predictions and improve later decisions on unfamiliar entities, rather than merely increasing confidence?

Each seed specifies a synthetic, hidden mapping between **tap/hold** and **north/south**, counterbalanced across seeds. Sonara receives the visible scene with actions and possible destinations but no action effects. The existing neural world model scores interventions; a fixed experimental evaluation policy chooses one action to reach a goal. The environment reveals the actual result only after execution. The canonical `SonaraMemorySystem` records the action and simulator-verified outcome. **Exactly one** `SonaraWorldModel.learn_step` is applied using the pre-outcome context and observed answer.

The same starting model without gradients, a deliberately wrong-label training control, unfamiliar actor/object contexts, and a checkpoint/memory restart check provide comparisons. Evaluation does not feed withheld outcome observations into the predictions. The experiment does **not** add a production runtime, planner, selector, memory store, or neural model.

## CI result: 2 seeds, dimension 16, 2 held-out pairs

| Metric | Result |
| --- | --- |
| Verified training steps | 1 per seed |
| Model parameter change | Nonzero for each seed |
| Mean correct-outcome probability change | +0.074551 (absolute probability) |
| Mean held-out destination ranking accuracy change | 0.0000 |
| Mean goal success change | -0.2500 (absolute rate) |
| Seeds improving goal success | 0 / 2 |
| Seeds improving true-outcome probability | 2 / 2 |
| No-gradient predictions match before state | 2 / 2 |
| Reloaded model predictions match after state | 2 / 2 |
| Reopened episodic memory preserves action/outcome evidence | 2 / 2 |

**Per seed:** Seed 100: goal-success delta 0.0, true-outcome probability +0.074576. Seed 101: goal-success delta -0.5, true-outcome probability +0.074526. Incorrect-feedback controls also increased some true-outcome probabilities, so the probability gain **cannot be interpreted as causal rule learning**.

The model often raised probabilities for *both* destinations after feedback instead of learning action-specific separation. This is a plausible explanation for failing to select the appropriate intervention; it is not established as the unique causal diagnosis. Only two counterbalanced worlds were tested in CI, so the numbers are not statistical estimates of general performance.

## Verdict and next mechanism to investigate

**Infrastructure and signal path: Integrated in experimental mode. Actual causal generalization: not demonstrated.** A passing test certifies that the observation is correctly routed to the existing neural weights, not that the resulting state is an accurate world model.

Next investigate **action-conditioned discrimination** in the current `WorldReasoningCore` and `PointerGeneratorHead`: compare correct vs incorrect action-specific logits and gradients before/after feedback, isolate token prior shifts from actual intervention dependence, and measure whether the learned distinction survives novel entities and restart. Retain the no-gradient and false-label controls. Do not infer unobserved counterfactual outcomes as training facts or add a parallel planner.

## Reproduction

```bash
python3 LLM_Training.py --mode experience --experience-seeds 2 --experience-dim 16 --experience-heldout 2
python3 -m unittest -q test_causal_loop
```

CI preserves the report JSON as an artifact named `experience-learning-results`; private experimental observations/checkpoints remain Git-ignored under `learning_data/causal_runs`. This simulated environment does not imply continuous autonomous learning in the ordinary session.


## Follow-up: two verified actions and contrastive action learning

The experiment has been extended to preserve the original one-action
baseline and *also* interact with the alternative control. Both outcomes are
verified by the hidden simulator before entering canonical episodic memory.
The existing world-model owner receives both (action question, outcome)
pairs and applies a bounded supervised + pairwise ranking objective:

`L = mean NLL(observed action, observed destination) + mean softplus(margin + NLL(correct action) - NLL(other action))`

The ranking term asks whether, for **the same desired outcome**, the
observed successful action scores above the verified alternative. It is not
a generic embedding-similarity metric and does not fabricate the opposite
action's result after the first experience. A matched supervised-only
ablation uses the same pairs without the ranking term. False-label and
no-gradient controls remain independent of authoritative memory.

Read `action_pair_runs` and the `mean_pair_*` keys in the generated
report for observed performance. The existing CI gate verifies executed
feedback, weight updates, controls, and restart consistency. Passing the
gate **does not certify decision improvement**; improvement must be measured
against the matched supervision baseline across held-out entities and seeds.

**Status:** Experimental implementation, not live-session cognition;
action-conditioned transfer must be demonstrated empirically.
