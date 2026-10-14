# Miniature LLM — Personal Learning Experiment

## Status

This is an **experimental, local PyTorch system**. It extends the original `LLM_Training.py` in the same Python file, retains its three training stages and optional Stage 4 comparison, and supports the supplied `ideas.txt` and `details.txt` format. The new learning cycle uses **real gradient updates** to the existing idea reader and conditioned Transformer writer. It is **not a capable general chatbot**, nor does it automatically understand experiences.

The included `starter_general.pt` is a **lightly trained diagnostic starter**, using 400 synthetic/template-derived examples, width 48, one attention layer and 40 minibatch updates per stage. It is not a capable language model and has **not learned anything about you**. Session mode opens with it to let you test personal learning without first waiting for general training.

## Quick start

1. Extract the ZIP into a folder. Install Python 3.10+ and PyTorch: `python -m pip install torch` (on some Macs use `python3`).
2. Run built-in checks: `python LLM_Training.py --mode self-test`.
3. Start a session: `python LLM_Training.py --mode session --checkpoint starter_general.pt`.
4. Type something like `My old dog Jake liked tennis balls.`. This is **saved as an observation**, not presumed to be understood. Type `Who was Jake?` to sample a marked **unverified** neural prediction. Type `/quit` to finish.
5. Review proposed learning examples: `python LLM_Training.py --mode review`. Approve (`a`), edit (`e`), reject (`r`), defer (`d`), or mark a manually verified example as a holdout (`t`). Initially, **you must write/verify the question and answer**. Editing saves a *pending* revision, not authorization. No unapproved item is trained. Batch approval (`b`) works only on already labeled candidates.
6. Update neural weights: `python LLM_Training.py --mode learn --checkpoint starter_general.pt`. The existing reader and writer are fine-tuned, with replay of older general-language examples and a retention check. Approved holdouts are not trained. A general-only checkpoint is saved as `starter_general.pt.general.pt` before first accepted personal learning.
7. Test **without access to your saved conversation or association files**: `python LLM_Training.py --mode evaluate --weights-only --checkpoint starter_general.pt --question "Who was Jake?" --expected "my old dog."`. You can change the question without providing an expected answer; the optional expected answer is used solely for loss scoring.

Newer checkpoints remember their source dataset and architecture parameters, so you do not have to repeat `--examples`, `--width`, or `--layers` when reopening them. To test an older checkpoint without those metadata, supply the original training flags manually.

## Other useful commands

`python LLM_Training.py --mode timeline` — inspect dated, source-labeled observations and elapsed times. Event order is logged, but causal relations are **not** inferred automatically.

`python LLM_Training.py --mode thoughts` — show a review queue of unresolved candidates (rule-based scaffolding, **not** autonomous initiative).

`python LLM_Training.py --mode review --noninteractive` — display candidates without changing their approval status.

`python LLM_Training.py --mode revise --candidate-id <ID>` — revise or reject previously approved material. Then run `--mode learn --checkpoint starter_general.pt`. If an old training item was corrected or revoked, the program rebuilds the personal model from its preserved general-only checkpoint and the currently approved training set. This is a retraining procedure, **not mathematical proof of complete unlearning**.

`python LLM_Training.py --mode train --checkpoint my_general.pt --examples 400 --layers 1 --width 48 --epochs 1 --idea-epochs 1 --steps 40` — start general pretraining from new weights rather than the packaged starter. Larger settings take more CPU time. Using `--resume` continues previous general training; the original demo modes (`--adapt`, `--assess`, etc.) remain available.

`python LLM_Training.py --mode train --ideas-file ideas.txt --details-file details.txt --no-save --steps 0` — check compatibility with the supplied original dataset files. The original 52 pairs are also embedded, so these two data files are **optional** for default training.

## What actually learns?

```
User observation (timestamp, speaker, event index)
  -> append-only event record [NO GRADIENTS]
  -> candidate Q/A + evidence [PENDING]
  -> user approval [TRAIN OR HOLDOUT]
  -> tokenizer + expandable word vocabulary
  -> existing GRU idea reader -> compact idea tensor
  -> existing idea-conditioned Transformer -> next-token logits
  -> cross-entropy(predicted answer, approved answer)
  -> AdamW gradient update of GRU + Transformer parameters
  + replay of original general examples
  -> validation / retention guardrail
  -> durable updated checkpoint
  -> later weights-only query (no history/retrieval)
```

Unfamiliar words are appended to the model vocabulary, while existing token IDs and weights are preserved. The program can record all user observations, but **only approved Q/A labels** enter neural training. Model-generated replies are explicitly tagged unverified and never promoted to evidence automatically. When multiple different approved questions exist, the trainer reports whether swapping the question worsens the answer loss. A question-independent decoder can memorize answers without learning conditional associations.

## Research limitations — important

- **Association vs memorization:** A low loss or an exact answer on a trained Q/A is not proof of semantic abstraction. Use new paraphrases and independent experiences as holdouts. The miniature network may fail them.
- **Interpretation:** The baseline GRU has toy WHO/ACTION/WHAT/WHERE/WHEN/CONDITION classification heads. No reliable user-experience WHO/WHAT/WHERE/WHEN/WHY/HOW interpreter has been trained. Proposed personal Q/A targets are **human-authored review scaffolds**. Other fields remain explicitly unknown unless independently supplied.
- **Time:** Session timestamps, within-session order, and elapsed time are recorded. Learned temporal inference, causal reasoning, and state-change understanding are **not proved or automatically trained** by this version.
- **Continual learning:** Approved Q/A fine-tuning, replay, checkpoint resume, and a retention guardrail are implemented. They can fail to prevent forgetting. Previously learned personal facts are replayed only via the **approved training set**, not retrieved while answering.
- **Self-assessment:** Predictions, user feedback, retained-loss scores, and swapped-question scores are logged. Independent uncertainty calibration, autonomous learning priorities, and substantive self-modeling remain future research.
- **Initiative:** The thought queue is visible only when requested. There is no background process or unsolicited notification.
- **Security:** Logs and checkpoints are local but **not encrypted**. Learned personal facts may be encoded in weights. Do not use real patient information or private clinical data with this research prototype. Restrict filesystem access and back up sensitive work separately.

## Validation completed

The updated single-file regression suite passed source attribution, event order, approved/train/test/rejected separation, dynamic vocabulary preservation, gradient updates, declining approved loss, checkpoint restoration, correction detection, and inference while personal-data reads were deliberately forbidden. `ideas.txt` and `details.txt` compatibility was checked. In a separate end-to-end experiment with two approved training questions and one holdout, the approved-target teacher-forced token loss fell from **7.760 to 0.206**; swapped-question loss was **1.380**, suggesting actual question conditioning in that tiny example. Original general validation loss fell **6.868 to 5.353** with replay. This is a **small in-sample demonstration**, not a generalization benchmark. Revocation and rebuild from the preserved general checkpoint also worked.

The starter checkpoint's general validation remained weak: Stage 2 combined label accuracy **1.5%** on a small synthetic validation sample, and the Stage 3 correct/shuffled idea losses were essentially identical. Do not treat the starter as a meaningful interpreter. Train/evaluate stronger baselines before investigating abstraction or independent agency.
