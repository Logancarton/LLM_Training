# Sonara: associative temporal memory experiment (v0)

This is an **integrated, deliberately narrow** research prototype, not a general language model, self-coding agent, or biological neural simulation.

**Input:** a paragraph about movements/typical locations. **Output:** evidence-backed, persistent events with stable scoped identity, time labels, novelty flags, and learned weighted links. A limited question grammar retrieves relevant experiences through indexed, bounded activation. Raw unsupported sentences are kept as notes instead of being misinterpreted.

## Try it (Python 3.10+, standard library only)

```bash
python train.py --model model.json --examples 500
python sonara.py --db sonara.db --model model.json store "On Monday morning, Sarah put her keys on the kitchen table. Later that afternoon, she moved them into her bedroom drawer. Usually, Sarah keeps her keys by the front door."
python sonara.py --db sonara.db --model model.json ask "Where are Sarah's keys now?"
python sonara.py --db sonara.db --model model.json ask "Where were Sarah's keys before?"
python sonara.py --db sonara.db --model model.json ask "What is unusual about Sarah's keys?"
python sonara.py --db sonara.db --model model.json ask "Why did Sarah's keys?"
python -m unittest discover -s tests -v
```

`model.json` is the reproducible trained association scorer; `sonara.db` is a runtime file and should not be committed. The thin CLI is `sonara.py`. Cognitive ownership lives in `sonara_memory.py`: conservative event encoding, fitted link estimation, durable temporal state, novelty, sparse recall. `train.py` owns synthetic curriculum, independent held-out names/objects/places, and evaluation.

## What gets learned vs. what is programmed

- **Learned:** shared-identity association strength from 2,500 labeled training pairs. Logistic weights change during supervised updates and persist in `model.json`. In held-out synthetic examples, unseen actor/object/place names are used. Edge strength is an uncalibrated model score, **not** a validated percentage chance that a memory is true.
- **Programmed bootstrap:** language parsing for specified sentence forms; identity key scoped to actor and item; sequence insertion order; novelty relative to an explicit habitual location; query interpretation; hard identity isolation; sparse expansion limit. These are transparent scaffolds and not claims of learned human-like cognition.
- **Retrieval:** a small indexed set of events seeds the answer, and at most `limit=3` events activate through eligible learned graph edges. Retrieval is not exhaustive. Causes are not inferred from coincidental events.

## Measured result on generated examples

Run `python train.py` for current values. The original seeded run with 500 stories (2,500 labeled pairs) achieved 100% relation-label accuracy on 500 held-out pairs; 400/400 memory questions correct on 100 held-out synthetic stories, with 1.88 events activated per question from a store of 300. Before training the default relation scorer yielded 60% accuracy (always negative at threshold 0.7). **These numbers measure a simple synthetic task, not open-domain generalization.** The generator varies actors, objects, places, verbs, times, and phrasing, but remains within the same parsing grammar and the link label is determined by shared identity. They cannot establish real-world memory competence or autonomous architecture discovery.

## Known boundaries

- Does not understand arbitrary paragraphs, ambiguous reference, nuanced pronouns, factual uncertainty, or causal claims. Unparsed text remains evidence-only notes.
- "Earlier" means narrative/insertion order, not a mathematically normalized timestamp. Chronologically out-of-order paragraphs are not yet reconciled.
- Identity is scoped to `actor::item` and cannot yet handle object transfers between owners or two same-named objects belonging to one person.
- A usual location is taken from the latest explicit `Usually` event. Contradicting reports need a richer evidence/certainty model.
- No real LLM advisor, autonomous code generation, training of the text encoder, verified calibration, or continual retraining loop is wired. Do not confuse a fitted link scorer with full Sonara cognition.
- Not ready for clinical information or private records.

## Next research gate

Replace the hand-coded paragraph encoder with a **supervised, trainable** paragraph-to-episode model using labeled training pairs, while retaining conservative abstention and evidence provenance. Validate held-out compositions, adversarial ambiguity, shuffled event order, state conflicts, and retrieval accuracy against a fixed indexed baseline before promoting any LLM-assisted architectural rewrite.
