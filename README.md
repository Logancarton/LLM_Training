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

## V1.1: fifty additional, learned association signals

This update adds **50 concrete pairwise signal channels** to the original seven (57 total) in `association_signals.py`. All 50 are exercised in training and in an independent signal-coverage test. They are organized into five groups of ten: identity/pattern separation, spatial context, temporal context, action/habit relationships, and source/uncertainty/novelty. Each channel measures a particular evidence cue or contrast: for example shared actor plus changing place, repeated time, same location across different people, conflicting action, overlapping source phrases, or explicit uncertainty terms. A channel can contribute positively or negatively to a learned association. **This is not fifty distinct brain regions, neural synapses, verified factual relations, or proof of biological fidelity.** Some channels are correlated and their individual utility has not been ablated.

Two supervised classifiers now learn from distinct labels:

- **Identity continuity:** an actor/object-scoped episode may be connected to its own history; the strict identity boundary remains enforced even when a score is high.
- **Contextual association:** episodes for different objects may be linked when they explicitly share an actor or place. Such links **never imply that two objects are the same, that two people were in the same place at the same time, or that one event caused another**.

Both are weighted using labeled synthetic evidence. The score is an **uncalibrated association score**, not the probability that a memory is factually true. Learned top-weighted evidence cues are saved with each link and available during sparse associative recall. Each incoming episode stores **at most three identity edges and two context edges**, rather than an unbounded dense network. Retrieval remains indexed and bounded; separate factual queries cannot traverse cross-identity context links as if they were direct evidence.

### Run the updated experiment

After `git pull --ff-only origin main`, retrain because the parameter schema has changed:

```bash
python3 train.py --model model.json --examples 500
python3 sonara.py --db sonara.db --model model.json store "At dawn, Sarah placed the keys on the table. At noon, Sarah moved the keys into the drawer."
python3 sonara.py --db sonara.db --model model.json store "At dusk, Sarah placed the map on the table."
python3 sonara.py --db sonara.db --model model.json related Sarah keys --limit 5
python3 sonara.py --db sonara.db --model model.json ask "Where are Sarah's keys now?"
python3 -m unittest discover -s tests -v
```

`related` returns a small number of actual stored episodes, relation types (`identity-continuity`, `shared-actor`, `shared-place`), association scores, and contributing cues. A missing association is not proof of no relationship. `model.json` is version 2 and must be retrained; existing `sonara.db` event histories are retained and the new link-evidence column is added in-place. Old stored link scores remain snapshots until re-ingestion; there is no claimed online link reoptimization.

The seeded 500-example experiment reports 100% association-label accuracy on simple held-out synthetic identities and context pairs, 400/400 bounded-grammar recall questions correct, 300 stored events, and ~1,188 edges (at most 5 per inserted event). This does **not** show that all 50 channels individually improve performance or that Sonara has learned an open-domain memory representation. Encoding remains conservative and rule-based; self-generated storage code, causal reasoning, online learning feedback, long-term score calibration, and free-text retrieval remain future research.

Biological analogies are deliberately narrow: hippocampal pattern separation and completion, temporal-context binding, and complementary fast episodic/slow structural learning motivate testable computational ideas, **not implementation-level equivalence**. Background: [Singh & Schapiro (2026)](https://pubmed.ncbi.nlm.nih.gov/42421581/), [McClelland et al. (1995)](https://pubmed.ncbi.nlm.nih.gov/7624455/), [Leal & Yassa (2018)](https://www.nature.com/articles/s41593-017-0065-1).

## V1.2: cue → context → episode (hippocampal-inspired)

A partial **detailed episode** is used as a retrieval cue. Instead of
consulting synthetic hierarchy tiers or comparing episode counts, the memory
owner follows **previously learned, persistent edges**. The first hop
reactivates a broader set of linked contexts (shared actor, shared recorded
location or same scoped identity). A second bounded hop may lead to another
episode that has no direct edge to the original cue. The last stage
**reinstates the original stored episode evidence**. The full link path and
relation type are returned so the mechanism can be audited.

This design is inspired by hippocampal indexing and pattern completion, but it
is **not** a mechanistic hippocampal neural simulation. It does not claim
CA3 attractor dynamics, dentate sparse coding, independently learned abstract
concepts, probabilistic calibration, or causal discovery.

The three phases are **query-time signal transitions within one memory store**,
not three memory databases or artificial facts. Memory still persists as
`events` and `links`. Retrieval uses a budgeted two-hop competitive
frontier (default beam 8; at most 20 results); association strengths are
uncalibrated model outputs. A two-hop connection never proves that the
outermost episodes describe the same object, happened together, or caused
each other.

### Reproduce and inspect

```bash
python3 sonara.py --db sonara.db --model model.json store "At dawn, Sarah moved the keys to the shelf."
python3 sonara.py --db sonara.db --model model.json store "At noon, Sarah placed the map on the drawer."
python3 sonara.py --db sonara.db --model model.json store "At dusk, Ruby moved the notebook to the drawer."
python3 sonara.py --db sonara.db --model model.json trace Sarah keys --hops 2 --limit 10
python3 sonara.py --db sonara.db --model model.json ask "How are Sarah's keys connected to Ruby's notebook?"
python3 -m unittest discover -s tests -v
```

There is deliberately **no shortcut** for associative questions: deleting the
learned connecting edge must remove the associative answer while leaving direct
factual locations unchanged. Tests under `tests/test_hierarchy.py` exercise
this ablation, two-hop retrieval, restart persistence, evidence provenance,
strict identity boundaries and sparse expansion.

**Remaining limitations:** cue resolution is still restricted to the rule-based
actor/object parser; graph expansion is supervised association, not learned
adaptive attention; no open-domain sentence embeddings, cortical conceptual
hierarchy, associative weight updates from retrieval success, or mechanism
for discovering explanations of novelty exists yet. The 400/400 original
factual benchmark remains **index-based** and should not be presented as
validation of associative pattern completion.
