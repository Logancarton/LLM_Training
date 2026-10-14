# LLM_Training: Miniature Cognitive LLM & Sonara Memory Architecture

A modular, experimental PyTorch architecture exploring language generation, psycholinguistic semantic frames, continual learning, and a structured cognitive memory system for **Sonara**.

---

## Two-model Sonara architecture (experimental; fully local)

**No paid AI API, subscription, or API key is needed.** The live Sonara session
uses two separate cognitive roles:

1. **World cognition** (`sonara_world.py`) keeps grounded observations, entity
   state, timestamps, and corrections in Sonara's original **single** memory
   store. A tiny neural change classifier can be trained on bounded, weakly
   labeled samples with a holdout acceptance check. It does **not** yet learn
   general causal laws.
2. **Local research cognition** (`sonara_research.py`) calls a small, quantized
   language model served by **Ollama on 127.0.0.1**. It suggests explanations
   and hypotheses only on explicit `/research` requests. A separate tiny
   neural utility head learns from `/rate-research` scores to estimate how
   useful research questions might be. This does **not** fine-tune the local
   language model or automatically update Sonara's world beliefs.

The existing three-stage language reader and writer are retained for backward
compatibility and language experiments. The old approved-Q/A neural learning
path remains distinct from the new bounded World Model. No parallel episodic
memory or automatic training on all stored text has been introduced.

```text
                            Incoming experience
                                    |
                    SonaraMemorySystem (one evidence store)
                                    |
                           WorldCognition
                          (neural + evidence)

User /research request --> ResearchCognition --> Ollama localhost
                                 |                    |
                     Tiny utility learner     Small frozen local LLM
                    (explicit rated feedback)        |
                                                Unverified hypotheses
                                      (not automatically world knowledge)
```

### Laptop setup

Install [Ollama](https://ollama.com/download) for Mac or Windows. Download the
small research model **once** while online:

```powershell
ollama pull qwen2.5:1.5b-instruct
ollama list
```

Qwen2.5 1.5B Instruct uses a roughly **986 MB quantized model download**,
though running it requires additional RAM. It can operate without internet
after initial installation and download. Start Ollama if it is not already
running (`ollama serve`), then start Sonara:

```powershell
python LLM_Training.py --mode session --enable-research
```

In Sonara's interactive session:

```text
Jake moved the ball into the pond.
Jake moved the ball into the shed.
/ask Where is the ball?
/ask Where was the ball before it was at the shed?
/research Why do some objects float?
/rate-research 0.8
```

Optionally select another **installed** local model with
`--research-model qwen2.5:3b`, which needs more resources.

The research model makes **no outgoing cloud calls** and never silently
retrieves web pages. This means it cannot independently verify current facts
or read scientific papers without a future separately controlled local document
ingestion mechanism. Local responses are hypotheses, not authenticated evidence.

The World Model's `/train-world` command remains bounded and explicit.
Presently it needs at least four explicit location observations and four
unrelated observations.

World-state history questions like "Where was the ball before it was at the
shed?" now find the earlier **reported** location using the same timestamped
memory evidence, even after restart. The World owner identifies a supported
before-condition, finds the referenced shed observation, and returns the
most recent different observed location before it. An absent or repeated
anchor, contradictory entity reference, or unsupported relative/time-qualified
wording returns `Unknown` instead of silently returning the latest location.
A reported location does not prove a continuously observed trajectory.

**This is bounded, explicit temporal query logic, not newly learned
temporal cognition.** The World Model neural weights are unchanged by
these questions. Truly generalized event reasoning is still unproven.
The memory log stays append-only and reloads into RAM; large-scale persistent
indexing is **not** implemented yet. Export the human-readable journal on demand
using `python LLM_Training.py --mode memories`.

**Development:** Run `python -m unittest -q test_two_models`. Tests mock the
Ollama connection: passing CI proves routing, storage boundaries, and feedback
behavior—not actual local inference, physical laptop performance, generalized
world understanding, or independent autonomous learning. Existing legacy
approved-Q/A learning is not removed in this change.

---

## Architecture Overview

```
                                      ┌────────────────────────────────────────┐
                                      │        Sonara Memory System            │
                                      │  (Episodic, Semantic, Reflections)     │
                                      │  ISO Timestamps + Cognitive Retrieval  │
                                      └──────────────────┬─────────────────────┘
                                                         │
                                               recalled context / frame
                                                         ▼
[ Situation / Prompt ] ──► [ Stage 2: IdeaReader (GRU) ] ──► [ Compact Idea Vector ]
                                                                     │
                                                                     ▼
[ Detailed Words History ] ──────────────────────────────► [ Stage 3: Generator ] ──► [ Detailed Output ]
                                                            (Conditioned Transformer)
```

### 1. The Three Neural Learning Stages
* **Stage 1: Word Prediction** — Causal self-attention Transformer trained to predict next tokens autoregressively.
* **Stage 2: Situation Understanding (`IdeaReader`)** — GRU sequence encoder compressing short situations into a compact representation and classifying semantic roles: `WHO`, `ACTION`, `WHAT`, `WHERE`, `WHEN`, and `CONDITION`.
* **Stage 3: Idea-Guided Generation** — Conditioned Transformer that generates rich, detailed language guided by the compressed idea representation.

### 2. Sonara Memory Architecture (`sonara_memory.py`)
Human-like episodic and semantic memory system decoupled from static neural weights:
* **Multi-Resolution Timestamps**: Every memory is stamped with timezone-aware ISO-8601, human-readable local time, Unix epoch, and relative time expressions (*"3 minutes ago"*, *"yesterday"*).
* **Cognitive Multi-Factor Recall**:
  $$\text{Score} = 0.55 \cdot \text{Relevance} + 0.25 \cdot \text{Recency Decay} (e^{-\lambda \Delta t}) + 0.20 \cdot \text{Importance}$$
* **Memory Categorization**:
  * `episodic`: Timestamped events and conversations.
  * `semantic`: Distilled facts and concepts.
  * `preference`: User preferences, communication style, and habits.
  * `reflection`: Higher-level insights synthesized from recurring experiences.
* **Semantic Frame Mapping**: Automatically parses observations into `WHO / ACTION / WHAT` frames matching the `IdeaReader`.
* **Entity Inverted Index**: Dynamic graph of known people, places, and objects.
* **Consolidation / Reflection Engine**: Consolidates clusters of episodic experiences into high-level synthesized reflections (similar to sleep/dream consolidation).
* **Raw persistence + optional export**: Machine-auditable append-only JSONL log (`memories.jsonl`); the human-readable Markdown Journal (`sonara_memory_journal.md`) is generated on demand to avoid rewriting it for every new memory.

### 3. Epistemic Rigor & Built-In Causal Checks
* **Causal Idea Check**: Evaluates generation loss with wrong/shuffled idea vectors against the correct idea vector to verify whether the generator actually conditions on semantic intent.
* **Swapped-Question Conditioning**: Validates whether the model is learning conditional associations or merely memorizing targets.
* **Retention Guardrails & Replay**: Fine-tuning on personal approved Q/A replays general training samples to prevent catastrophic forgetting.

---

## Project Structure

```
├── LLM_Training.py               # Core 3-stage neural pipeline, CLI, and interactive session
├── sonara_world.py               # Evidence-grounded world owner and bounded neural transition estimation
├── sonara_research.py            # Opt-in external AI research and trainable provider selection
├── test_two_models.py            # Live-owner learning/isolation verification
├── sonara_memory.py              # Cognitive memory system with timestamps & reflection
├── test_sonara_memory.py         # Automated verification suite for the memory system
├── test_five_minute_experience.py # 5-minute sequential experience simulation & recall
├── test_deliberation.py          # Deliberative System 2 loop (Assess -> Create -> Verify -> Re-assess)
├── test_without_ideas.py         # Ablation experiment comparing conditioned vs blind generation
├── ideas.txt                     # 287 aligned core situations and semantic frames
├── details.txt                   # 287 detailed descriptive sentence targets
├── LLM_Training_checkpoint_v2.pt # Trained model checkpoint
└── README.md                     # Project documentation
```

---

## Quickstart

### 1. Requirements
* Python 3.10+
* PyTorch (`python3 -m pip install torch`)

### 2. Interactive Session with Sonara
Start an interactive session with full memory and neural generation:
```bash
python3 LLM_Training.py --mode session --checkpoint LLM_Training_checkpoint_v2.pt
```

**Commands inside Session:**
* Natural input: Any observation you enter is automatically timestamped, entity-indexed, and stored in Sonara's memory.
* Natural questions: Ending with `?` or starting with `/ask` queries both memory and neural weights.
* `/remember <text>`: Save an explicit priority memory.
* `/recall <query>`: Search Sonara's memory bank using cognitive relevance and recency scoring.
* `/timeline`: Display chronological memory timeline.
* `/reflect`: Consolidate recent experiences into higher-level insights.
* `/feedback <text>`: Record unverified feedback on the previous response.

### 3. Memory System CLI
Interact with Sonara's memory directly:
```bash
# Save a memory with timestamp
python3 sonara_memory.py --remember "Logan is training Sonara on an Apple Silicon Mac."

# Search memories
python3 sonara_memory.py --recall "Mac"

# View memory timeline
python3 sonara_memory.py --timeline

# Interactive memory console
python3 sonara_memory.py
```

### 4. Inspect Sonara's Memory Journal
```bash
python3 LLM_Training.py --mode memories
```

### 5. Run Experiments & Tests
```bash
# Test memory system suite
python3 test_sonara_memory.py

# Simulate 5 minutes of continuous living experiences
python3 test_five_minute_experience.py

# Test deliberative generation (System 2 / Process Supervision)
python3 test_deliberation.py

# Run idea ablation study (with vs without ideas)
python3 test_without_ideas.py

# Run built-in self-test
python3 LLM_Training.py --mode self-test
```

---

## Training from Scratch

To pretrain all three stages from scratch:
```bash
python3 LLM_Training.py --examples 8000 --epochs 2 --idea-epochs 8 --width 64 --layers 2 --checkpoint my_checkpoint.pt
```

---

## License
MIT License. Created by Logan Carton.


## Controlled experience-learning research (synthetic only)

Run `python experiment_experience_learning.py --seeds 1 --steps 60` to compare
the existing trained reader/writer before and after synthetic Q/A supervision.
The reproducible harness holds out entire actor–object combinations and reports:
weights-only accuracy, memory retrieval **evidence coverage**, neural prediction
with current observations, and neural prediction with retrieved observations.
It also reports uncertainty questions, correction cases, a shuffled-context
ablation, parameter change, and sampled general-task retention.

Interpretation limits: retrieval evidence coverage is NOT answer accuracy;
synthetic oracle Q/A targets are supervised data, NOT autonomous learning;
the context-fed neural evaluation is not the same as the production live session.
A reduction in training loss without held-out gains is a negative result.
No personal session data or modified model checkpoints should be committed.
The GitHub Actions job uploads a JSON artifact for scientific review.
