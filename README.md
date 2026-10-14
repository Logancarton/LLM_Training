# LLM_Training: Miniature Cognitive LLM & Sonara Memory Architecture

A modular, experimental PyTorch architecture exploring language generation, psycholinguistic semantic frames, continual learning, and a structured cognitive memory system for **Sonara**.

---

## Two-model Sonara architecture (experimental)

The live session now uses **two independently trainable neural roles** beside the existing
language reader/writer. These are research mechanisms, **not** proof of autonomous
understanding or broad factual intelligence:

1. **World cognition** (sonara_world.py) reads user observations into the **existing**
   episodic memory store, preserving event-time and observation provenance. Explicitly
   stated locations can be answered from evidence; unknown locations are acknowledged.
   A small neural model separately estimates whether text signals a location/state change.
   The user must explicitly request bounded training. It uses weak pattern-supervised
   examples and a held-out-loss gate; it does **not** yet learn causal state dynamics.
2. **Research cognition** (sonara_research.py) uses a small neural utility predictor
   to choose among configured ChatGPT/OpenAI, Claude/Anthropic, or Gemini API calls.
   Only explicit `/research` prompts may leave the machine, and only with
   `--enable-research`. Users can score usefulness with `/rate-research`; only
   the **provider-selection policy** learns from that rating. External responses
   remain **unverified evidence** and never automatically enter episodic memory,
   world-model weights, or the older approved Q/A training pipeline.

The earlier Stage 1/2/3 language models are preserved for language generation
and the existing research benchmarks, not treated as the world-model owner.
Old `--mode learn` still performs legacy approved-Q/A language fine-tuning.
The world and research models do **not** share a duplicate memory database.

```text
User experience ─> SonaraMemorySystem (canonical evidence + event times)
                       │
                       └──> WorldCognition ──> evidence-grounded state answer
                                  │                  │
                                  └──> neural change predictor (explicit training)

Explicit /research ─> ResearchCognition ─> permitted external AI API
                            │                     │
                   neural provider policy <── rated utility (/rate-research)
                                                  │
                                         unverified external evidence
                                        (never auto-promoted to world facts)
```

**Try the real session:**

```bash
python LLM_Training.py --mode session
```

Within the session:

```text
Jake moved the ball into the pond.
/ask Where is the ball?
/train-world
/research Why do objects float?
```

The first state answer is attributed to evidence. Explicit time-qualified
questions currently abstain rather than pretending historical reasoning.
The human-readable memory journal is exported on demand (for example,
`python LLM_Training.py --mode memories`) rather than regenerated after
every stored observation. Raw JSONL memory is still loaded into RAM, so
large-scale persistence/indexing remains future work. `/train-world` needs at
least **four explicit location observations and four unrelated observations**;
it will not invent training targets from unconfirmed external AI answers.
`/research` remains disabled until the operator opts in:

```bash
export OPENAI_API_KEY="..."       # Or ANTHROPIC_API_KEY / GEMINI_API_KEY
python LLM_Training.py --mode session --enable-research
```

On Windows PowerShell, use `$env:OPENAI_API_KEY = "..."` instead of export.
Research prompts are sent to the selected third-party provider and may incur
API charges. In a session, `/rate-research 0.8` teaches the routing model
using explicit usefulness feedback. Provider replies should be independently
checked. `--research-provider openai` pins a provider.

**Development gate:** `python -m unittest -q test_two_models`; the
GitHub Actions benchmark also exercises the actual session path.
**Not yet demonstrated:** generalized causal prediction, calibrated world
uncertainty, continual self-supervised learning, or research-policy transfer
to unseen questions.

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
