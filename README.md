# LLM_Training: Miniature Cognitive LLM & Sonara Memory Architecture

A modular, experimental PyTorch architecture exploring language generation, psycholinguistic semantic frames, continual learning, and a structured cognitive memory system for **Sonara**.

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
* **Dual Persistence**: Machine-auditable append-only JSONL log (`memories.jsonl`) + Human-readable Markdown Journal (`sonara_memory_journal.md`).

### 3. Epistemic Rigor & Built-In Causal Checks
* **Causal Idea Check**: Evaluates generation loss with wrong/shuffled idea vectors against the correct idea vector to verify whether the generator actually conditions on semantic intent.
* **Swapped-Question Conditioning**: Validates whether the model is learning conditional associations or merely memorizing targets.
* **Retention Guardrails & Replay**: Fine-tuning on personal approved Q/A replays general training samples to prevent catastrophic forgetting.

---

## Project Structure

```
├── LLM_Training.py               # Core 3-stage neural pipeline, CLI, and interactive session
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
