# Sonara Adaptive Loop — v0

A frozen local LLM wrapped in small, fast-learning state. Sonara perceives each message,
looks up where answers live, sets her modulators, chooses an action, briefs the LLM, and
learns from your next reply. She runs as a never-ending loop with an On/Off button.

This is the alternative to training a model from scratch: the LLM is never trained here.
Nothing in this folder changes `LLM_Training.py`, the world model, or the training code.

Local by default, like the rest of the repo: Ollama with `qwen2.5:1.5b-instruct`, no API
key, no cloud. Standard-library Python only (no torch needed).

## Start

From the `LLM_Training` folder, with the Ollama app open:

```bash
python3 adaptive_loop/run.py
```

Your browser opens the control panel at http://127.0.0.1:8765. The round button is the
On/Off switch. Stop the program with Ctrl+C; the On/Off setting is remembered.

If Ollama is not running, Sonara still starts, with placeholder replies. Open Ollama and
restart to get real ones. For better replies, pull a larger model and set it in `.env`
(see `.env.example`).

## Shared memory with the LLM_Training session

Sonara reads and writes the same evidence store as `LLM_Training.py --mode session`:
`learning_data/sonara_memories/memories.jsonl`, through the repo's `sonara_memory.py`.

- **In:** only your own words, saved after each turn as `source="conversation"`.
- **Out:** up to two relevant past statements, quoted in the brief with their age
  (for example: `evidence: user said "..." (2 days ago)`), presented as your words, not
  as verified facts.
- **Never in:** model replies, the brief, or system-derived salience such as affect
  shifts and recurring topics. Those stay in Sonara's own graph at
  `learning_data/adaptive_loop/sonara.db`.

The graph is the index; the evidence store holds the content.

## Check that everything works

```bash
cd adaptive_loop && python3 -m unittest discover tests -v
```

24 tests cover each build step in the spec, the evidence bridge, the runtime, and the bugs
found in testing. They run offline and never touch your real memory files.

## Run the A/B evaluation (spec step 8)

```bash
python3 adaptive_loop/eval_ab.py
```

Writes a blind comparison to `adaptive_loop/eval/results/`: the full loop versus the same
local LLM with no brief. Rate each pair, then open the key file. Eval runs use a temporary
memory store. Add your own conversations to `eval/scripts.json` (the spec calls for 20).

## Keep her running across reboots (optional)

```bash
bash adaptive_loop/launchd/install.sh      # start at login, restart if she crashes
bash adaptive_loop/launchd/uninstall.sh    # undo
```

launchd only keeps the panel alive. The button still decides whether the loop runs.

## How it fits together

| Step | File | Brain analog |
| --- | --- | --- |
| Perceive the message | sensor.py | Thalamus |
| Find where the answer lives | retrieval.py | Hippocampus |
| Quote what you said before | evidence.py | Hippocampal index → cortex |
| Set curiosity, caution, trust | modulators.py | NE, 5-HT, ACh, DA |
| Choose answer / clarify / hedge | action.py | Basal ganglia |
| Track affect, hot items, corrections | layers.py | Amygdala, PFC, cerebellum |
| Build the ~200-token brief | brief.py | Salience gate |
| Write the reply | generate.py | Cortex (frozen) |
| Learn from your next message | postloop.py | Dopamine |
| Heartbeat, states, the button | runtime.py | Wake / quiet / sleep |

Every tunable number is in `config.py`. Delete `learning_data/adaptive_loop/` to reset
Sonara's graph; your evidence store is not affected.

## Not yet

Sleep replay and consolidation, open loops, and the user, meaning and context layers are
v1–v2 in the spec. Logs are local but not encrypted: do not put patient health information
here.
