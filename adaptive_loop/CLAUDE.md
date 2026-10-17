# Sonara Adaptive Loop — v0 rules

These rules come from the v0 build spec. They exist to stop drift toward building a model.
This folder sits inside the LLM_Training repo and follows its rules too: local by default
(no API keys, no paid or cloud services unless the user opts in), user-sourced evidence only,
and honest abstention over guessing. Do not modify files outside this folder.

- The LLM is frozen. Do not train, fine-tune, or build any neural network, embedding model, or language model in v0.
- No component may learn what an LLM can produce on demand. Layers learn only user-specific residuals.
- LLM calls originate in exactly two files: sensor.py and generate.py. llm.py is transport only. Everything else is plain Python arithmetic over SQLite.
- Layers and edges update only from user-turn fields. Never write model output into any layer, edge, or location.
- The evidence store (../sonara_memory.py) receives only the user's own words, after the turn commits. System-derived salience (affect shifts, recurring topics) stays in sonara.db and never enters it.
- One file per component: sensor, retrieval, modulators, action, layers, brief, postloop. Keep each under ~300 lines.
- Every tunable number lives in config.py, never inline.
- The runtime is a continuous heartbeat process. Quiet-wake ticks make no LLM calls. The On/Off flag (learning_data/adaptive_loop/control.json) is changed only by the control panel; no LLM call or tool may change it.
- Follow the spec's build order. Do not start a step until the previous step's test passes. Run `python3 -m unittest discover tests` after every change.
- Do not add layers, features, or phases beyond v0 without asking first. When unsure, stop and ask.

## File map

| File | Brain analog | Job |
| --- | --- | --- |
| sensor.py | Thalamus | Cheap LLM call → JSON tags (cue, affect, stakes, ambiguity, correction) |
| retrieval.py | Hippocampus | Similar cues → edges → where the answer lives |
| modulators.py | Neuromodulators | NE, 5-HT, ACh levels and DA prediction error |
| action.py | Basal ganglia | ANSWER / CLARIFY / RETRIEVE_MORE / HEDGE |
| layers.py | Amygdala, PFC, cerebellum | Affect, temporal window, corrections |
| brief.py | Salience gate | The ~200-token state brief |
| generate.py | Cortex | The frozen LLM writes the reply |
| postloop.py | Dopamine learning | Scores turn N from turn N+1, updates edges |
| brain.py | — | Runs one turn in order; heartbeat decay |
| runtime.py | Brain states | Awake / quiet / sleep / off, heartbeat, the button |
| server.py, panel.html | — | Local control panel |
| evidence.py | Hippocampal index → cortex | Bridge to the repo's sonara_memory.py evidence store |
| config.py | — | Every tunable number |
