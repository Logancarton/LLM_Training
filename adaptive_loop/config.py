"""Every tunable number lives here (CLAUDE.md rule). Change values here, never inline.

All numbers are starting points from the v0 spec, meant to be tuned once you see real runs.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file so API keys never live in code."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

# --- Storage ---------------------------------------------------------------
# This folder lives inside the LLM_Training repo. Private records go under the repo's
# learning_data/ folder, which its .gitignore already keeps out of Git.
REPO_DIR = BASE_DIR.parent
DATA_DIR = Path(os.environ.get("SONARA_DATA_DIR", REPO_DIR / "learning_data" / "adaptive_loop"))
DB_PATH = DATA_DIR / "sonara.db"
CONTROL_PATH = DATA_DIR / "control.json"   # the On/Off flag; only the panel writes it

# --- Evidence store (the repo's sonara_memory.py) --------------------------
# The graph is the index; the evidence store holds the content (hippocampal indexing).
# Same store the LLM_Training.py session writes to. Only the user's own words go in.
MEMORY_BRIDGE = os.environ.get("SONARA_MEMORY_BRIDGE", "1") != "0"
MEMORY_DIR = Path(os.environ.get("SONARA_MEMORY_DIR",
                                 REPO_DIR / "learning_data" / "sonara_memories"))
EVIDENCE_LIMIT = 2            # past statements quoted in the brief
EVIDENCE_MIN_SCORE = 0.35     # sonara_memory's own default (0.15) is too loose for a brief

# --- LLM providers ---------------------------------------------------------
# Local by default, matching the repo: no API keys, no paid services, no cloud.
# "auto"              -> local Ollama if it is running, otherwise "mock"
# "ollama"            -> local Ollama on 127.0.0.1 only (proxy settings ignored)
# "mock"              -> fully offline stand-in (tests, or Ollama not running)
# "anthropic"         -> Claude API: opt-in only (needs ANTHROPIC_API_KEY in .env)
# "openai_compatible" -> LM Studio, mlx_lm.server, or OpenAI
SENSOR_PROVIDER = os.environ.get("SONARA_SENSOR_PROVIDER", "auto")
GENERATOR_PROVIDER = os.environ.get("SONARA_GENERATOR_PROVIDER", "auto")
DEFAULT_MODELS = {
    "ollama": {"sensor": "qwen2.5:1.5b-instruct", "generator": "qwen2.5:1.5b-instruct"},
    "anthropic": {"sensor": "claude-haiku-5-5", "generator": "claude-sonnet-5-5"},
    "openai_compatible": {"sensor": "local-model", "generator": "local-model"},
}
SENSOR_MODEL = os.environ.get("SONARA_SENSOR_MODEL")        # None -> provider default
GENERATOR_MODEL = os.environ.get("SONARA_GENERATOR_MODEL")  # None -> provider default
OLLAMA_URL = os.environ.get("SONARA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_KEEP_ALIVE = "10m"
LLM_TEMPERATURE = 0.2
OPENAI_BASE_URL = os.environ.get("SONARA_OPENAI_BASE_URL", "http://localhost:1234/v1")
LLM_TIMEOUT_SECONDS = 60
SENSOR_MAX_TOKENS = 400
GENERATOR_MAX_TOKENS = 1000
HISTORY_TURNS = 6            # recent exchanges the generator sees (context, never learning)

# --- Embeddings ------------------------------------------------------------
# "auto" uses sentence-transformers if installed, else a zero-dependency hashing embedder.
EMBEDDER = os.environ.get("SONARA_EMBEDDER", "auto")
ST_MODEL = "all-MiniLM-L6-v2"
HASH_DIM = 512
# Similarity thresholds differ by embedder because their scales differ.
SIMILARITY_FLOOR = {"hash": 0.45, "sentence_transformers": 0.75}
REPEAT_SIMILARITY = {"hash": 0.85, "sentence_transformers": 0.92}
TOPIC_SHIFT_SIMILARITY = {"hash": 0.15, "sentence_transformers": 0.35}

# --- Retrieval (hippocampus) -----------------------------------------------
RETRIEVAL_K_BASE = 3          # k = base + round(range * NE)
RETRIEVAL_K_RANGE = 7
EXTRA_HOP_NE = 0.6            # NE above this adds one hop (unless caution is high)
HOP_ATTENUATION = 0.5         # second-hop scores are discounted
LLM_GENERAL_PRIOR = 0.65      # starting weight for "the model already knows this"
CORRECTION_EDGE_WEIGHT = 0.9
CORRECTION_WEIGHT_FLOOR = 0.5
EPISODE_EDGE_WEIGHT = 0.5

# --- Neuromodulator bus ----------------------------------------------------
MOD_BASELINE = {"ne": 0.3, "ht": 0.3, "ach": 0.2}
MOD_HALF_LIFE_TURNS = {"ne": 3, "ht": 6, "ach": 2}
MOD_HALF_LIFE_MINUTES = {"ne": 10, "ht": 30, "ach": 5}   # wall-clock decay between turns
MOD_MAX_STEP = 0.3            # no channel moves more than this per turn...
CAUTION_FAST_ONSET = True     # ...except 5-HT rising: caution has a fast onset, slow offset
NE_NOVELTY_GAIN = 0.4
NE_SURPRISE_GAIN = 0.2
HT_STAKES_GAIN = 0.5
HT_DISTRESS_GAIN = 0.3
HT_LOW_CONFIDENCE_GAIN = 0.3
ACH_CORRECTION_GAIN = 0.6
ACH_TOPIC_SHIFT_GAIN = 0.3
SAFETY_HT = 0.7               # at or above: curiosity-driven behaviour switches off

# --- Action selection (basal ganglia) --------------------------------------
ANSWER_BAR_BASE = 0.4         # answer bar = base + ht_gain * 5-HT
ANSWER_BAR_HT = 0.4
CLARIFY_OFFSET = 0.5          # clarify score = ambiguity - offset

# --- Learning --------------------------------------------------------------
EDGE_LR_BASE = 0.1            # learning rate = base * (0.5 + ACh)
# Outcome when the user moves on without thanks or complaint. Set equal to the expected
# outcome of the prior (2 * 0.65 - 1 = 0.3) so "no news" neither rewards nor punishes.
NEUTRAL_OUTCOME = 0.3
REPLAY_LR_FACTOR = 0.2        # v1: replay re-applies recorded outcomes at this fraction

# --- Affect layer ----------------------------------------------------------
AFFECT_BASELINE_RATE = 0.05
AFFECT_FAST_RATE_BASE = 0.3   # fast rate = base + ach_gain * ACh
AFFECT_FAST_RATE_ACH = 0.4
AFFECT_HALF_LIFE_MINUTES = 120
AFFECT_SHIFT_THRESHOLD = 0.3
DISTRESS_DROP = 0.3           # outcome scoring: valence drop with high arousal = -1
DISTRESS_AROUSAL = 0.6

# --- Temporal window -------------------------------------------------------
WINDOW_CAP_BASE = 5           # capacity = base + round(ne_gain * NE)
WINDOW_CAP_NE = 4
WINDOW_DECAY_PER_TURN = 0.8
WINDOW_HALF_LIFE_MINUTES = 30
WINDOW_REHEAT = 0.5
WINDOW_PROMOTE_REHEATS = 2
WINDOW_MIN_HEAT = 0.05

# --- State brief -----------------------------------------------------------
BRIEF_MAX_CHARS = 800         # roughly 200 tokens
BRIEF_MAX_CORRECTIONS = 3
BRIEF_MAX_HOT = 3

# --- Runtime ---------------------------------------------------------------
HEARTBEAT_SECONDS = 60
AWAKE_MINUTES = 10
SLEEP_AFTER_MINUTES = 120
PANEL_HOST = "127.0.0.1"      # localhost only
PANEL_PORT = int(os.environ.get("SONARA_PORT", "8765"))
