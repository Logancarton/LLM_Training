"""Local research assistant for Sonara, separate from her world model.

A compact Ollama-served LLM supplies tentative explanations on explicit request.
A tiny independently trainable head learns *predicted research usefulness* from
feedback; it does not retrain the LLM or authorize updates to world knowledge.
No API keys, paid services, cloud endpoints, or automatic web requests.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_LOCAL_MODEL = "qwen2.5:1.5b-instruct"
LOCAL_ENDPOINT = "http://127.0.0.1:11434/api/generate"


def query_features(question: str) -> torch.Tensor:
    features = torch.zeros(1, 256)
    words = re.findall(r"[a-z0-9_-]+", question.lower())[:128]
    for word in words:
        idx = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 256
        features[0, idx] += 1
    return features / max(len(words), 1)


class ResearchModel(nn.Module):
    """Predict whether local research is useful, not whether its answer is true."""
    def __init__(self):
        super().__init__()
        self.utility_head = nn.Sequential(nn.Linear(256, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.utility_head(features))


def ollama_local_generate(question: str, model: str) -> str:
    """Send only to the fixed local-loopback Ollama endpoint, with no proxy."""
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", model):
        raise ValueError("Invalid local Ollama model name")
    payload = {
        "model": model,
        "system": ("You are Sonara's LOCAL research assistant, not her world model. "
                   "Offer concise hypotheses and explanations, distinguish observed "
                   "evidence from speculation, and say when you do not know. "
                   "You have no live internet or guaranteed source verification."),
        "prompt": question,
        "stream": False,
        "keep_alive": "3m",
        "options": {"temperature": 0.2, "num_predict": 256, "num_ctx": 2048},
    }
    request = urllib.request.Request(
        LOCAL_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Never inherit HTTP_PROXY/HTTPS_PROXY: the request must remain local.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=90) as response:
            raw = response.read(100_000)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError(f"Local model is missing. Run: ollama pull {model}") from error
        raise ValueError(f"Ollama returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError("Cannot reach local Ollama. Start Ollama or run: ollama serve") from error
    try:
        answer = json.loads(raw.decode("utf-8"))["response"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Unexpected response from local Ollama") from error
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Local Ollama produced an empty response")
    return answer.strip()[:6000]


class ResearchCognition:
    def __init__(self, data_dir: str | Path, local_model: str = DEFAULT_LOCAL_MODEL,
                 transport=None):
        # Separate from the former cloud-provider routing checkpoint, which had
        # incompatible output dimensions and must never silently load or overwrite.
        self.checkpoint = Path(data_dir) / "research_utility_model.pt"
        self.model = ResearchModel()
        self.local_model = local_model
        self.transport = transport or ollama_local_generate
        if self.checkpoint.exists():
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["weights"])
        self.model.eval()

    def expected_utility(self, question: str) -> float:
        self.model.eval()
        with torch.no_grad():
            return float(self.model(query_features(question))[0, 0])

    def research(self, question: str) -> dict:
        if not question.strip() or len(question) > 4000:
            raise ValueError("Research question must contain 1-4000 characters")
        answer = self.transport(question, self.local_model)
        return {
            "model": self.local_model,
            "question": question,
            "answer": answer,
            "status": "unverified_local_model_output",
            "estimated_utility": round(self.expected_utility(question), 3),
        }

    def feedback(self, question: str, utility: float, steps: int = 8) -> dict:
        """Fit usefulness estimates only when rated; never ingest the LLM answer."""
        if not question.strip() or not 0.0 <= utility <= 1.0 or not 1 <= steps <= 32:
            raise ValueError("Usefulness score must be 0..1 and training steps 1..32")
        features = query_features(question)
        target = torch.tensor([[utility]])
        self.model.eval()
        before = float((self.model(features) - target).abs().item())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.004)
        self.model.train()
        for _ in range(steps):
            predicted = self.model(features)
            loss = F.mse_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.model.eval()
        after = float((self.model(features) - target).abs().item())
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self.model.state_dict(), "version": 1}, self.checkpoint)
        return {
            "trained": after < before,
            "utility_error_before": round(before, 4),
            "utility_error_after": round(after, 4),
            "note": "Local feedback fit only; not evidence of general research skill.",
        }
