"""Opt-in research model: learns which external AI source is useful, never owns world truth.

No API keys or external answers enter checkpoints. Only explicit research prompts leave
the local machine; replies are candidate evidence and require independent verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROVIDERS = ("openai", "anthropic", "gemini")
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY"}


def query_features(question: str) -> torch.Tensor:
    """Stable hashed question features; no vocabulary expansion or stored secrets."""
    features = torch.zeros(1, 256)
    words = re.findall(r"[a-z0-9_-]+", question.lower())[:128]
    for word in words:
        idx = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 256
        features[0, idx] += 1
    return features / max(len(words), 1)


class ResearchModel(nn.Module):
    """Small trainable provider-utility estimator, not a second truth authority."""
    def __init__(self):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(256, 24), nn.Tanh(), nn.Linear(24, len(PROVIDERS)))

    def forward(self, features):
        return torch.sigmoid(self.router(features))


def _json_post(url: str, headers: dict, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=35) as response:
        return json.loads(response.read(800_000).decode("utf-8"))


def ask_provider(provider: str, question: str) -> str:
    """One explicit outbound request. Provider output is always untrusted evidence."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    key = os.environ.get(KEYS[provider], "")
    if not key:
        raise ValueError(f"Set {KEYS[provider]} to enable this provider")
    system = ("Respond with verifiable, concise factual analysis, label uncertainty, "
              "and provide original references when available. Do not claim verification.")
    if provider == "openai":
        result = _json_post("https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {"model": os.environ.get("SONARA_OPENAI_MODEL", "gpt-4.1-mini"),
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": question}], "max_completion_tokens": 450})
        answer = result["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        result = _json_post("https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": os.environ.get("SONARA_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
             "max_tokens": 450, "system": system,
             "messages": [{"role": "user", "content": question}]})
        answer = "\n".join(part.get("text", "") for part in result["content"] if part.get("type") == "text")
    else:
        import urllib.parse
        model = os.environ.get("SONARA_GEMINI_MODEL", "gemini-2.5-flash")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", model):
            raise ValueError("Invalid Gemini model name")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?"
               + urllib.parse.urlencode({"key": key}))
        result = _json_post(url, {},
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": question}]}],
             "generationConfig": {"maxOutputTokens": 450}})
        answer = "\n".join(part.get("text", "") for part in
                result["candidates"][0]["content"]["parts"])
    return str(answer or "").strip()[:6000]


class ResearchCognition:
    def __init__(self, data_dir: str | Path, transport=ask_provider):
        self.checkpoint = Path(data_dir) / "research_model.pt"
        self.model = ResearchModel()
        self.transport = transport
        if self.checkpoint.exists():
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["weights"])
        self.model.eval()

    def available(self):
        return tuple(p for p in PROVIDERS if os.environ.get(KEYS[p]))

    def select(self, question: str, allowed: tuple[str, ...]) -> str:
        if not allowed or any(p not in PROVIDERS for p in allowed):
            raise ValueError("No configured permitted AI providers")
        self.model.eval()
        with torch.no_grad():
            scores = self.model(query_features(question))[0]
        return max(allowed, key=lambda provider: float(scores[PROVIDERS.index(provider)]))

    def research(self, question: str, provider: str | None = None) -> dict:
        if not question.strip() or len(question) > 4000:
            raise ValueError("Research question must contain 1-4000 characters")
        permitted = self.available()
        if not permitted:
            raise ValueError("No AI provider configured. Set an API key environment variable.")
        selected = provider or self.select(question, permitted)
        if selected not in permitted:
            raise ValueError("Provider was not enabled with an API key")
        answer = self.transport(selected, question)
        return {"provider": selected, "question": question, "answer": answer,
                "status": "unverified_external_evidence"}

    def feedback(self, question: str, provider: str, utility: float, steps: int = 8) -> dict:
        """Train routing only from explicitly graded usefulness, never from answer content."""
        if provider not in PROVIDERS or not 0.0 <= utility <= 1.0:
            raise ValueError("Provider invalid or score outside [0,1]")
        features = query_features(question)
        target = torch.tensor([utility])
        index = PROVIDERS.index(provider)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.004)
        before = float((self.model(features)[0, index] - target[0]).abs().detach())
        for _ in range(steps):
            predicted = self.model(features)[0, index:index+1]
            loss = F.mse_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.model.eval()
        after = float((self.model(features)[0, index] - target[0]).abs().detach())
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self.model.state_dict()}, self.checkpoint)
        return {"trained": after < before, "utility_error_before": round(before, 4),
                "utility_error_after": round(after, 4),
                "note": "Local feedback fit, not proven cross-question provider routing"}
