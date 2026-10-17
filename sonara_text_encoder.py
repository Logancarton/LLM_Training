"""sonara_text_encoder.py — Frozen pretrained text vectors for the world model.

Fetches embeddings from a LOCAL Ollama embedding model (loopback only, no proxy,
no cloud) and caches them. The vectors are frozen inputs: the world model learns
only a projection on top of them. They are derived text features, never memory
or evidence, and a cache file can always be rebuilt from the text.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch

DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_ENDPOINT = "http://127.0.0.1:11434/api/embed"


class PretrainedTextEncoder:
    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        cache_path: Optional[Union[str, Path]] = None,
        offline: bool = False,
        batch_size: int = 128,
    ):
        """offline=True never contacts Ollama and raises on a cache miss
        (used by benchmark workers so every system sees identical vectors)."""
        self.model = model
        self.offline = offline
        self.batch_size = batch_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.dim: Optional[int] = None
        if cache_path is not None and Path(cache_path).exists():
            payload = torch.load(cache_path, map_location="cpu")
            if payload["model"] != model:
                raise ValueError(f"Embedding cache was built with {payload['model']}, not {model}")
            self.cache = payload["vectors"]
            self.dim = payload["dim"]

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model, "dim": self.dim, "vectors": self.cache}, path)

    def _fetch(self, texts: Sequence[str]) -> None:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            EMBED_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=600) as response:
            body = json.load(response)
        if "embeddings" not in body:
            raise RuntimeError(f"Local embedding model failed: {body.get('error', body)}")
        for text, vec in zip(texts, body["embeddings"]):
            self.cache[text] = torch.tensor(vec, dtype=torch.float32)
        self.dim = len(body["embeddings"][0])

    def ensure(self, texts: Iterable[str]) -> int:
        """Embed any texts not yet cached. Returns how many were fetched."""
        missing = sorted({t for t in texts if t not in self.cache})
        if missing and self.offline:
            raise KeyError(f"{len(missing)} texts missing from offline embedding cache, e.g. {missing[0]!r}")
        for i in range(0, len(missing), self.batch_size):
            self._fetch(missing[i:i + self.batch_size])
        return len(missing)

    def vector_dim(self) -> int:
        if self.dim is None:
            self.ensure(["the"])
        return self.dim

    def vectors(self, texts: Sequence[str]) -> torch.Tensor:
        """(N, dim) frozen vectors, fetching misses unless offline."""
        self.ensure(texts)
        return torch.stack([self.cache[t] for t in texts], dim=0)
