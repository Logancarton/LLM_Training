"""Turns cue text into vectors so similar situations can be found.

Default is a zero-dependency hashing embedder (word and word-pair features).
If sentence-transformers is installed, "auto" uses it instead for better matching.
Embeddings are a perception tool from outside; nothing here is trained.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import List, Optional

import config

_STOP = set(("a an the and or but if of to in on at for with is are was were be been it "
             "this that these those i you he she we they me my your our their do does did "
             "so as by from am can could would should will just about what how").split())


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP]


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = config.HASH_DIM):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        words = _tokens(text)
        feats = [(w, 1.0) for w in words] + [(a + " " + b, 0.5) for a, b in zip(words, words[1:])]
        for feat, weight in feats:
            vec[zlib.crc32(feat.encode()) % self.dim] += weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    name = "sentence_transformers"

    def __init__(self, model_name: str = config.ST_MODEL):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return [float(x) for x in self.model.encode(text, normalize_embeddings=True)]


def get_embedder(choice: Optional[str] = None):
    choice = choice or config.EMBEDDER
    if choice in ("auto", "sentence_transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if choice == "sentence_transformers":
                raise
    return HashEmbedder()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
