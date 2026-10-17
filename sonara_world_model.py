"""sonara_world_model.py — Genuine Trainable Neural World Model for Sonara.
=============================================================================

PURPOSE
    Provides Sonara with a genuinely trainable neural World Model that learns
    relationships between events, objects, actions, locations, and time from
    sequences of experiences.

PRINCIPLES & CONSTRAINTS
    - No hardcoded keyword state-machine or rule-based QA shortcut.
    - Learns state transitions, entity continuity, and temporal relationships
      purely through neural parameter updates via gradient descent.
    - Subword/character-level hash embeddings ensure unfamiliar entities
      (unseen names, objects, places) have non-degenerate representations.
    - Sequential/recurrent processing of timestamped experiences maintains
      an evolving world state over time (both causal forward and anchor-aware
      backward transitions).
    - Dual pointer-generator head combines closed-class relational answers
      ("unknown", "yes", "no") with hierarchical event-attended token pointer
      distributions over observed experiences without vocabulary leakage.
    - Canonical episodic memory (SonaraMemorySystem) remains the sole authoritative
      evidence store; neural predictions never silently overwrite observed evidence.
    - Checkpoint persistence supports atomic saving, process restart, and verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. CONSTANTS & TOKENIZATION
# =====================================================================

CLOSED_VOCABULARY = (
    "<pad>",
    "<unk>",
    "unknown",
    "yes",
    "no",
    "before",
    "after",
    "true",
    "false",
)

CLOSED_TO_ID = {word: idx for idx, word in enumerate(CLOSED_VOCABULARY)}
ID_TO_CLOSED = {idx: word for idx, word in enumerate(CLOSED_VOCABULARY)}
PAD_ID = CLOSED_TO_ID["<pad>"]
UNK_ID = CLOSED_TO_ID["<unk>"]
UNKNOWN_ID = CLOSED_TO_ID["unknown"]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "that", "this", "it"
}

STRUCTURAL_WORDS = {
    "where", "what", "who", "when", "did", "does", "is", "was", "will", "be", "if",
    "before", "after", "happen", "happened", "in", "into", "to", "from", "at", "on",
    "by", "with", "the", "a", "an", "it", "never", "not", "no", "yes", "correction",
    "reached", "stayed", "left", "carried", "moved", "placed", "held", "transferred",
    "inspected", "brought", "chased", "chase", "?", ".", ";", ":"
}


def tokenize(text: str) -> List[str]:
    """Extract normalized lowercase alphanumeric tokens and punctuation."""
    return re.findall(r"[\w]+(?:['-][\w]+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def is_content_token(token: str) -> bool:
    """Return True if token is a non-stopword alphanumeric token suitable for entity matching."""
    t = token.lower()
    return t not in STOPWORDS and len(t) > 1 and any(c.isalnum() for c in t)


def extract_content_candidates(events: Sequence[str]) -> List[str]:
    """Extract distinct non-stopword tokens from observations as pointer candidates."""
    seen = set()
    candidates = []
    for event in events:
        for token in tokenize(event):
            t = token.lower()
            if is_content_token(t) and t not in seen:
                seen.add(t)
                candidates.append(t)
    return candidates


# =====================================================================
# 2. NEURAL EMBEDDINGS (WORDS + SUBWORD/CHAR N-GRAM HASHES)
# =====================================================================

class CharSubwordEmbedding(nn.Module):
    """
    Hybrid token + character n-gram hash embedding.
    Ensures that unfamiliar entities (e.g. 'sophia', 'frisbee', 'attic')
    receive non-zero, morphologically distinct, and trainable representations.
    """
    def __init__(self, vocab_size: int = 512, hash_buckets: int = 1024, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.hash_buckets = hash_buckets
        self.word_embeddings = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.hash_embeddings = nn.Embedding(hash_buckets, dim)
        self.projection = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self._ngram_cache: Dict[str, List[int]] = {}
        # Internal ablation switch: no learned weights or representational features change.
        self._vectorized_char_bags = True

    def _hash_ngram(self, ngram: str) -> int:
        h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.hash_buckets

    def _get_buckets(self, token: str) -> List[int]:
        t = token.lower()
        cached = self._ngram_cache.get(t)
        if cached is not None:
            return cached
        wrapped = f"<{t}>"
        ngrams = [wrapped[i:i+n] for n in (2, 3) for i in range(len(wrapped) - n + 1)]
        if not ngrams:
            ngrams = [wrapped]
        buckets = [self._hash_ngram(ng) for ng in ngrams]
        self._ngram_cache[t] = buckets
        return buckets

    def embed_char_only(self, token_strings: Sequence[str], device: torch.device) -> torch.Tensor:
        """Compute character n-gram hash embeddings (L, dim) for a sequence of tokens."""
        if not token_strings:
            return torch.zeros(1, self.dim, device=device)
        all_buckets: List[int] = []
        offsets: List[Tuple[int, int]] = []
        for tok in token_strings:
            b = self._get_buckets(tok)
            start = len(all_buckets)
            all_buckets.extend(b)
            offsets.append((start, len(all_buckets)))
        flat_ids = torch.tensor(all_buckets, dtype=torch.long, device=device)
        if self._vectorized_char_bags:
            # Equivalent per-token means in one embedding-bag kernel. Hash indices
            # are weight-independent, while gradients still reach hash_embeddings.
            starts = torch.tensor([s for s, _ in offsets], dtype=torch.long, device=device)
            return F.embedding_bag(
                flat_ids, self.hash_embeddings.weight, starts, mode="mean"
            )
        flat_vecs = self.hash_embeddings(flat_ids)
        char_vecs = [flat_vecs[s:e].mean(dim=0) for s, e in offsets]
        return torch.stack(char_vecs, dim=0)

    def embed_identity(self, token_strings: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Compute L2-normalized morphological identity vectors for content tokens,
        zeroing stopwords and structural syntax words so entity matching only binds
        entities, objects, places, and timestamps.
        """
        if not token_strings:
            return torch.zeros(1, self.dim, device=device)
        char_vecs = self.embed_char_only(token_strings, device)
        normed = F.normalize(char_vecs, p=2, dim=-1)
        mask = torch.tensor(
            [1.0 if (is_content_token(t) and t.lower() not in STRUCTURAL_WORDS) else 0.0 for t in token_strings],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        return normed * mask

    def embed_token(self, token: str, word_id: int, device: torch.device) -> torch.Tensor:
        t_low = token.lower()
        is_struct = t_low in STRUCTURAL_WORDS
        eff_id = UNK_ID if (word_id >= self.vocab_size or token.isdigit() or not is_struct) else word_id
        w_id = torch.tensor([eff_id], device=device)
        w_vec = self.word_embeddings(w_id).squeeze(0)  # (dim,)
        bucket_ids = torch.tensor(self._get_buckets(token), dtype=torch.long, device=device)
        char_vec = self.hash_embeddings(bucket_ids).mean(dim=0)  # (dim,)
        scale = 1.0 if is_struct else 0.1
        combined = torch.cat([w_vec, char_vec * scale], dim=-1)
        return self.norm(self.projection(combined))

    def forward(self, token_strings: Sequence[str], word_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        """Embed a sequence of tokens into a (seq_len, dim) tensor."""
        if not token_strings:
            return torch.zeros(1, self.dim, device=device)
        eff_ids = []
        scales = []
        for tok, wid in zip(token_strings, word_ids):
            t_low = tok.lower()
            is_struct = t_low in STRUCTURAL_WORDS
            if wid >= self.vocab_size or tok.isdigit() or not is_struct:
                eff_ids.append(UNK_ID)
                scales.append(0.1)
            else:
                eff_ids.append(wid)
                scales.append(1.0)
        w_tensor = torch.tensor(eff_ids, dtype=torch.long, device=device)
        w_vecs = self.word_embeddings(w_tensor)  # (L, dim)
        char_vecs = self.embed_char_only(token_strings, device)  # (L, dim)
        scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(-1)
        combined = torch.cat([w_vecs, char_vecs * scale_tensor], dim=-1)
        return self.norm(self.projection(combined))


# =====================================================================
# 3. EVENT ENCODER & TEMPORAL ENCODING
# =====================================================================

class TemporalEncoding(nn.Module):
    """Encodes step order / relative time into representation space."""
    def __init__(self, dim: int = 64, max_steps: int = 64):
        super().__init__()
        self.dim = dim
        self.temporal_embeddings = nn.Embedding(max_steps, dim)

    def forward(self, step_indices: torch.Tensor) -> torch.Tensor:
        clamped = step_indices.clamp(0, self.temporal_embeddings.num_embeddings - 1)
        return self.temporal_embeddings(clamped)


class EventEncoder(nn.Module):
    """
    Encodes an event sentence (or question) into a fixed-width vector
    using token embeddings, cross-aligned query/temporal features, and a
    bidirectional GRU with attention pooling.
    """
    def __init__(self, embedding: CharSubwordEmbedding, dim: int = 64):
        super().__init__()
        self.embedding = embedding
        self.dim = dim
        self.align_proj = nn.Linear(dim, dim)
        self.scalar_proj = nn.Sequential(
            nn.Linear(4, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.attn_query = nn.Linear(dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: List[str],
        word_to_id: Dict[str, int],
        device: torch.device,
        return_token_states: bool = False,
        aligned_vecs: Optional[torch.Tensor] = None,
        match_scalars: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not tokens:
            zero_vec = torch.zeros(self.dim, device=device)
            if return_token_states:
                return zero_vec, torch.zeros(1, self.dim, device=device)
            return zero_vec
        word_ids = [word_to_id.get(t.lower(), UNK_ID) for t in tokens]
        token_vecs = self.embedding(tokens, word_ids, device)  # (L, dim)

        if aligned_vecs is not None:
            token_vecs = token_vecs + self.align_proj(aligned_vecs)
        if match_scalars is not None:
            token_vecs = token_vecs + self.scalar_proj(match_scalars)
        token_vecs = self.norm(token_vecs)

        h, _ = self.gru(token_vecs.unsqueeze(0))  # (1, L, dim)
        h = h.squeeze(0)  # (L, dim)

        attn_scores = self.attn_query(h).softmax(dim=0)  # (L, 1)
        pooled = (h * attn_scores).sum(dim=0)  # (dim,)
        endpoints = torch.cat([h[-1, :self.dim // 2], h[0, self.dim // 2:]], dim=-1)
        norm_pooled = self.norm(pooled + endpoints)
        if return_token_states:
            return norm_pooled, h
        return norm_pooled


# =====================================================================
# 4. RECURRENT WORLD STATE & RELATIONAL REASONING
# =====================================================================

class WorldStateTracker(nn.Module):
    """
    Maintains an evolving world state representation updated sequentially
    as observations occur over time:
      - Forward GRUCell tracks causal progression (s_fwd_t = GRU(z_t, s_fwd_{t-1}))
      - Backward GRUCell propagates subsequent temporal anchors (e.g. 'before 12:16')
        to immediately preceding states (s_bwd_t = GRU(z_t, s_bwd_{t+1})).
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.event_feat_proj = nn.Sequential(
            nn.Linear(5, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.cell = nn.GRUCell(dim, dim)
        self.bwd_cell = nn.GRUCell(dim, dim)
        self.state_fusion = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        event_vectors: torch.Tensor,
        event_local_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        event_vectors: (num_events, dim)
        event_local_features: optional (num_events, 5) local event match features
        Returns:
            all_states: (num_events, dim)
            final_state: (dim,)
        """
        num_events = event_vectors.size(0)
        device = event_vectors.device
        if num_events == 0:
            zero_state = torch.zeros(self.dim, device=device)
            return zero_state.unsqueeze(0), zero_state

        if event_local_features is not None:
            inputs = event_vectors + self.event_feat_proj(event_local_features)
        else:
            inputs = event_vectors

        fwd_state = torch.zeros(self.dim, device=device)
        fwd_states = []
        for t in range(num_events):
            fwd_state = self.cell(inputs[t].unsqueeze(0), fwd_state.unsqueeze(0)).squeeze(0)
            fwd_states.append(fwd_state)

        bwd_state = torch.zeros(self.dim, device=device)
        bwd_states = [torch.zeros(self.dim, device=device) for _ in range(num_events)]
        for t in range(num_events - 1, -1, -1):
            bwd_state = self.bwd_cell(inputs[t].unsqueeze(0), bwd_state.unsqueeze(0)).squeeze(0)
            bwd_states[t] = bwd_state

        fused_states = []
        zero_vec = torch.zeros(self.dim, device=device)
        for t in range(num_events):
            prev_fwd = fwd_states[t - 1] if t > 0 else zero_vec
            next_bwd = bwd_states[t + 1] if (t + 1) < num_events else zero_vec
            cat_vec = torch.cat([fwd_states[t], bwd_states[t], inputs[t], prev_fwd, next_bwd], dim=-1)
            fused_states.append(self.state_fusion(cat_vec))

        all_states = torch.stack(fused_states, dim=0)
        return all_states, all_states[-1]


class WorldReasoningCore(nn.Module):
    """
    Cross-attends over past events and temporal states given a query,
    then predicts relational state transitions.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.state_k_proj = nn.Linear(dim, dim, bias=False)
        self.state_v_proj = nn.Linear(dim, dim, bias=False)
        self.temp_feat_mlp = nn.Sequential(
            nn.Linear(6, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.event_scorer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

        # State transition MLP combining query, attended events, attended states, and final state
        self.transition_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        query_vec: torch.Tensor,
        event_vecs: torch.Tensor,
        state_vecs: torch.Tensor,
        final_state: torch.Tensor,
        event_temporal_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_vec: (dim,)
        event_vecs: (T, dim)
        state_vecs: (T, dim)
        final_state: (dim,)
        event_temporal_features: optional (T, 6) per-event entity progression features
        Returns:
            reasoning_vector: (dim,)
            event_attention_weights: (T,)
        """
        T = event_vecs.size(0)
        device = event_vecs.device
        q = self.q_proj(query_vec.unsqueeze(0))  # (1, dim)
        k = self.k_proj(event_vecs)             # (T, dim)
        v = self.v_proj(event_vecs)             # (T, dim)

        sk = self.state_k_proj(state_vecs)      # (T, dim)
        sv = self.state_v_proj(state_vecs)      # (T, dim)

        if event_temporal_features is None:
            event_temporal_features = torch.zeros(T, 6, device=device)

        t_feat_vec = self.temp_feat_mlp(event_temporal_features)  # (T, dim)
        q_exp = query_vec.unsqueeze(0).expand(T, -1)
        mlp_scores = self.event_scorer(
            torch.cat([q_exp, t_feat_vec, q_exp * t_feat_vec], dim=-1)
        ).squeeze(-1)
        dot_scores = ((q @ k.t()).squeeze(0) + (q @ sk.t()).squeeze(0)) / math.sqrt(self.dim)
        scores = 0.25 * dot_scores + mlp_scores
        attn_weights = F.softmax(scores, dim=-1)              # (T,)

        c_event = (attn_weights.unsqueeze(-1) * v).sum(dim=0) # (dim,)
        c_state = (attn_weights.unsqueeze(-1) * sv).sum(dim=0)

        fused = torch.cat([query_vec, c_event, c_state, final_state], dim=-1)
        reasoning_vec = self.transition_mlp(fused)
        return reasoning_vec, attn_weights


# =====================================================================
# 5. DUAL POINTER-GENERATOR OUTPUT HEAD
# =====================================================================

class PointerGeneratorHead(nn.Module):
    """
    Generates probability distribution over:
      1. Closed vocabulary ('unknown', 'yes', 'no', 'before', 'after')
      2. Hierarchical event-attended pointer distribution over content candidates
         from observations: P_ptr(w) = sum_t event_attn[t] * P_tok(w | event_t).
    Gated by learned p_gen: P(w) = p_gen * P_closed(w) + (1 - p_gen) * P_pointer(w).
    """
    def __init__(self, dim: int = 64, closed_size: int = len(CLOSED_VOCABULARY)):
        super().__init__()
        self.dim = dim
        self.closed_size = closed_size
        self.rel_proj = nn.Sequential(
            nn.Linear(7, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.closed_linear = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.Linear(dim, closed_size),
        )
        self.gate_linear = nn.Sequential(
            nn.Linear(dim * 3, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        self.tok_scalar_proj = nn.Sequential(
            nn.Linear(4, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.token_scorer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
        self._vectorized_token_scoring = True

    def forward(
        self,
        reasoning_vec: torch.Tensor,
        candidate_embeddings: Optional[torch.Tensor],
        event_attn: torch.Tensor,
        candidate_event_indices: Optional[List[List[int]]],
        event_token_occ_vecs: Optional[List[torch.Tensor]] = None,
        event_token_occ_scalars: Optional[List[torch.Tensor]] = None,
        event_token_cand_ids: Optional[List[List[int]]] = None,
        global_rel_features: Optional[torch.Tensor] = None,
        query_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        reasoning_vec: (dim,)
        candidate_embeddings: (num_candidates, dim) or None
        event_attn: (T,)
        candidate_event_indices: mapping candidate index to list of event indices where it appears
        Returns:
            p_gen: scalar float tensor in [0, 1]
            closed_probs: (closed_size,)
            pointer_probs: (num_candidates,) or empty
        """
        device = reasoning_vec.device
        q_v = query_vec if query_vec is not None else reasoning_vec
        if global_rel_features is None:
            global_rel_features = torch.zeros(6, device=device)

        if candidate_embeddings is None or candidate_embeddings.numel() == 0:
            rel_7 = torch.cat([global_rel_features, torch.zeros(1, device=device)], dim=-1)
            rel_vec = self.rel_proj(rel_7)
            aug_r = torch.cat([q_v, rel_vec, q_v * rel_vec], dim=-1)
            closed_logits = self.closed_linear(aug_r)
            closed_probs = F.softmax(closed_logits, dim=-1)
            p_gen = torch.sigmoid(self.gate_linear(aug_r)).squeeze(0)
            return p_gen, closed_probs, torch.empty(0, device=device)

        num_cands = candidate_embeddings.size(0)
        sem_scores = (candidate_embeddings @ reasoning_vec) / math.sqrt(self.dim)  # (num_candidates,)

        ptr_query_overlap = torch.tensor(0.0, device=device)

        if (
            event_token_occ_vecs is not None
            and event_token_occ_scalars is not None
            and event_token_cand_ids is not None
        ):
            cand_prob_accum = torch.zeros(num_cands, device=device)
            overlap_accum = torch.tensor(0.0, device=device)
            if self._vectorized_token_scoring:
                # Score all occurrences in one network call, then keep softmax
                # within each event. This preserves hierarchical event weights.
                counts = [len(ids) for ids in event_token_cand_ids]
                active = [t for t, n in enumerate(counts) if n]
                if active:
                    flat_vecs = torch.cat([event_token_occ_vecs[t] for t in active], dim=0)
                    flat_scalars = torch.cat([event_token_occ_scalars[t] for t in active], dim=0)
                    flat_ids = torch.tensor(
                        [i for t in active for i in event_token_cand_ids[t]],
                        dtype=torch.long, device=device,
                    )
                    token_features = flat_vecs + self.tok_scalar_proj(flat_scalars)
                    query_features = q_v.unsqueeze(0).expand(token_features.size(0), -1)
                    logits = self.token_scorer(torch.cat([
                        token_features, query_features, token_features * query_features
                    ], dim=-1)).squeeze(-1) + 0.05 * sem_scores[flat_ids]
                    offset = 0
                    for t in active:
                        n = counts[t]
                        tok_probs = F.softmax(logits[offset:offset + n], dim=-1)
                        weighted = event_attn[t] * tok_probs
                        cand_prob_accum.scatter_add_(0, flat_ids[offset:offset + n], weighted)
                        overlap_accum = overlap_accum + event_attn[t] * (
                            tok_probs * flat_scalars[offset:offset + n, 0]
                        ).sum()
                        offset += n
            else:
                for t, (occ_vecs, occ_sc, c_ids) in enumerate(
                    zip(event_token_occ_vecs, event_token_occ_scalars, event_token_cand_ids)
                ):
                    if not c_ids or occ_vecs.numel() == 0:
                        continue
                    M = len(c_ids)
                    sc_vec = self.tok_scalar_proj(occ_sc)
                    tok_feat = occ_vecs + sc_vec
                    q_exp = q_v.unsqueeze(0).expand(M, -1)
                    tok_logits = self.token_scorer(
                        torch.cat([tok_feat, q_exp, tok_feat * q_exp], dim=-1)
                    ).squeeze(-1)
                    c_idx_tensor = torch.tensor(c_ids, dtype=torch.long, device=device)
                    tok_logits = tok_logits + 0.05 * sem_scores[c_idx_tensor]
                    tok_probs = F.softmax(tok_logits, dim=-1)
                    cand_prob_accum.scatter_add_(0, c_idx_tensor, event_attn[t] * tok_probs)
                    overlap_accum = overlap_accum + event_attn[t] * (
                        tok_probs * occ_sc[:, 0]
                    ).sum()
            pointer_probs = cand_prob_accum / cand_prob_accum.sum().clamp_min(1e-9)
            ptr_query_overlap = overlap_accum
        else:
            if candidate_event_indices is not None and event_attn.numel() > 0:
                ev_weights = []
                for ev_list in candidate_event_indices:
                    if ev_list:
                        idx_t = torch.tensor(ev_list, dtype=torch.long, device=device)
                        ev_weights.append(event_attn[idx_t].sum())
                    else:
                        ev_weights.append(torch.tensor(1e-6, device=device))
                ev_w_tensor = torch.stack(ev_weights, dim=0)
                sem_scores = sem_scores + torch.log(ev_w_tensor.clamp_min(1e-6))
            pointer_probs = F.softmax(sem_scores, dim=-1)

        rel_7 = torch.cat([global_rel_features, ptr_query_overlap.unsqueeze(0)], dim=-1)
        rel_vec = self.rel_proj(rel_7)
        aug_r = torch.cat([q_v, rel_vec, q_v * rel_vec], dim=-1)
        closed_logits = self.closed_linear(aug_r)
        closed_probs = F.softmax(closed_logits, dim=-1)
        p_gen = torch.sigmoid(self.gate_linear(aug_r)).squeeze(0)

        return p_gen, closed_probs, pointer_probs


# =====================================================================
# 6. COMPLETE SONARA NEURAL WORLD MODEL
# =====================================================================

@dataclass
class WorldPrediction:
    answer: str
    confidence: float
    uncertainty: float
    p_gen: float
    distribution: Dict[str, float]
    attended_events: List[float]


class NeuralWorldModel(nn.Module):
    """
    End-to-end trainable neural World Model for Sonara.
    """
    def __init__(
        self,
        dim: int = 64,
        vocab_size: int = 512,
        hash_buckets: int = 1024,
        max_events: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.embedding = CharSubwordEmbedding(vocab_size=vocab_size, hash_buckets=hash_buckets, dim=dim)
        self.temporal_encoding = TemporalEncoding(dim=dim, max_steps=max_events)
        self.event_encoder = EventEncoder(self.embedding, dim=dim)
        self.tracker = WorldStateTracker(dim=dim)
        self.reasoner = WorldReasoningCore(dim=dim)
        self.head = PointerGeneratorHead(dim=dim)
        # Cache only weight-independent tokenization and candidate positions,
        # never gradient-bearing neural states or authoritative memory.
        self._event_plan_cache: OrderedDict = OrderedDict()
        self._event_plan_capacity = 16
        self._cache_event_plans = True

    def _prepare_events(
        self, events: Sequence[str]
    ) -> Tuple[List[List[str]], List[str], List[List[int]]]:
        """Bounded cache of static event text features (never neural tensors)."""
        key = tuple(events)
        if self._cache_event_plans and key in self._event_plan_cache:
            self._event_plan_cache.move_to_end(key)
            return self._event_plan_cache[key]
        tokens = [tokenize(event) for event in events]
        candidates: List[str] = []
        hits: List[List[int]] = []
        indexed: Dict[str, int] = {}
        for event_index, event_tokens in enumerate(tokens):
            for token in dict.fromkeys(event_tokens):
                if not is_content_token(token):
                    continue
                if token not in indexed:
                    indexed[token] = len(candidates)
                    candidates.append(token)
                    hits.append([])
                hits[indexed[token]].append(event_index)
        prepared = tokens, candidates, hits
        if self._cache_event_plans:
            self._event_plan_cache[key] = prepared
            if len(self._event_plan_cache) > self._event_plan_capacity:
                self._event_plan_cache.popitem(last=False)
        return prepared

    def _encode_episode_and_query(
        self,
        events: List[str],
        query: str,
        word_to_id: Dict[str, int],
        device: torch.device,
    ) -> Tuple[
        List[str],
        List[List[int]],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[List[int]]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Shared differentiable forward backbone for both inference and loss computation.
        """
        query_tokens = tokenize(query)
        tokenized_events, candidates, candidate_indices = self._prepare_events(events)
        cand_to_idx = {c: idx for idx, c in enumerate(candidates)}

        if not events:
            q_vec = self.event_encoder(query_tokens, word_to_id, device)
            empty_state = torch.zeros(self.dim, device=device)
            reasoning_vec, _ = self.reasoner(q_vec, empty_state.unsqueeze(0), empty_state.unsqueeze(0), empty_state)
            return (
                candidates,
                candidate_indices,
                reasoning_vec,
                None,
                torch.zeros(1, device=device),
                None,
                None,
                None,
                torch.zeros(6, device=device),
                q_vec,
            )

        T = len(events)
        Lq = len(query_tokens)
        step_indices = torch.arange(T, dtype=torch.long, device=device)
        t_vecs = self.temporal_encoding(step_indices)  # (T, dim)

        # 1. Compute morphological identity embeddings for query and events
        q_id_vecs = self.embedding.embed_identity(query_tokens, device)  # (L_q, dim)
        ev_id_vecs_list = [self.embedding.embed_identity(ev_toks, device) for ev_toks in tokenized_events]

        # 2. Compute smooth differentiable identity similarity matrices M_t: (L_t, L_q)
        sim_matrices: List[torch.Tensor] = []
        m_tj_list: List[torch.Tensor] = []
        for t in range(T):
            ev_id = ev_id_vecs_list[t]  # (L_t, dim)
            raw_sim = ev_id @ q_id_vecs.t()  # (L_t, L_q)
            active_mask = ((ev_id.norm(dim=-1, keepdim=True) > 0.1) & (q_id_vecs.norm(dim=-1).unsqueeze(0) > 0.1)).float()
            M_t = torch.sigmoid(14.0 * (raw_sim - 0.55)) * active_mask
            sim_matrices.append(M_t)
            m_tj_list.append(M_t.max(dim=0).values if M_t.size(0) > 0 else torch.zeros(Lq, device=device))

        m_tj = torch.stack(m_tj_list, dim=0)  # (T, L_q)
        m_sum_t = m_tj.sum(dim=0, keepdim=True).clamp_min(1e-6)  # (1, L_q)
        spec_tj = m_tj / m_sum_t  # (T, L_q)

        # Ground each query token with the temporal step embedding of the event(s) it matched
        q_time_vecs = spec_tj.t() @ t_vecs  # (L_q, dim)
        q_exist = m_tj.max(dim=0).values  # (L_q,)
        rel_steps = torch.linspace(0.0, 1.0, steps=T, device=device).unsqueeze(1) if T > 1 else torch.zeros(T, 1, device=device)
        q_rel_tau = (spec_tj * rel_steps).sum(dim=0)  # (L_q,)
        q_uniq = q_exist * spec_tj.max(dim=0).values  # (L_q,)
        q_pos = torch.linspace(0.0, 1.0, steps=Lq, device=device) if Lq > 1 else torch.zeros(Lq, device=device)
        q_scalars = torch.stack([q_exist, q_rel_tau, q_uniq, q_pos], dim=-1)  # (L_q, 4)

        q_vec, q_tok_h = self.event_encoder(
            query_tokens,
            word_to_id,
            device,
            return_token_states=True,
            aligned_vecs=q_time_vecs,
            match_scalars=q_scalars,
        )

        # Compute left-vs-right matched temporal progression across query tokens (for event order)
        left_w = (1.0 - q_pos) * (q_uniq ** 2)
        right_w = q_pos * (q_uniq ** 2)
        tau_left = (left_w * q_rel_tau).sum() / left_w.sum().clamp_min(1e-6)
        tau_right = (right_w * q_rel_tau).sum() / right_w.sum().clamp_min(1e-6)
        delta_tau = tau_right - tau_left

        # Separate non-digit entity matches from timestamp/anchor matches in the query
        non_digit_mask = torch.tensor(
            [1.0 if (not tok.isdigit()) else 0.0 for tok in query_tokens],
            dtype=torch.float32,
            device=device,
        )
        digit_mask = 1.0 - non_digit_mask
        ev_raw_ent_match = (m_tj * non_digit_mask.unsqueeze(0)).max(dim=-1).values  # (T,)
        ev_anchor_match = (m_tj * spec_tj * digit_mask.unsqueeze(0)).max(dim=-1).values  # (T,)
        has_time_anchor = (q_exist * digit_mask).max()

        # 3. Encode each event conditioned on query-token matches and temporal encodings
        encoded_events = []
        tok_h_list: List[torch.Tensor] = []
        ev_scalars_list: List[torch.Tensor] = []
        ev_has_complement_list: List[torch.Tensor] = []
        final_end_entity_match = torch.tensor(0.0, device=device)

        for t, ev_toks in enumerate(tokenized_events):
            M_t = sim_matrices[t]  # (L_t, L_q)
            s_max = M_t.max(dim=-1).values  # (L_t,)
            spec_weights = M_t * spec_tj[t].unsqueeze(0)  # (L_t, L_q)
            s_uniq = spec_weights.max(dim=-1).values  # (L_t,)
            is_struct_or_digit = torch.tensor(
                [1.0 if (tok.lower() in STRUCTURAL_WORDS or tok.isdigit()) else 0.0 for tok in ev_toks],
                dtype=torch.float32,
                device=device,
            )
            tok_pos_scalar = (
                torch.linspace(0.0, 1.0, steps=len(ev_toks), device=device)
                if len(ev_toks) > 1
                else torch.zeros_like(s_max)
            )
            ev_scalars = torch.stack([s_max, s_uniq, is_struct_or_digit, tok_pos_scalar], dim=-1)  # (L_t, 4)

            ent_positions = [
                idx for idx, tok in enumerate(ev_toks)
                if is_content_token(tok) and tok.lower() not in STRUCTURAL_WORDS and not tok.isdigit()
            ]
            if ent_positions:
                last_ent_match = s_max[ent_positions[-1]]
                if t == T - 1:
                    final_end_entity_match = last_ent_match
                # An event predicates a state transition/location for the queried entity if there are
                # at least 3 entity slots (actor, object, location) or the final entity is a complement
                has_comp = 1.0 if len(ent_positions) >= 3 else float(1.0 - last_ent_match.detach().item())
                ev_has_complement_list.append(torch.tensor(has_comp, device=device))
            else:
                ev_has_complement_list.append(torch.tensor(0.0, device=device))

            m_row_sum = M_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            u_aligned = (M_t @ q_tok_h) / m_row_sum  # (L_t, dim)

            ev_vec, tok_h = self.event_encoder(
                ev_toks,
                word_to_id,
                device,
                return_token_states=True,
                aligned_vecs=u_aligned,
                match_scalars=ev_scalars,
            )
            encoded_events.append(ev_vec + t_vecs[t])
            tok_h_list.append(tok_h)
            ev_scalars_list.append(ev_scalars)

        ev_has_comp_tensor = torch.stack(ev_has_complement_list, dim=0)  # (T,)
        ev_ent_match = ev_raw_ent_match * ev_has_comp_tensor  # (T,)

        # Sequential progression features along t = 0..T-1
        ev_temp_feats = []
        ev_local_feats = []
        for t in range(T):
            ent_here = ev_ent_match[t]
            ent_after = ev_ent_match[t + 1:].sum() if (t + 1) < T else torch.tensor(0.0, device=device)
            ent_before = ev_ent_match[:t].sum() if t > 0 else torch.tensor(0.0, device=device)
            is_latest_ent = ent_here * torch.exp(-3.0 * ent_after)
            is_earliest_ent = ent_here * torch.exp(-3.0 * ent_before)
            next_anchor = ev_anchor_match[t + 1] if (t + 1) < T else torch.tensor(0.0, device=device)
            is_before_anchor = ent_here * next_anchor
            rel_t = torch.tensor(float(t) / max(T - 1, 1), device=device)
            ev_temp_feats.append(
                torch.stack([ent_here, is_latest_ent, is_earliest_ent, is_before_anchor, has_time_anchor, rel_t])
            )
            ev_local_feats.append(
                torch.stack([ent_here, ev_has_comp_tensor[t], ev_anchor_match[t], has_time_anchor, rel_t])
            )
        event_temporal_features = torch.stack(ev_temp_feats, dim=0)  # (T, 6)
        event_local_features = torch.stack(ev_local_feats, dim=0)    # (T, 5)

        event_tensor = torch.stack(encoded_events, dim=0)  # (T, dim)
        state_tensor, final_state = self.tracker(event_tensor, event_local_features=event_local_features)
        reasoning_vec, event_attn = self.reasoner(
            q_vec,
            event_tensor,
            state_tensor,
            final_state,
            event_temporal_features=event_temporal_features,
        )

        # 4. Build per-event candidate occurrence states and aggregate candidate embeddings
        token_state_map = defaultdict(list)
        event_occ_vecs: List[torch.Tensor] = []
        event_occ_scalars: List[torch.Tensor] = []
        event_occ_cids: List[List[int]] = []

        for t, ev_toks in enumerate(tokenized_events):
            tok_h = tok_h_list[t]
            ev_sc = ev_scalars_list[t]
            occ_vecs_t = []
            occ_sc_t = []
            occ_cids_t = []
            for i, tok in enumerate(ev_toks):
                t_low = tok.lower()
                token_state_map[t_low].append(event_attn[t] * tok_h[i])
                if t_low in cand_to_idx:
                    occ_vecs_t.append(tok_h[i])
                    occ_sc_t.append(ev_sc[i])
                    occ_cids_t.append(cand_to_idx[t_low])
            if occ_vecs_t:
                event_occ_vecs.append(torch.stack(occ_vecs_t, dim=0))
                event_occ_scalars.append(torch.stack(occ_sc_t, dim=0))
            else:
                event_occ_vecs.append(torch.empty(0, self.dim, device=device))
                event_occ_scalars.append(torch.empty(0, 4, device=device))
            event_occ_cids.append(occ_cids_t)

        cand_vecs = None
        if candidates:
            c_embeddings = []
            for cand in candidates:
                cand_static = self.embedding.embed_token(cand, word_to_id.get(cand, UNK_ID), device)
                if cand in token_state_map:
                    cand_ctx = torch.stack(token_state_map[cand], dim=0).sum(dim=0)
                    c_embeddings.append(F.layer_norm(cand_ctx + 0.1 * cand_static, [self.dim]))
                else:
                    c_embeddings.append(cand_static)
            cand_vecs = torch.stack(c_embeddings, dim=0)

        global_rel_features = torch.stack([
            q_exist.max(),
            (q_exist * non_digit_mask).max(),
            tau_left,
            tau_right,
            delta_tau,
            final_end_entity_match,
        ])

        return (
            candidates,
            candidate_indices,
            reasoning_vec,
            cand_vecs,
            event_attn,
            event_occ_vecs,
            event_occ_scalars,
            event_occ_cids,
            global_rel_features,
            q_vec,
        )

    def forward(
        self,
        events: List[str],
        query: str,
        word_to_id: Dict[str, int],
        device: torch.device,
    ) -> Tuple[Dict[str, float], WorldPrediction]:
        """
        Infers predictions for a query given a sequence of observation events.
        """
        (
            candidates,
            candidate_indices,
            reasoning_vec,
            cand_vecs,
            event_attn,
            event_occ_vecs,
            event_occ_scalars,
            event_occ_cids,
            global_rel_features,
            q_vec,
        ) = self._encode_episode_and_query(events, query, word_to_id, device)

        if not events:
            p_gen, closed_probs, _ = self.head(reasoning_vec, None, event_attn, None, query_vec=q_vec)
            dist = {ID_TO_CLOSED[i]: float(closed_probs[i]) for i in range(len(CLOSED_VOCABULARY))}
            best_word = max(dist, key=dist.get)
            entropy = -sum(p * math.log(max(p, 1e-9)) for p in dist.values())
            return dist, WorldPrediction(
                answer=best_word,
                confidence=dist[best_word],
                uncertainty=round(entropy, 4),
                p_gen=float(p_gen),
                distribution=dist,
                attended_events=[],
            )

        p_gen, closed_probs, ptr_probs = self.head(
            reasoning_vec,
            cand_vecs,
            event_attn,
            candidate_indices,
            event_token_occ_vecs=event_occ_vecs,
            event_token_occ_scalars=event_occ_scalars,
            event_token_cand_ids=event_occ_cids,
            global_rel_features=global_rel_features,
            query_vec=q_vec,
        )

        # Combine probability distribution
        combined_dist: Dict[str, float] = {}
        for idx, word in ID_TO_CLOSED.items():
            if word not in ("<pad>", "<unk>"):
                combined_dist[word] = float(p_gen * closed_probs[idx])

        if cand_vecs is not None and ptr_probs.numel() > 0:
            for idx, cand in enumerate(candidates):
                p_ptr = float((1.0 - p_gen) * ptr_probs[idx])
                combined_dist[cand] = combined_dist.get(cand, 0.0) + p_ptr

        total_p = sum(combined_dist.values()) or 1.0
        norm_dist = {w: p / total_p for w, p in combined_dist.items()}
        best_word = max(norm_dist, key=norm_dist.get)
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in norm_dist.values())

        prediction = WorldPrediction(
            answer=best_word,
            confidence=norm_dist[best_word],
            uncertainty=round(entropy, 4),
            p_gen=float(p_gen),
            distribution=norm_dist,
            attended_events=[float(w) for w in event_attn.detach().cpu()],
        )
        return norm_dist, prediction

    def compute_loss(
        self,
        events: List[str],
        query: str,
        target_answer: str,
        word_to_id: Dict[str, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Differentiable cross-entropy loss against a target answer.
        """
        (
            candidates,
            candidate_indices,
            reasoning_vec,
            cand_vecs,
            event_attn,
            event_occ_vecs,
            event_occ_scalars,
            event_occ_cids,
            global_rel_features,
            q_vec,
        ) = self._encode_episode_and_query(events, query, word_to_id, device)

        target_norm = target_answer.lower().strip()

        if not events:
            p_gen, closed_probs, _ = self.head(reasoning_vec, None, event_attn, None, query_vec=q_vec)
            if target_norm in CLOSED_TO_ID:
                target_idx = CLOSED_TO_ID[target_norm]
                return -torch.log(closed_probs[target_idx].clamp_min(1e-9))
            return -torch.log(closed_probs[UNKNOWN_ID].clamp_min(1e-9))

        p_gen, closed_probs, ptr_probs = self.head(
            reasoning_vec,
            cand_vecs,
            event_attn,
            candidate_indices,
            event_token_occ_vecs=event_occ_vecs,
            event_token_occ_scalars=event_occ_scalars,
            event_token_cand_ids=event_occ_cids,
            global_rel_features=global_rel_features,
            query_vec=q_vec,
        )

        loss_components = []
        if target_norm in CLOSED_TO_ID:
            idx = CLOSED_TO_ID[target_norm]
            p_word = p_gen * closed_probs[idx]
            loss_components.append(p_word)

        if cand_vecs is not None and target_norm in candidates:
            c_idx = candidates.index(target_norm)
            p_ptr = (1.0 - p_gen) * ptr_probs[c_idx]
            loss_components.append(p_ptr)

        if not loss_components:
            p_word = p_gen * closed_probs[UNKNOWN_ID]
            loss_components.append(p_word)

        total_prob = sum(loss_components).clamp_min(1e-9)
        return -torch.log(total_prob)


    def compute_action_contrast_loss(
        self,
        events: List[str],
        verified_pairs: Sequence[Tuple[str, str]],
        word_to_id: Dict[str, int],
        device: torch.device,
        *,
        contrast_weight: float = 1.0,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Learn the *relative* consequences of two independently verified actions.

        The caller must supply two observed (action-query, outcome) pairs from
        the same pre-intervention context. This loss never invents an outcome
        for an untried action. With contrast_weight=0 it is the matched
        supervised-only control, trained on the exact same two observations.
        """
        (query_a, outcome_a), (query_b, outcome_b) = verified_pairs
        loss_aa = self.compute_loss(events, query_a, outcome_a, word_to_id, device)
        loss_bb = self.compute_loss(events, query_b, outcome_b, word_to_id, device)
        supervised = (loss_aa + loss_bb) / 2
        if contrast_weight == 0:
            return supervised

        # Negative examples are action/query mismatches, not fabricated events.
        # softplus(margin + NLL(correct) - NLL(incorrect action)) rewards
        # the observed action for each outcome over its verified alternative.
        loss_ba = self.compute_loss(events, query_b, outcome_a, word_to_id, device)
        loss_ab = self.compute_loss(events, query_a, outcome_b, word_to_id, device)
        separation = (
            F.softplus(margin + loss_aa - loss_ba)
            + F.softplus(margin + loss_bb - loss_ab)
        ) / 2
        return supervised + contrast_weight * separation


# =====================================================================
# 7. HIGH-LEVEL COGNITIVE OWNER & INTERACTION LAYER
# =====================================================================

class SonaraWorldModel:
    """
    Cognitive owner for Sonara's neural world reasoning.
    Connects to canonical episodic memory, maintains active session experiences,
    performs forward predictions before outcomes, and bounded learning updates.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        dim: int = 64,
        lr: float = 0.002,
    ):
        self.device = self._select_device(device)
        self.dim = dim
        self.lr = lr
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.word_to_id: Dict[str, int] = dict(CLOSED_TO_ID)
        self.id_to_word: Dict[int, str] = dict(ID_TO_CLOSED)
        self.model = NeuralWorldModel(dim=dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.session_events: List[str] = []
        self.active_expectations: List[Dict[str, Any]] = []
        self.replay_buckets: Dict[str, List[Tuple[List[str], str, str]]] = defaultdict(list)
        self._max_replay_buckets = 64
        self._session_event_terms: List[frozenset[str]] = []
        self._sparse_session_activation = True
        self._max_active_events = 24
        self.trained_examples_count: int = 0
        self.enabled: bool = False

        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_checkpoint(self.checkpoint_path)

    def _select_device(self, val: str) -> torch.device:
        if val != "auto":
            return torch.device(val)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def register_vocabulary(self, words: Sequence[str]) -> None:
        """Register base vocabulary tokens."""
        for w in words:
            w_lower = w.lower().strip()
            if w_lower and w_lower not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[w_lower] = new_id
                self.id_to_word[new_id] = w_lower

    def observe(self, observation: str) -> None:
        """Ingest a new experience into active session memory."""
        cleaned = observation.strip()
        if cleaned:
            self.session_events.append(cleaned)
            terms = frozenset(tokenize(cleaned))
            self._session_event_terms.append(terms)
            self.register_vocabulary(terms)

    def clear_session(self) -> None:
        self.session_events.clear()
        self._session_event_terms.clear()
        self.active_expectations.clear()

    def _activate_session_events(self, question: str) -> List[str]:
        """Conservative bounded activation from the existing session evidence.

        Skip only clearly unrelated events. When temporal continuity would
        overflow the budget, fall back to full evidence rather than fabricate
        a confident answer from a truncated causal sequence. No memory mutation.
        """
        history = self.session_events
        budget = self._max_active_events
        if not self._sparse_session_activation or len(history) <= budget:
            return history
        cues = set(tokenize(question)) - STOPWORDS - STRUCTURAL_WORDS
        if not cues:
            return history
        matches = [
            i for i, terms in enumerate(self._session_event_terms)
            if cues.intersection(terms)
        ]
        if not matches:
            return history
        selected = set(range(max(0, len(history) - 2), len(history)))
        for i in matches:
            selected.update(range(max(0, i - 1), min(len(history), i + 2)))
            if len(selected) > budget:
                # Ambiguous or unusually long dependency: do not prune.
                return history
        return [history[i] for i in sorted(selected)]

    def predict(self, question: str, events: Optional[List[str]] = None) -> WorldPrediction:
        """Predict from evidence; selectively activate only long live histories."""
        self.model.eval()
        evs = events if events is not None else self._activate_session_events(question)
        with torch.no_grad():
            _, pred = self.model(evs, question, self.word_to_id, self.device)
        return pred

    def predict_state_transition(self, entity: str, action_event: str) -> WorldPrediction:
        """Form a pre-outcome prediction before an action's result is verified."""
        hypothetical_events = self.session_events + [action_event]
        query = f"where is the {entity}?"
        return self.predict(query, events=hypothetical_events)

    def _bucket_key(self, events: List[str], question: str, verified_answer: str) -> str:
        q_toks = tokenize(question)
        q_head = q_toks[0] if q_toks else "q"
        ans = verified_answer.lower().strip()
        if ans in CLOSED_TO_ID:
            return f"{q_head}_{ans}"
        has_corr = any("correction" in ev.lower() for ev in events)
        has_before = "before" in q_toks
        return f"{q_head}_ptr_{int(has_corr)}_{int(has_before)}"

    def learn_step(
        self,
        events: List[str],
        question: str,
        verified_answer: str,
        clip_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """Verified supervision with bounded, ephemeral cross-template replay."""
        self.model.train()
        self.register_vocabulary(tokenize(" ".join(events) + " " + question + " " + verified_answer))
        primary_loss = self.model.compute_loss(events, question, verified_answer, self.word_to_id, self.device)
        key = self._bucket_key(events, question, verified_answer)
        losses = [primary_loss]
        for old_key, examples in self.replay_buckets.items():
            if old_key != key and examples:
                replay_events, replay_query, replay_answer = examples[-1]
                losses.append(self.model.compute_loss(
                    replay_events, replay_query, replay_answer, self.word_to_id, self.device
                ))
        metrics = self._apply_verified_loss(torch.stack(losses).mean(), clip_norm)
        bucket = self.replay_buckets[key]
        bucket.append((list(events), question, verified_answer))
        if len(bucket) > 8:
            bucket.pop(0)
        # Replay is derived from verified feedback and remains bounded.
        if len(self.replay_buckets) > self._max_replay_buckets:
            oldest = next(iter(self.replay_buckets))
            del self.replay_buckets[oldest]
        metrics["loss"] = round(float(primary_loss.detach().item()), 4)
        metrics["replay_examples"] = len(losses) - 1
        return metrics

    def _apply_verified_loss(self, loss: torch.Tensor, clip_norm: float) -> Dict[str, Any]:
        """Shared gradient update for ordinary and action-contrast learning."""
        before_weights = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        self.optimizer.step()

        # Compute L1 parameter change
        param_delta = sum(
            float((p.detach() - before_weights[name]).abs().sum().item())
            for name, p in self.model.named_parameters()
        )
        self.trained_examples_count += 1

        return {
            "loss": round(float(loss.item()), 4),
            "parameter_delta": round(param_delta, 6),
            "trained_count": self.trained_examples_count,
        }


    def learn_action_contrast(
        self,
        events: List[str],
        verified_pairs: Sequence[Tuple[str, str]],
        *,
        steps: int = 8,
        contrast_weight: float = 1.0,
        margin: float = 0.5,
        clip_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """Replay two *verified* alternative interventions to learn their distinction.

        This is an existing-world-model gradient path, not a new selector or
        persistent store. Source verification belongs to the caller; a
        single observed action cannot authorize a counterfactual label.
        """
        pairs = list(verified_pairs)
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError("Two verified (action query, outcome) pairs required")
        (query_a, outcome_a), (query_b, outcome_b) = pairs
        if not query_a.strip() or not query_b.strip() or query_a == query_b:
            raise ValueError("Action queries must be distinct and nonempty")
        if not outcome_a.strip() or not outcome_b.strip() or outcome_a == outcome_b:
            raise ValueError("Contrast requires two distinct verified outcomes")
        if not isinstance(steps, int) or not 1 <= steps <= 32:
            raise ValueError("Replay steps must be an integer from 1 to 32")
        if not 0 <= contrast_weight <= 4 or not 0 <= margin <= 4:
            raise ValueError("Invalid contrast weight or ranking margin")
        candidates = set(extract_content_candidates(events)) | set(CLOSED_TO_ID)
        if outcome_a.lower().strip() not in candidates or outcome_b.lower().strip() not in candidates:
            raise ValueError("Both outcomes must be representable in pre-action context")
        self.register_vocabulary(tokenize(" ".join(events + [query_a, query_b, outcome_a, outcome_b])))
        self.model.train()
        total_delta = 0.0
        first_loss = None
        last = None
        for _ in range(steps):
            loss = self.model.compute_action_contrast_loss(
                events, pairs, self.word_to_id, self.device,
                contrast_weight=contrast_weight, margin=margin,
            )
            if first_loss is None:
                first_loss = round(float(loss.detach().item()), 4)
            last = self._apply_verified_loss(loss, clip_norm)
            total_delta += last["parameter_delta"]
        return {
            "initial_loss": first_loss,
            "loss": last["loss"],
            "parameter_delta": round(total_delta, 6),
            "gradient_steps": steps,
            "verified_pairs": 2,
            "contrast_weight": contrast_weight,
            "trained_count": self.trained_examples_count,
        }

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        target = Path(path) if path else self.checkpoint_path
        if not target:
            raise ValueError("No checkpoint path provided")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = str(target) + ".tmp"

        payload = {
            "version": 2,
            "architecture": {"dim": self.dim, "family": "temporal_hierarchical"},
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "trained_examples_count": self.trained_examples_count,
            "replay_buckets": dict(self.replay_buckets),
            "state_dict": self.model.state_dict(),
        }
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return target

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint not found: {target}")
        try:
            payload = torch.load(target, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(target, map_location=self.device)

        # Check first: a former model's layers can have different shapes even
        # when its checkpoint also claims version 1. Never partially mutate
        # a live owner or overwrite an incompatible identity-bearing model.
        dims = payload.get("architecture", {}).get("dim")
        old_state = payload["state_dict"]
        current = self.model.state_dict()
        if dims != self.dim or old_state.keys() != current.keys() or any(
            tuple(old_state[key].shape) != tuple(current[key].shape)
            for key in current
        ):
            raise ValueError(
                "Incompatible world-model checkpoint architecture; original weights "
                "must be preserved and explicitly migrated or retrained"
            )
        if payload.get("version") not in (1, 2):
            raise ValueError("Unsupported world-model checkpoint version")
        restored = payload.get("replay_buckets", {})
        if not isinstance(restored, dict) or len(restored) > self._max_replay_buckets:
            raise ValueError("Invalid or unbounded checkpoint replay data")
        for key, examples in restored.items():
            if not isinstance(key, str) or not isinstance(examples, list) or len(examples) > 8:
                raise ValueError("Malformed checkpoint replay bucket")
            for example in examples:
                if (
                    not isinstance(example, (tuple, list)) or len(example) != 3
                    or not isinstance(example[0], list)
                    or not all(isinstance(part, str) for part in example[0])
                    or not all(isinstance(part, str) for part in example[1:])
                ):
                    raise ValueError("Malformed verified replay example")
        self.model.load_state_dict(old_state)
        self.word_to_id = payload.get("word_to_id", self.word_to_id)
        self.id_to_word = payload.get("id_to_word", self.id_to_word)
        self.trained_examples_count = payload.get("trained_examples_count", 0)
        self.replay_buckets = defaultdict(list, restored)
        self.model.eval()
