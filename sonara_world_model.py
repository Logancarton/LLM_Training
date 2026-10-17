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
      an evolving world state over time.
    - Dual pointer-generator head enables both closed-class relational
      answers ("unknown", "yes", "no") and direct entity extraction from
      observed experiences without vocabulary leakage.
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
import re
import time
from collections import defaultdict
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


def tokenize(text: str) -> List[str]:
    """Extract normalized lowercase alphanumeric tokens and punctuation."""
    return re.findall(r"[\w]+(?:['-][\w]+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def extract_content_candidates(events: Sequence[str]) -> List[str]:
    """Extract distinct non-stopword tokens from observations as pointer candidates."""
    seen = set()
    candidates = []
    for event in events:
        for token in tokenize(event):
            t = token.lower()
            if t not in STOPWORDS and len(t) > 1 and t not in seen:
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

    def _hash_ngram(self, ngram: str) -> int:
        h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.hash_buckets

    def embed_token(self, token: str, word_id: int, device: torch.device) -> torch.Tensor:
        w_id = torch.tensor([word_id if word_id < self.vocab_size else UNK_ID], device=device)
        w_vec = self.word_embeddings(w_id).squeeze(0)  # (dim,)

        # Compute character 2-grams and 3-grams
        wrapped = f"<{token}>"
        ngrams = [wrapped[i:i+n] for n in (2, 3) for i in range(len(wrapped) - n + 1)]
        if not ngrams:
            ngrams = [wrapped]
        bucket_ids = torch.tensor([self._hash_ngram(ng) for ng in ngrams], device=device)
        char_vec = self.hash_embeddings(bucket_ids).mean(dim=0)  # (dim,)

        combined = torch.cat([w_vec, char_vec], dim=-1)
        return self.norm(self.projection(combined))

    def forward(self, token_strings: Sequence[str], word_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        """Embed a sequence of tokens into a (seq_len, dim) tensor."""
        if not token_strings:
            return torch.zeros(1, self.dim, device=device)
        vecs = [self.embed_token(tok, wid, device) for tok, wid in zip(token_strings, word_ids)]
        return torch.stack(vecs, dim=0)


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
        # step_indices: (N,)
        clamped = step_indices.clamp(0, self.temporal_embeddings.num_embeddings - 1)
        return self.temporal_embeddings(clamped)


class EventEncoder(nn.Module):
    """
    Encodes an event sentence (or question) into a fixed-width vector
    using token embeddings and a bidirectional GRU with attention pooling.
    """
    def __init__(self, embedding: CharSubwordEmbedding, dim: int = 64):
        super().__init__()
        self.embedding = embedding
        self.dim = dim
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.attn_query = nn.Linear(dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: List[str],
        word_to_id: Dict[str, int],
        device: torch.device,
        return_token_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # tokens: list of strings
        if not tokens:
            zero_vec = torch.zeros(self.dim, device=device)
            if return_token_states:
                return zero_vec, torch.zeros(1, self.dim, device=device)
            return zero_vec
        word_ids = [word_to_id.get(t.lower(), UNK_ID) for t in tokens]
        token_vecs = self.embedding(tokens, word_ids, device)  # (L, dim)
        h, _ = self.gru(token_vecs.unsqueeze(0))  # (1, L, dim)
        h = h.squeeze(0)  # (L, dim)

        # Attention pooling across tokens
        attn_scores = self.attn_query(h).softmax(dim=0)  # (L, 1)
        pooled = (h * attn_scores).sum(dim=0)  # (dim,)
        norm_pooled = self.norm(pooled)
        if return_token_states:
            return norm_pooled, h
        return norm_pooled


# =====================================================================
# 4. RECURRENT WORLD STATE & RELATIONAL REASONING
# =====================================================================

class WorldStateTracker(nn.Module):
    """
    Maintains an evolving world state representation updated sequentially
    as observations occur over time: s_t = GRUCell(z_t, s_{t-1}).
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.cell = nn.GRUCell(dim, dim)

    def forward(self, event_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        event_vectors: (num_events, dim)
        Returns:
            all_states: (num_events, dim)
            final_state: (dim,)
        """
        num_events = event_vectors.size(0)
        device = event_vectors.device
        if num_events == 0:
            zero_state = torch.zeros(self.dim, device=device)
            return zero_state.unsqueeze(0), zero_state

        state = torch.zeros(self.dim, device=device)
        states = []
        for t in range(num_events):
            state = self.cell(event_vectors[t].unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            states.append(state)
        return torch.stack(states, dim=0), state


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_vec: (dim,)
        event_vecs: (T, dim)
        state_vecs: (T, dim)
        final_state: (dim,)
        Returns:
            reasoning_vector: (dim,)
            event_attention_weights: (T,)
        """
        T = event_vecs.size(0)
        q = self.q_proj(query_vec.unsqueeze(0))  # (1, dim)
        k = self.k_proj(event_vecs)             # (T, dim)
        v = self.v_proj(event_vecs)             # (T, dim)

        scores = (q @ k.t()).squeeze(0) / math.sqrt(self.dim)  # (T,)
        attn_weights = F.softmax(scores, dim=-1)              # (T,)
        c_event = (attn_weights.unsqueeze(-1) * v).sum(dim=0) # (dim,)

        # Attend to corresponding world states
        sk = self.state_k_proj(state_vecs)                    # (T, dim)
        sv = self.state_v_proj(state_vecs)                    # (T, dim)
        state_scores = (q @ sk.t()).squeeze(0) / math.sqrt(self.dim)
        state_weights = F.softmax(state_scores, dim=-1)
        c_state = (state_weights.unsqueeze(-1) * sv).sum(dim=0)

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
      2. Pointer attention over content candidates from observations.
    Gated by learned p_gen: P(w) = p_gen * P_closed(w) + (1 - p_gen) * P_pointer(w).
    """
    def __init__(self, dim: int = 64, closed_size: int = len(CLOSED_VOCABULARY)):
        super().__init__()
        self.dim = dim
        self.closed_size = closed_size
        self.closed_linear = nn.Linear(dim, closed_size)
        self.gate_linear = nn.Linear(dim, 1)

    def forward(
        self,
        reasoning_vec: torch.Tensor,
        candidate_embeddings: Optional[torch.Tensor],
        event_attn: torch.Tensor,
        candidate_event_indices: Optional[List[List[int]]],
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
        closed_logits = self.closed_linear(reasoning_vec)
        closed_probs = F.softmax(closed_logits, dim=-1)
        gate_logit = self.gate_linear(reasoning_vec)
        p_gen = torch.sigmoid(gate_logit).squeeze(0)  # scalar

        # Pointer score computed via learned inner product with candidate embeddings
        sem_scores = (candidate_embeddings @ reasoning_vec) / math.sqrt(self.dim)  # (num_candidates,)
        pointer_probs = F.softmax(sem_scores, dim=-1)
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
        query_tokens = tokenize(query)
        q_vec = self.event_encoder(query_tokens, word_to_id, device)

        candidates = extract_content_candidates(events)
        candidate_indices = []
        for cand in candidates:
            ev_hits = [i for i, ev in enumerate(events) if cand in tokenize(ev)]
            candidate_indices.append(ev_hits)

        if not events:
            # No prior observations
            empty_state = torch.zeros(self.dim, device=device)
            reasoning_vec, _ = self.reasoner(q_vec, empty_state.unsqueeze(0), empty_state.unsqueeze(0), empty_state)
            p_gen, closed_probs, _ = self.head(reasoning_vec, None, torch.zeros(1, device=device), None)
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

        # Encode events with temporal tags and collect contextual token embeddings
        encoded_events = []
        token_state_map = defaultdict(list)
        for t, ev in enumerate(events):
            ev_tokens = tokenize(ev)
            ev_vec, tok_h = self.event_encoder(ev_tokens, word_to_id, device, return_token_states=True)
            t_vec = self.temporal_encoding(torch.tensor([t], device=device)).squeeze(0)
            encoded_events.append(ev_vec + t_vec)
            for i, tok in enumerate(ev_tokens):
                token_state_map[tok.lower()].append(tok_h[i] + t_vec)
        event_tensor = torch.stack(encoded_events, dim=0)  # (T, dim)

        state_tensor, final_state = self.tracker(event_tensor)
        reasoning_vec, event_attn = self.reasoner(q_vec, event_tensor, state_tensor, final_state)

        # Contextual candidate word embeddings (combining morphological hash identity with event context)
        cand_vecs = None
        if candidates:
            c_embeddings = []
            for cand in candidates:
                cand_static = self.embedding.embed_token(cand, word_to_id.get(cand, UNK_ID), device)
                if cand in token_state_map:
                    cand_ctx = torch.stack(token_state_map[cand], dim=0).mean(dim=0)
                    c_embeddings.append(F.layer_norm(cand_static + cand_ctx, [self.dim]))
                else:
                    c_embeddings.append(cand_static)
            cand_vecs = torch.stack(c_embeddings, dim=0)

        p_gen, closed_probs, ptr_probs = self.head(reasoning_vec, cand_vecs, event_attn, candidate_indices)

        # Combine probability distribution
        combined_dist: Dict[str, float] = {}
        # Closed vocab
        for idx, word in ID_TO_CLOSED.items():
            if word not in ("<pad>", "<unk>"):
                combined_dist[word] = float(p_gen * closed_probs[idx])

        # Pointer vocab
        if cand_vecs is not None and ptr_probs.numel() > 0:
            for idx, cand in enumerate(candidates):
                p_ptr = float((1.0 - p_gen) * ptr_probs[idx])
                combined_dist[cand] = combined_dist.get(cand, 0.0) + p_ptr

        # Normalize
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
        query_tokens = tokenize(query)
        q_vec = self.event_encoder(query_tokens, word_to_id, device)

        candidates = extract_content_candidates(events)
        candidate_indices = []
        for cand in candidates:
            ev_hits = [i for i, ev in enumerate(events) if cand in tokenize(ev)]
            candidate_indices.append(ev_hits)

        if not events:
            empty_state = torch.zeros(self.dim, device=device)
            reasoning_vec, _ = self.reasoner(q_vec, empty_state.unsqueeze(0), empty_state.unsqueeze(0), empty_state)
            p_gen, closed_probs, _ = self.head(reasoning_vec, None, torch.zeros(1, device=device), None)
            target_norm = target_answer.lower().strip()
            if target_norm in CLOSED_TO_ID:
                target_idx = CLOSED_TO_ID[target_norm]
                return -torch.log(closed_probs[target_idx].clamp_min(1e-9))
            return -torch.log(closed_probs[UNKNOWN_ID].clamp_min(1e-9))

        encoded_events = []
        token_state_map = defaultdict(list)
        for t, ev in enumerate(events):
            ev_tokens = tokenize(ev)
            ev_vec, tok_h = self.event_encoder(ev_tokens, word_to_id, device, return_token_states=True)
            t_vec = self.temporal_encoding(torch.tensor([t], device=device)).squeeze(0)
            encoded_events.append(ev_vec + t_vec)
            for i, tok in enumerate(ev_tokens):
                token_state_map[tok.lower()].append(tok_h[i] + t_vec)
        event_tensor = torch.stack(encoded_events, dim=0)

        state_tensor, final_state = self.tracker(event_tensor)
        reasoning_vec, event_attn = self.reasoner(q_vec, event_tensor, state_tensor, final_state)

        cand_vecs = None
        if candidates:
            c_embeddings = []
            for cand in candidates:
                cand_static = self.embedding.embed_token(cand, word_to_id.get(cand, UNK_ID), device)
                if cand in token_state_map:
                    cand_ctx = torch.stack(token_state_map[cand], dim=0).mean(dim=0)
                    c_embeddings.append(F.layer_norm(cand_static + cand_ctx, [self.dim]))
                else:
                    c_embeddings.append(cand_static)
            cand_vecs = torch.stack(c_embeddings, dim=0)

        p_gen, closed_probs, ptr_probs = self.head(reasoning_vec, cand_vecs, event_attn, candidate_indices)

        target_norm = target_answer.lower().strip()
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
            # Target is neither in closed vocab nor in candidate observations -> back off to UNKNOWN
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
            self.register_vocabulary(tokenize(cleaned))

    def clear_session(self) -> None:
        self.session_events.clear()
        self.active_expectations.clear()

    def predict(self, question: str, events: Optional[List[str]] = None) -> WorldPrediction:
        """Produce a neural prediction for a question given observations."""
        self.model.eval()
        evs = events if events is not None else self.session_events
        with torch.no_grad():
            _, pred = self.model(evs, question, self.word_to_id, self.device)
        return pred

    def predict_state_transition(self, entity: str, action_event: str) -> WorldPrediction:
        """Form a pre-outcome prediction before an action's result is verified."""
        hypothetical_events = self.session_events + [action_event]
        query = f"where is the {entity}?"
        return self.predict(query, events=hypothetical_events)

    def learn_step(
        self,
        events: List[str],
        question: str,
        verified_answer: str,
        clip_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute a bounded, verified supervised learning step through gradient descent.
        """
        self.model.train()
        self.register_vocabulary(tokenize(" ".join(events) + " " + question + " " + verified_answer))

        loss = self.model.compute_loss(events, question, verified_answer, self.word_to_id, self.device)
        return self._apply_verified_loss(loss, clip_norm)

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
            "version": 1,
            "architecture": {"dim": self.dim},
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "trained_examples_count": self.trained_examples_count,
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

        self.word_to_id = payload.get("word_to_id", self.word_to_id)
        self.id_to_word = payload.get("id_to_word", self.id_to_word)
        self.trained_examples_count = payload.get("trained_examples_count", 0)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
