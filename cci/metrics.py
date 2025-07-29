from __future__ import annotations
import numpy as np
from typing import Set, Dict
import functools

from sentence_transformers import SentenceTransformer  # NEW heavy‑lift
from .config import (
    EPS, ALPHA, SIGMA,
    SEMANTIC_THRESHOLD,
    SBERT_MODEL             # e.g. "all-mpnet-base-v2"
)

# ================  fast cosine
def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    if du < EPS or dv < EPS:
        return 1.0
    return 1.0 - float(np.dot(u, v) / (du * dv))

# ================  divergence (now vs centroid)
def divergence(e_t: np.ndarray, e_prev_window: np.ndarray) -> float:
    """
    e_prev_window: expected shape (n, dim) – window of embeddings to compare
                   (caller should supply either 2‑D array or 1‑D single vec)
    """
    if e_prev_window.ndim == 1:
        centroid = e_prev_window
    else:
        centroid = e_prev_window.mean(axis=0)
    return SIGMA(cosine_distance(e_t, centroid))

# ------------------------------------------------------------------  semantic similarity (SBERT)
_EMBEDDER: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(SBERT_MODEL or "all-mpnet-base-v2", use_auth_token=False)
    return _EMBEDDER

@functools.lru_cache(maxsize=30_000)
def _embed(text: str) -> np.ndarray:
    return _get_embedder().encode(text, normalize_embeddings=True)

@functools.lru_cache(maxsize=100_000)
def semantic_similarity(concept1: str, concept2: str) -> float:
    """Cosine similarity between SBERT embeddings ∈[‑1,1] ⇒ mapped to [0,1]."""
    v1, v2 = _embed(concept1), _embed(concept2)
    sim = float(np.dot(v1, v2))          # because vectors already unit‑norm
    return 0.5 * (sim + 1.0)             # map [‑1,1] → [0,1]

# ------------------------------------------------------------------  soft incorporation
def incorporation(novel_prev: Set[str], grounded_next: Set[str]) -> float:
    """
    Soft incorporation score = average(max similarity to any future concept).
    0 if novel_prev empty.  Range ≈ [0,1].
    """
    if not novel_prev:
        return 0.0
    if not grounded_next:
        return 0.0

    sims = []
    for n in novel_prev:
        best = max(semantic_similarity(n, g) for g in grounded_next)
        sims.append(best)

    return float(np.mean(sims))

# ------------------------------------------------------------------  shared growth (Jaccard gain)
def shared_growth(g_prev: Set[str], g_next: Set[str], union_prev: Set[str]) -> float:
    if union_prev:
        j_prev = len(g_prev) / (len(union_prev) + EPS)
        j_next = len(g_next) / (len(union_prev) + EPS)
        return j_next - j_prev
    return 0.0

# ------------------------------------------------------------------  cc_turn (smoothed product)
def cc_turn(D_t: float, I_tp1: float, S_tp1: float,
            alpha: float = ALPHA, beta: float = 0.5, gamma: float = 1.0) -> float:
    """
    New combination:
        CCI = (β D  +  (1‑β)(α I + (1‑α) S))^γ
    Default β=0.5 is equal blend; γ=1 retains linearity.
    """
    mix = beta * D_t + (1 - beta) * (alpha * I_tp1 + (1 - alpha) * S_tp1)
    return mix ** gamma