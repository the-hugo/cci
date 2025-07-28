# ---------------------------------------------------------------------
# Pure numerical functions for CCI computation (vectorised with NumPy)
# ---------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Set

from .config import EPS, ALPHA, SIGMA

# --------------------------------------------------  basic helpers
def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cos(u,v) ; if any vector is 0, distance → 1 (orthogonal)."""
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    if du < EPS or dv < EPS:
        return 1.0
    return 1.0 - float(np.dot(u, v) / (du * dv))

def divergence(e_t: np.ndarray, e_prev: np.ndarray) -> float:
    return SIGMA(cosine_distance(e_t, e_prev))

def grounded_set(c_prev: Set[str], c_t: Set[str]) -> Set[str]:
    return c_prev & c_t

def novel_set(c_t: Set[str],
              c_prev: Set[str] | None,
              c_prev_prev: Set[str] | None) -> Set[str]:
    union_prev = set()
    if c_prev is not None:
        union_prev |= c_prev
    if c_prev_prev is not None:
        union_prev |= c_prev_prev
    return c_t - union_prev

def incorporation(novel_prev: Set[str], grounded_next: Set[str]) -> float:
    if not novel_prev:
        return 0.0
    return len(novel_prev & grounded_next) / (len(novel_prev) + EPS)

def shared_growth(g_prev: Set[str], g_next: Set[str], union_prev: Set[str]) -> float:
    if union_prev:
        return (len(g_next) - len(g_prev)) / (len(union_prev) + EPS)
    return 0.0

def cc_turn(D_t: float, I_tp1: float, S_tp1: float, alpha: float = ALPHA) -> float:
    return D_t * (alpha * I_tp1 + (1 - alpha) * S_tp1)