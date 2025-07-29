# ---------------------------------------------------------------------
# Pure numerical functions for CCI computation (vectorised with NumPy)
# ---------------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Set
import functools

from .config import EPS, ALPHA, SIGMA, SEMANTIC_THRESHOLD

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
    """
    Calculate incorporation using both exact matching and semantic similarity.
    
    Args:
        novel_prev: Novel concepts from previous turn
        grounded_next: Concepts that appear in future turns
    
    Returns:
        Float between 0 and 1 representing incorporation rate
    """
    if not novel_prev:
        return 0.0
    
    # Start with exact matching
    exact_matches = len(novel_prev & grounded_next)
    
    # Add semantic matches for concepts that didn't match exactly
    unmatched_novel = novel_prev - grounded_next
    semantic_matches = 0
    
    if unmatched_novel and grounded_next:
        for novel_concept in unmatched_novel:
            for grounded_concept in grounded_next:
                if semantic_similarity(novel_concept, grounded_concept) >= SEMANTIC_THRESHOLD:
                    semantic_matches += 1
                    break  # Count each novel concept only once
    
    total_incorporated = exact_matches + semantic_matches
    return total_incorporated / (len(novel_prev) + EPS)

def shared_growth(g_prev: Set[str], g_next: Set[str], union_prev: Set[str]) -> float:
    if union_prev:
        return (len(g_next) - len(g_prev)) / (len(union_prev) + EPS)
    return 0.0

def cc_turn(D_t: float, I_tp1: float, S_tp1: float, alpha: float = ALPHA) -> float:
    return D_t * (alpha * I_tp1 + (1 - alpha) * S_tp1)

# --------------------------------------------------  semantic similarity
@functools.lru_cache(maxsize=10000)
def _get_nlp_for_similarity():
    """Get spaCy model for semantic similarity (cached)."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None

def semantic_similarity(concept1: str, concept2: str) -> float:
    """Calculate semantic similarity between two concepts using spaCy vectors."""
    nlp = _get_nlp_for_similarity()
    if nlp is None:
        return 1.0 if concept1 == concept2 else 0.0  # Fallback to exact matching
    
    doc1 = nlp(concept1)
    doc2 = nlp(concept2)
    
    # Use document similarity if available
    if doc1.has_vector and doc2.has_vector:
        return float(doc1.similarity(doc2))
    
    # Fallback to exact matching
    return 1.0 if concept1 == concept2 else 0.0

def semantic_incorporation(novel_concepts: Set[str], future_concepts: Set[str], 
                         threshold: float = SEMANTIC_THRESHOLD) -> float:
    """
    Calculate incorporation rate using semantic similarity.
    
    For each novel concept, check if any future concept is semantically similar.
    Returns fraction of novel concepts that were incorporated semantically.
    """
    if not novel_concepts:
        return 0.0
    
    incorporated_count = 0
    
    for novel_concept in novel_concepts:
        # Check if this novel concept has semantic similarity with any future concept
        max_similarity = 0.0
        for future_concept in future_concepts:
            similarity = semantic_similarity(novel_concept, future_concept)
            max_similarity = max(max_similarity, similarity)
            
            if similarity >= threshold:
                incorporated_count += 1
                break  # Found a match, no need to check other future concepts
    
    return incorporated_count / (len(novel_concepts) + EPS)