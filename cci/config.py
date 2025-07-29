# ---------------------------------------------------------------------
# Global configuration defaults. Override programmatically or through
# a CLI / environment variables as needed.
# ---------------------------------------------------------------------
from collections.abc import Callable
import math

# Concept extraction back‑end; see concepts.py
CONCEPT_METHOD: str = "spacy_noun_chunks"
CONCEPT_MODEL: str = "en_core_web_lg"
INCLUDE_VERBS: bool = True
MIN_TOKEN_LEN: int = 3
STOP_CONCEPTS: set = set()

# Semantic similarity
SBERT_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'  # Smaller, more available model
SEMANTIC_THRESHOLD: float = 0.6   # still used for any binary fallbacks

# Incorporation detection
LOOKAHEAD_WINDOW: int = 8  # How many turns to look ahead for incorporation

# Novelty vs. shared‑growth weight
ALPHA: float = 0.7
BETA: float = 0.5

# Small number to prevent division by zero
EPS: float = 1e-8

# Divergence rescaling function σ(x)
def default_sigma(x: float) -> float:
    "Linear rescale from [0,2] cosine distance → [0,1]"
    return x / 2.0

SIGMA: Callable[[float], float] = default_sigma

# Parallelisation
BATCH_SIZE: int = 10_000
N_PROCESSES: int | None = None