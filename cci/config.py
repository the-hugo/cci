# ---------------------------------------------------------------------
# Global configuration defaults. Override programmatically or through
# a CLI / environment variables as needed.
# ---------------------------------------------------------------------
from collections.abc import Callable
import math

# Novelty vs. shared‑growth weight
ALPHA: float = 0.7

# Small number to prevent division by zero
EPS: float = 1e-8

# Divergence rescaling function σ(x)
def default_sigma(x: float) -> float:
    "Linear rescale from [0,2] cosine distance → [0,1]"
    return x / 2.0

SIGMA: Callable[[float], float] = default_sigma

# Concept extraction back‑end; see concepts.py
CONCEPT_METHOD: str = "spacy_noun_chunks"

# Incorporation detection
LOOKAHEAD_WINDOW: int = 8  # How many turns to look ahead for incorporation
SEMANTIC_THRESHOLD: float = 0.5  # Minimum similarity for semantic concept matching (lowered for models without vectors)

# Parallelisation
BATCH_SIZE: int = 10_000
N_PROCESSES: int | None = None  # None → os.cpu_count()
