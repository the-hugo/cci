# ---------------------------------------------------------------------
# High‑level orchestration: from raw DataFrame → CCI per dialogue.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import numpy as np
import multiprocessing as mp

from .concepts import extract_concepts
from .metrics import (
    divergence, novel_set, grounded_set,
    incorporation, shared_growth, cc_turn
)
from .config import EPS, ALPHA, N_PROCESSES, BATCH_SIZE

# --------------------------------------------------  concept extraction
def _extract_concepts_worker(texts: list[str]) -> list[set[str]]:
    return [extract_concepts(t) for t in texts]

def add_concepts_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'concepts' column (set[str]) to the DataFrame."""
    if "concepts" in df.columns:
        return df  # already done
    n = len(df)
    batches = [df["text"].iloc[i:i+BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    with mp.Pool(processes=N_PROCESSES) as pool:
        results = pool.map(_extract_concepts_worker, batches)
    concepts_flat = [c for batch in results for c in batch]
    df = df.copy()
    df["concepts"] = concepts_flat
    return df

# --------------------------------------------------  core computation
def _compute_per_dialogue(dialogue: pd.DataFrame) -> float:
    """Return dialogue‑level CCI."""
    dialogue = dialogue.reset_index(drop=True)
    n_turns = len(dialogue)
    if n_turns < 3:
        return 0.0  # not enough turns for metric

    embeddings = np.stack(dialogue["embedding"].to_numpy(), axis=0)
    concepts = dialogue["concepts"].tolist()

    cc_scores: list[float] = []

    # iterate from t = 2 to n‑2 (inclusive) for CC_t definition
    for t in range(1, n_turns - 1):
        e_t = embeddings[t]
        e_prev = embeddings[t-1]
        D_t = divergence(e_t, e_prev)

        # Conceptual sets
        c_t = concepts[t]
        c_prev = concepts[t-1]
        c_prev_prev = concepts[t-2] if t >= 2 else None

        n_t = novel_set(c_t, c_prev, c_prev_prev)
        g_t = grounded_set(c_prev, c_t)

        # Look‑ahead (t+1)
        c_next = concepts[t+1]
        g_next = grounded_set(c_t, c_next)
        union_prev = c_prev | c_t

        I_tp1 = incorporation(n_t, g_next)
        S_tp1 = shared_growth(g_t, g_next, union_prev)

        cc_scores.append(cc_turn(D_t, I_tp1, S_tp1))

    if not cc_scores:
        return 0.0
    return float(np.mean(cc_scores))

def compute_cci(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame [dialogue_id, CCI] as result."""
    df = add_concepts_column(df)

    # Run per dialogue in parallel
    grouped = [g for _, g in df.groupby("dialogue_id", sort=True)]

    with mp.Pool(processes=N_PROCESSES) as pool:
        cci_values = pool.map(_compute_per_dialogue, grouped)

    result = pd.DataFrame({
        "dialogue_id": [g["dialogue_id"].iloc[0] for g in grouped],
        "CCI": cci_values
    })
    return result.sort_values("CCI", ascending=False).reset_index(drop=True)