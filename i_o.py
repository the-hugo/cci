# ---------------------------------------------------------------------
# I/O utilities: load and validate dialogue data stored as a pickle.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import pickle
import numpy as np

REQUIRED_COLUMNS = {
    "dialogue_id", "turn_index", "speaker", "text", "embedding"
}

def load_dialogues(path: str) -> pd.DataFrame:
    """Load pickled DataFrame and ensure required schema."""
    with open(path, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle must contain a pandas.DataFrame")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    # Sort deterministically
    df = df.sort_values(["dialogue_id", "turn_index"]).reset_index(drop=True)

    # Check embeddings look numeric
    if not np.issubdtype(df["embedding"].iloc[0].dtype, np.number):
        raise TypeError("embedding column must contain numeric vectors")

    return df
