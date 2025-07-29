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
    print(f"Loading dialogue data from {path}...")
    
    with open(path, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle must contain a pandas.DataFrame")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    print(f"Data loaded successfully")
    print(f"  - Total turns: {len(df):,}")
    print(f"  - Unique dialogues: {df['dialogue_id'].nunique():,}")
    print(f"  - Average turns per dialogue: {len(df) / df['dialogue_id'].nunique():.1f}")

    # Sort deterministically
    print("Sorting data by dialogue_id and turn_index...")
    df = df.sort_values(["dialogue_id", "turn_index"]).reset_index(drop=True)

    # Check embeddings look numeric
    print("Validating embeddings...")
    if not np.issubdtype(df["embedding"].iloc[0].dtype, np.number):
        raise TypeError("embedding column must contain numeric vectors")
    
    print("Data validation complete")
    return df
