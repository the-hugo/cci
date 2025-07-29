# ---------------------------------------------------------------------
# I/O utilities: load and validate dialogue data stored as a pickle.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import pickle
import numpy as np
import os
import warnings

REQUIRED_COLUMNS = {
    "dialogue_id", "turn_index", "speaker", "text", "embedding"
}

def load_dialogues(path: str) -> pd.DataFrame:
    """Load pickled DataFrame and ensure required schema.
    
    WARNING: This function uses pickle.load(), which can execute arbitrary code.
    Only load pickle files from trusted sources.
    """
    # Security warning
    warnings.warn(
        "Loading pickle files can execute arbitrary code. Only load files from trusted sources.",
        UserWarning,
        stacklevel=2
    )
    
    print(f"Loading dialogue data from {path}...")
    
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check file is readable
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read file: {path}")
    
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Failed to load pickle file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading file: {e}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle must contain a pandas.DataFrame")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    print(f"Data loaded successfully")
    print(f"  - Total turns: {len(df):,}")
    
    # Prevent division by zero
    n_dialogues = df['dialogue_id'].nunique()
    if n_dialogues == 0:
        raise ValueError("No unique dialogues found in data")
        
    print(f"  - Unique dialogues: {n_dialogues:,}")
    print(f"  - Average turns per dialogue: {len(df) / n_dialogues:.1f}")

    # Sort deterministically
    print("Sorting data by dialogue_id and turn_index...")
    try:
        df = df.sort_values(["dialogue_id", "turn_index"]).reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to sort data - incompatible data types: {e}")

    # Check embeddings look numeric
    print("Validating embeddings...")
    if len(df) == 0:
        raise ValueError("DataFrame became empty after sorting")
        
    first_embedding = df["embedding"].iloc[0]
    if first_embedding is None:
        raise ValueError("First embedding is None")
        
    if not isinstance(first_embedding, np.ndarray):
        raise TypeError("Embeddings must be numpy arrays")
        
    if not np.issubdtype(first_embedding.dtype, np.number):
        raise TypeError("embedding column must contain numeric vectors")
    
    # Validate all embeddings have same shape (sample check)
    sample_size = min(100, len(df))
    sample_embeddings = df["embedding"].iloc[:sample_size]
    expected_shape = first_embedding.shape
    
    for i, emb in enumerate(sample_embeddings):
        if emb is None or not isinstance(emb, np.ndarray):
            raise ValueError(f"Invalid embedding at index {i}: not a numpy array")
        if emb.shape != expected_shape:
            raise ValueError(f"Embedding shape mismatch at index {i}: expected {expected_shape}, got {emb.shape}")
    
    print("Data validation complete")
    return df
