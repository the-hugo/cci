# ---------------------------------------------------------------------
# Lightweight visual aids using matplotlib (single plots, no seaborn).
# ---------------------------------------------------------------------
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_turn_scores(dialogue_df: pd.DataFrame, title: str | None = None):
    """Assumes dialogue_df already contains a cc_turn column."""
    if "cc_turn" not in dialogue_df.columns:
        raise ValueError("DataFrame must include 'cc_turn' per turn")
    plt.figure()
    plt.plot(dialogue_df["turn_index"], dialogue_df["cc_turn"], marker="o")
    plt.xlabel("Turn index")
    plt.ylabel("CC_t")
    plt.title(title or f"Turn‑level CC_t for dialogue {dialogue_df['dialogue_id'].iloc[0]}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

