# ---------------------------------------------------------------------
# Optional hyper‑parameter search if you have ground‑truth ratings.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import itertools
from scipy.stats import spearmanr

from .pipeline import compute_cci
from .config import ALPHA, SIGMA, default_sigma

def grid_search(
    df: pd.DataFrame,
    ratings: pd.DataFrame,
    alphas=(0.5, 0.6, 0.7, 0.8, 0.9),
    sigmas=(default_sigma,)
):
    """
    ratings: DataFrame with ['dialogue_id', 'human_score']
    Returns: best (alpha, sigma_fn, rho)
    """
    best_rho = -2
    best_params = None

    for alpha, sigma_fn in itertools.product(alphas, sigmas):
        # monkey patch config for this iteration
        import cci.config as cfg
        cfg.ALPHA = alpha
        cfg.SIGMA = sigma_fn

        cci_scores = compute_cci(df)
        merged = pd.merge(cci_scores, ratings, on="dialogue_id")
        rho, _ = spearmanr(merged["CCI"], merged["human_score"])
        if rho > best_rho:
            best_rho = rho
            best_params = (alpha, sigma_fn)

    return best_params + (best_rho,)

