"""
Creative Convergence Index (CCI) package
---------------------------------------

Compute turn‑level and dialogue‑level CCI scores for datasets that
already contain pre‑computed text embeddings.

Typical usage
-------------
>>> import cci
>>> import pandas as pd
>>> df = cci.io.load_dialogues("dialogues.pkl")
>>> scores = cci.pipeline.compute_cci(df)
>>> scores.head()
"""
from . import config, i_o, concepts, metrics, pipeline, tuning, viz

_all_ = ["config", "io", "concepts", "metrics", "pipeline",
           "tuning", "viz"]