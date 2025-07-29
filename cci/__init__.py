"""
Creative Convergence Index (CCI) package
---------------------------------------

Compute CCI scores per speaker pair, measuring how each speaker builds upon
others' contributions using a windowed approach examining up to LOOKBACK_WINDOW previous
turns from the target speaker.

For each speaker pair (from_speaker, to_speaker), calculates how from_speaker
builds upon to_speaker's recent contributions based on concept novelty,
grounding, and incorporation patterns.

Requires datasets with pre‑computed text embeddings and speaker information.

Typical usage
-------------
>>> import cci
>>> import pandas as pd
>>> df = cci.i_o.load_dialogues("dialogues.pkl")  # Must include 'speaker' column
>>> result_df = cci.pipeline.compute_cci(df)
>>> 
>>> # Result is a DataFrame with columns:
>>> # - dialogue_id: dialogue identifier
>>> # - from_speaker: speaker building upon another
>>> # - to_speaker: speaker being built upon  
>>> # - CCI_score: CCI score for this interaction
>>> # - n_interactions: number of interactions analyzed
>>> 
>>> print(result_df.head())
>>> 
>>> # Filter for specific speaker interactions
>>> a_to_b = result_df[(result_df['from_speaker'] == 'A') & 
>>>                    (result_df['to_speaker'] == 'B')]
"""
from . import config, i_o, concepts, metrics, pipeline

__all__ = ["config", "i_o", "concepts", "metrics", "pipeline"]