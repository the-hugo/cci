# ---------------------------------------------------------------------
# Concept extraction strategies.  Default: noun‑phrase lemmas via spaCy.
# ---------------------------------------------------------------------
from __future__ import annotations
import functools
import os
from typing import Iterable, Set

import spacy
from spacy.lang.en import English

from .config import CONCEPT_METHOD

# Lazy global NLP model
_NLP: English | None = None

def _get_nlp() -> English:
    global _NLP
    if _NLP is None:  # load once
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not installed. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _NLP

@functools.lru_cache(maxsize=65_536)
def extract_concepts(text: str) -> Set[str]:
    """Return a set of lemmatized noun phrases / proper nouns."""
    if CONCEPT_METHOD != "spacy_noun_chunks":
        raise ValueError(f"Unsupported concept method: {CONCEPT_METHOD}")

    doc = _get_nlp()(text)

    # Gather lemmatized noun‑phrases and proper nouns
    tokens: Iterable[str] = (
        chunk.lemma_.lower()
        for chunk in doc.noun_chunks
        if not chunk.root.is_stop
    )
    proper_nouns = (
        token.lemma_.lower() for token in doc if token.pos_ == "PROPN"
    )

    return set(tokens) | set(proper_nouns)
