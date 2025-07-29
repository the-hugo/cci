# ---------------------------------------------------------------------
# Enhanced concept‑extraction strategies.
# Default: noun‑phrase lemmas + NE fusion + optional verbs + coref.
# ---------------------------------------------------------------------
from __future__ import annotations
import functools
import threading
from typing import Iterable, Set, List

import spacy
from spacy.lang.en import English
from spacy.tokens import Doc

from .config import (
    CONCEPT_METHOD,            # "spacy_noun_chunks" (kept for compat)
    CONCEPT_MODEL,             # NEW: e.g. "en_core_web_lg"
    INCLUDE_VERBS,             # NEW: bool
    MIN_TOKEN_LEN,             # NEW: filter junk like "us"
    STOP_CONCEPTS              # NEW: optional blacklist list[str]
)

# ------------------------------------------------------------------  NLP loader
_NLP: English | None = None
_COREF_ENABLED = False
_NLP_LOCK = threading.Lock()

def _get_nlp() -> English:
    global _NLP, _COREF_ENABLED
    if _NLP is not None:
        return _NLP

    with _NLP_LOCK:
        # Double-check locking pattern
        if _NLP is not None:
            return _NLP

        model_name = CONCEPT_MODEL or "en_core_web_lg"
        try:
            _NLP = spacy.load(model_name)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' not installed.\n"
                f"Run: python -m spacy download {model_name}"
            ) from e

        # -------- optional coreference resolver (silently skip if unavailable)
        try:
            import spacy_coref  # type: ignore
            _NLP.add_pipe("spacy-coref", config={"resolve_text": True})
            _COREF_ENABLED = True
        except (ImportError, ValueError):
            _COREF_ENABLED = False

    return _NLP

# ------------------------------------------------------------------  helpers
def _resolve_coref(doc: Doc) -> Doc:
    """Return the doc with coreferences resolved if the pipe is available."""
    if _COREF_ENABLED and hasattr(doc._, 'resolved') and doc._.resolved is not None:
        return _get_nlp()(doc._.resolved)
    return doc

def _iter_concepts(doc: Doc) -> Iterable[str]:
    """
    Yield candidate concept strings (lemmatised, lower‑cased).
    • noun‑phrase lemmas (head‑root lemma)       (existing behaviour)
    • fused named entities (multi‑token kept)    (NEW)
    • verbs (optional)                           (NEW)
    """
    # ---- 1. noun‑phrase / proper‑noun same as before
    for chunk in doc.noun_chunks:
        if chunk.root.is_stop:
            continue
        lemma = chunk.lemma_.lower()
        if len(lemma) >= MIN_TOKEN_LEN:
            yield lemma

    # ---- 2. named entities (keep full span)
    for ent in doc.ents:
        ent_text = ent.text.lower()
        if len(ent_text) >= MIN_TOKEN_LEN:
            yield ent_text
        if " " in ent.text:
            acronym = "".join(tok[0] for tok in ent.text.split() if tok[0].isalpha())
            if len(acronym) >= MIN_TOKEN_LEN:
                yield acronym.lower()
    # ---- 3. verbs capturing actions / processes
    if INCLUDE_VERBS:
        for tok in doc:
            if tok.pos_ == "VERB" and not tok.is_stop:
                lemma = tok.lemma_.lower()
                if len(lemma) >= MIN_TOKEN_LEN:
                    yield lemma

@functools.lru_cache(maxsize=65_536)
def extract_concepts(text: str) -> Set[str]:
    """Return a set of extracted concepts according to strategy above."""
    if CONCEPT_METHOD != "spacy_noun_chunks":
        raise ValueError(f"Unsupported concept method: {CONCEPT_METHOD}")

    nlp = _get_nlp()
    doc = nlp(text)
    doc = _resolve_coref(doc)

    concepts = {c for c in _iter_concepts(doc) if c not in STOP_CONCEPTS}

    return concepts