# ---------------------------------------------------------------------
# High‑level orchestration: from raw DataFrame → CCI per dialogue.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import Set

from .concepts import extract_concepts
from .metrics import divergence, incorporation, shared_growth, cc_turn
from .config import EPS, ALPHA, N_PROCESSES, BATCH_SIZE, LOOKBACK_WINDOW, LOOKAHEAD_WINDOW


# --------------------------------------------------  concept extraction
def _extract_concepts_worker(texts: list[str]) -> list[set[str]]:
    return [extract_concepts(t) for t in texts]


def add_concepts_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'concepts' column (set[str]) to the DataFrame."""
    if "concepts" in df.columns:
        print("SUCCESS: Concepts already extracted")
        return df  # already done

    n = len(df)
    print(f"Extracting concepts from {n:,} turns...")

    batches = [
        df["text"].iloc[i : i + BATCH_SIZE].tolist() for i in range(0, n, BATCH_SIZE)
    ]

    try:
        with mp.Pool(processes=N_PROCESSES) as pool:
            results = list(
                tqdm(
                    pool.imap(_extract_concepts_worker, batches),
                    total=len(batches),
                    desc="Processing batches",
                )
            )
    except Exception as e:
        raise RuntimeError(f"Multiprocessing failed during concept extraction: {e}")

    concepts_flat = [c for batch in results for c in batch]
    df = df.copy()
    df["concepts"] = concepts_flat
    print("SUCCESS: Concept extraction complete")
    return df


# --------------------------------------------------  core computation
def _compute_per_dialogue(dialogue: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return k×k CCI matrix and interaction count matrix.

    For each speaker pair (j, i), calculates how speaker j builds upon speaker i's
    contributions by looking at a window of up to LOOKBACK_WINDOW previous turns from speaker i
    when speaker j speaks.

    Returns:
        tuple: (cci_matrix, count_matrix) where both are k×k matrices
               cci_matrix[j,i] = weighted CCI score of speaker j relative to speaker i
               count_matrix[j,i] = number of interactions used for this score
    """
    dialogue = dialogue.reset_index(drop=True)
    n_turns = len(dialogue)
    
    # Validate required columns exist
    required_cols = {"embedding", "concepts", "speaker"}
    missing_cols = required_cols - set(dialogue.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data integrity
    if n_turns == 0:
        raise ValueError("Empty dialogue")
        
    # Validate embeddings before stacking
    embeddings_series = dialogue["embedding"]
    if embeddings_series.isnull().any():
        raise ValueError("Found null embeddings in dialogue")
        
    # Check first embedding to get expected shape
    first_emb = embeddings_series.iloc[0]
    if not isinstance(first_emb, np.ndarray):
        raise TypeError("Embeddings must be numpy arrays")
    expected_shape = first_emb.shape
    
    # Validate all embeddings have consistent shape
    for i, emb in enumerate(embeddings_series):
        if emb is None or not isinstance(emb, np.ndarray):
            raise ValueError(f"Invalid embedding at turn {i}: not a numpy array")
        if emb.shape != expected_shape:
            raise ValueError(f"Embedding shape mismatch at turn {i}: expected {expected_shape}, got {emb.shape}")
    
    try:
        embeddings = np.stack(embeddings_series.to_numpy(), axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to stack embeddings: {e}")
        
    concepts = dialogue["concepts"].tolist()
    speakers = dialogue["speaker"].tolist()
    
    # Validate concepts
    if any(c is None for c in concepts):
        raise ValueError("Found null concepts in dialogue")
    if any(not isinstance(c, set) for c in concepts):
        raise ValueError("All concepts must be sets")
        
    # Validate speakers
    if any(s is None or s == "" for s in speakers):
        raise ValueError("Found null or empty speaker names in dialogue")

    # Get unique speakers and create mapping
    unique_speakers = sorted(list(set(speakers)))
    k = len(unique_speakers)
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    
    if n_turns < LOOKBACK_WINDOW + 1:  # Need at least LOOKBACK_WINDOW+1 turns for meaningful windowed calculation
        return np.zeros((k, k)), np.zeros((k, k), dtype=int)

    # Initialize k×k CCI matrix and count matrix
    cci_matrix = np.zeros((k, k))
    count_matrix = np.zeros((k, k), dtype=int)

    # For each speaker pair (j, i), calculate CCI score of j building upon i
    for j_speaker in unique_speakers:
        j_idx = speaker_to_idx[j_speaker]

        for i_speaker in unique_speakers:
            if j_speaker == i_speaker:
                continue  # Skip self-interactions

            i_idx = speaker_to_idx[i_speaker]
            cc_scores = []
            cc_weights = []

            # Find all turns by speaker j
            for t in range(LOOKBACK_WINDOW, n_turns):
                if speakers[t] != j_speaker:
                    continue

                current_embedding = embeddings[t]
                current_concepts = concepts[t]

                # Find window of previous turns from speaker i only
                window_start = max(0, t - LOOKBACK_WINDOW)
                i_window_turns = []

                for idx in range(t - 1, window_start - 1, -1):  # Go backwards from t-1
                    if speakers[idx] == i_speaker:
                        i_window_turns.append(idx)
                    if len(i_window_turns) >= LOOKBACK_WINDOW:  # Limit to configured turns from speaker i
                        break

                if len(i_window_turns) == 0:
                    continue  # Skip if no speaker i turns in window

                i_window_embeddings = embeddings[i_window_turns]  # shape (m, dim)
                D_t = divergence(
                    current_embedding, i_window_embeddings
                )  # centroid inside

                # ---  concepts sets same as before  ---------------------------------
                i_windowed_concepts = set().union(
                    *(concepts[idx] for idx in i_window_turns)
                )
                n_t = current_concepts - i_windowed_concepts
                g_t = current_concepts & i_windowed_concepts
                # --------------------------------------------------------------------

                # ---- look‑ahead for future‑i concepts identical to earlier code ----
                future_i_concepts: Set[str] = set()
                look_ahead_limit = min(n_turns, t + LOOKAHEAD_WINDOW + 1)
                for future_idx in range(t + 1, look_ahead_limit):
                    if speakers[future_idx] == i_speaker:
                        future_i_concepts.update(concepts[future_idx])

                # ---  NEW soft incorporation / shared growth  -----------------------
                I_tp1 = incorporation(n_t, future_i_concepts)
                union_context = i_windowed_concepts | current_concepts
                S_tp1 = shared_growth(
                    g_t, future_i_concepts & current_concepts, union_context
                )

                cc_score = cc_turn(D_t, I_tp1, S_tp1)
                cc_scores.append(cc_score)
                cc_weights.append(
                    len(i_window_turns)
                )  # Weight by number of interactions

            # Set matrix entry C[j,i] = weighted average CCI score of j building upon i
            if cc_scores:
                # Weighted average by interaction frequency
                weighted_avg = np.average(cc_scores, weights=cc_weights)
                cci_matrix[j_idx, i_idx] = float(weighted_avg)
                count_matrix[j_idx, i_idx] = len(cc_scores)

    return cci_matrix, count_matrix


def _compute_per_dialogue_with_components(dialogue: pd.DataFrame) -> dict:
    """
    Return detailed component analysis for each speaker pair.

    Returns:
        dict: {(j_idx, i_idx): {'cci_score': float, 'n_interactions': int,
                               'D_mean': float, 'D_std': float,
                               'I_mean': float, 'I_std': float,
                               'S_mean': float, 'S_std': float}}
    """
    dialogue = dialogue.reset_index(drop=True)
    n_turns = len(dialogue)
    result = {}

    # Validate required columns exist
    required_cols = {"embedding", "concepts", "speaker"}
    missing_cols = required_cols - set(dialogue.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data integrity
    if n_turns == 0:
        raise ValueError("Empty dialogue")
        
    # Validate embeddings before stacking
    embeddings_series = dialogue["embedding"]
    if embeddings_series.isnull().any():
        raise ValueError("Found null embeddings in dialogue")
        
    # Check first embedding to get expected shape
    first_emb = embeddings_series.iloc[0]
    if not isinstance(first_emb, np.ndarray):
        raise TypeError("Embeddings must be numpy arrays")
    expected_shape = first_emb.shape
    
    # Validate all embeddings have consistent shape
    for i, emb in enumerate(embeddings_series):
        if emb is None or not isinstance(emb, np.ndarray):
            raise ValueError(f"Invalid embedding at turn {i}: not a numpy array")
        if emb.shape != expected_shape:
            raise ValueError(f"Embedding shape mismatch at turn {i}: expected {expected_shape}, got {emb.shape}")
    
    try:
        embeddings = np.stack(embeddings_series.to_numpy(), axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to stack embeddings: {e}")
        
    concepts = dialogue["concepts"].tolist()
    speakers = dialogue["speaker"].tolist()
    
    # Validate concepts
    if any(c is None for c in concepts):
        raise ValueError("Found null concepts in dialogue")
    if any(not isinstance(c, set) for c in concepts):
        raise ValueError("All concepts must be sets")
        
    # Validate speakers
    if any(s is None or s == "" for s in speakers):
        raise ValueError("Found null or empty speaker names in dialogue")

    unique_speakers = sorted(list(set(speakers)))
    k = len(unique_speakers)
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    if n_turns < LOOKBACK_WINDOW + 1:
        # Return empty result for all speaker pairs
        for j_idx in range(k):
            for i_idx in range(k):
                if j_idx != i_idx:
                    result[(j_idx, i_idx)] = {
                        "cci_score": 0.0,
                        "n_interactions": 0,
                        "D_mean": 0.0, "D_std": 0.0,
                        "I_mean": 0.0, "I_std": 0.0,
                        "S_mean": 0.0, "S_std": 0.0
                    }
        return result

    # For each speaker pair, collect component values
    for j_speaker in unique_speakers:
        j_idx = speaker_to_idx[j_speaker]

        for i_speaker in unique_speakers:
            if j_speaker == i_speaker:
                continue

            i_idx = speaker_to_idx[i_speaker]

            # Collect all component values for this speaker pair
            D_values = []
            I_values = []
            S_values = []
            cc_scores = []
            cc_weights = []

            # Find all turns by speaker j
            for t in range(LOOKBACK_WINDOW, n_turns):
                if speakers[t] != j_speaker:
                    continue

                current_embedding = embeddings[t]
                current_concepts = concepts[t]

                # Find window of previous turns from speaker i only
                window_start = max(0, t - LOOKBACK_WINDOW)
                i_window_turns = []

                for idx in range(t - 1, window_start - 1, -1):
                    if speakers[idx] == i_speaker:
                        i_window_turns.append(idx)
                    if len(i_window_turns) >= LOOKBACK_WINDOW:
                        break

                if len(i_window_turns) == 0:
                    continue

                weight = len(i_window_turns)

                # Calculate components
                i_window_embeddings = embeddings[i_window_turns]  # (m, dim)
                D_t = divergence(current_embedding, i_window_embeddings)

                i_windowed_concepts = set()
                for idx in i_window_turns:
                    i_windowed_concepts.update(concepts[idx])

                n_t = current_concepts - i_windowed_concepts
                g_t = current_concepts & i_windowed_concepts

                future_i_concepts = set()
                look_ahead_limit = min(n_turns, t + LOOKAHEAD_WINDOW + 1)
                for future_idx in range(t + 1, look_ahead_limit):
                    if speakers[future_idx] == i_speaker:
                        future_i_concepts.update(concepts[future_idx])

                I_tp1 = incorporation(n_t, future_i_concepts)
                union_context = i_windowed_concepts | current_concepts
                S_tp1 = shared_growth(
                    g_t, future_i_concepts & current_concepts, union_context
                )

                # Store component values
                D_values.append(D_t)
                I_values.append(I_tp1)
                S_values.append(S_tp1)

                cc_score = cc_turn(D_t, I_tp1, S_tp1)
                cc_scores.append(cc_score)
                cc_weights.append(weight)

            # Calculate statistics
            if cc_scores:
                weighted_avg = np.average(cc_scores, weights=cc_weights)
                result[(j_idx, i_idx)] = {
                    "cci_score": float(weighted_avg),
                    "n_interactions": len(cc_scores),
                    "D_mean": float(np.mean(D_values)),
                    "D_std": float(np.std(D_values)) if len(D_values) > 1 else 0.0,
                    "I_mean": float(np.mean(I_values)),
                    "I_std": float(np.std(I_values)) if len(I_values) > 1 else 0.0,
                    "S_mean": float(np.mean(S_values)),
                    "S_std": float(np.std(S_values)) if len(S_values) > 1 else 0.0,
                }
            else:
                result[(j_idx, i_idx)] = {
                    "cci_score": 0.0,
                    "n_interactions": 0,
                    "D_mean": 0.0,
                    "D_std": 0.0,
                    "I_mean": 0.0,
                    "I_std": 0.0,
                    "S_mean": 0.0,
                    "S_std": 0.0,
                }

    return result


def compute_cci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return CCI scores per speaker pair.

    Returns:
        pd.DataFrame with columns:
        - dialogue_id: dialogue identifier
        - from_speaker: speaker who is building upon another
        - to_speaker: speaker being built upon
        - CCI_score: CCI score for this speaker pair interaction
        - n_interactions: number of interactions used to compute this score
    """
    df = add_concepts_column(df)

    # Run per dialogue in parallel
    grouped = [g for _, g in df.groupby("dialogue_id", sort=True)]
    n_dialogues = len(grouped)

    print(f"Computing CCI matrices for {n_dialogues:,} dialogues...")

    try:
        with mp.Pool(processes=N_PROCESSES) as pool:
            results = list(
                tqdm(
                    pool.imap(_compute_per_dialogue, grouped),
                    total=n_dialogues,
                    desc="Computing CCI",
                )
            )
    except Exception as e:
        raise RuntimeError(f"Multiprocessing failed during CCI computation: {e}")

    # Separate CCI matrices and count matrices
    cci_matrices = [result[0] for result in results]
    count_matrices = [result[1] for result in results]

    # Flatten matrices into speaker pair results
    speaker_pair_results = []
    all_cci_scores = []

    for group, matrix, count_matrix in zip(grouped, cci_matrices, count_matrices):
        dialogue_id = group["dialogue_id"].iloc[0]
        speakers = sorted(list(set(group["speaker"].tolist())))

        # Extract speaker pair scores from matrix
        for j_idx, from_speaker in enumerate(speakers):
            for i_idx, to_speaker in enumerate(speakers):
                if j_idx != i_idx:  # Skip self-interactions
                    cci_score = matrix[j_idx, i_idx]
                    n_interactions = int(count_matrix[j_idx, i_idx])

                    speaker_pair_results.append(
                        {
                            "dialogue_id": dialogue_id,
                            "from_speaker": from_speaker,
                            "to_speaker": to_speaker,
                            "CCI_score": float(cci_score),
                            "n_interactions": n_interactions,
                        }
                    )

                    if abs(cci_score) > EPS:  # Only include non-zero scores in statistics
                        all_cci_scores.append(cci_score)

    result_df = pd.DataFrame(speaker_pair_results)

    # Print summary statistics
    if len(all_cci_scores) > 0:
        print("SUCCESS: CCI computation complete")
        print(f"  - Total speaker pairs: {len(result_df):,}")
        print(f"  - Non-zero CCI scores: {len(all_cci_scores):,}")
        print(f"  - Mean CCI: {np.mean(all_cci_scores):.4f}")
        print(f"  - Std CCI: {np.std(all_cci_scores):.4f}")
        print(f"  - Max CCI: {np.max(all_cci_scores):.4f}")
        print(f"  - Min CCI: {np.min(all_cci_scores):.4f}")
    else:
        print("SUCCESS: CCI computation complete (no non-zero scores found)")

    return result_df.sort_values(
        ["dialogue_id", "CCI_score"], ascending=[True, False]
    ).reset_index(drop=True)


def compute_cci_with_components(
    df: pd.DataFrame, use_multiprocessing: bool = True
) -> pd.DataFrame:
    """
    Return CCI scores with component breakdown for diagnostic analysis.

    Args:
        df: Input DataFrame with dialogue data
        use_multiprocessing: Whether to use multiprocessing (disable on Windows if issues)

    Returns:
        pd.DataFrame with additional columns:
        - D_mean, D_std: Divergence component statistics
        - I_mean, I_std: Incorporation component statistics
        - S_mean, S_std: Shared growth component statistics
        - component_counts: Number of turns used for each component
    """
    df = add_concepts_column(df)

    # Run per dialogue with optional multiprocessing
    grouped = [g for _, g in df.groupby("dialogue_id", sort=True)]
    n_dialogues = len(grouped)

    print(f"Computing CCI with component analysis for {n_dialogues:,} dialogues...")

    if use_multiprocessing and N_PROCESSES != 1:
        try:
            with mp.Pool(processes=N_PROCESSES) as pool:
                results = list(
                    tqdm(
                        pool.imap(_compute_per_dialogue_with_components, grouped),
                        total=n_dialogues,
                        desc="Computing CCI with components",
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Multiprocessing failed during CCI components computation: {e}")
    else:
        # Single-threaded version for Windows compatibility
        results = list(
            tqdm(
                (_compute_per_dialogue_with_components(g) for g in grouped),
                total=n_dialogues,
                desc="Computing CCI with components (single-threaded)",
            )
        )

    speaker_pair_results = []

    for group, result in zip(grouped, results):
        dialogue_id = group["dialogue_id"].iloc[0]
        speakers = sorted(list(set(group["speaker"].tolist())))

        for j_idx, from_speaker in enumerate(speakers):
            for i_idx, to_speaker in enumerate(speakers):
                if j_idx != i_idx:
                    data = result[(j_idx, i_idx)]

                    speaker_pair_results.append(
                        {
                            "dialogue_id": dialogue_id,
                            "from_speaker": from_speaker,
                            "to_speaker": to_speaker,
                            "CCI_score": data["cci_score"],
                            "n_interactions": data["n_interactions"],
                            "D_mean": data["D_mean"],
                            "D_std": data["D_std"],
                            "I_mean": data["I_mean"],
                            "I_std": data["I_std"],
                            "S_mean": data["S_mean"],
                            "S_std": data["S_std"],
                        }
                    )

    result_df = pd.DataFrame(speaker_pair_results)
    return result_df.sort_values(
        ["dialogue_id", "CCI_score"], ascending=[True, False]
    ).reset_index(drop=True)
