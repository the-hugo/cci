# ---------------------------------------------------------------------
# High‑level orchestration: from raw DataFrame → CCI per dialogue.
# ---------------------------------------------------------------------
from __future__ import annotations
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from .concepts import extract_concepts
from .metrics import (
    divergence, novel_set, grounded_set,
    incorporation, shared_growth, cc_turn
)
from .config import EPS, ALPHA, N_PROCESSES, BATCH_SIZE

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
    
    batches = [df["text"].iloc[i:i+BATCH_SIZE].tolist() for i in range(0, n, BATCH_SIZE)]
    
    with mp.Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(_extract_concepts_worker, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    concepts_flat = [c for batch in results for c in batch]
    df = df.copy()
    df["concepts"] = concepts_flat
    print("SUCCESS: Concept extraction complete")
    return df

# --------------------------------------------------  core computation
def _compute_per_dialogue(dialogue: pd.DataFrame) -> np.ndarray:
    """
    Return k×k CCI matrix where C[j,i] is the CCI score of speaker j building upon speaker i.
    
    For each speaker pair (j, i), calculates how speaker j builds upon speaker i's
    contributions by looking at a window of up to 4 previous turns from speaker i
    when speaker j speaks.
    
    Returns:
        np.ndarray: k×k matrix where k is the number of unique speakers
                   C[j,i] = CCI score of speaker j relative to speaker i
    """
    dialogue = dialogue.reset_index(drop=True)
    n_turns = len(dialogue)
    if n_turns < 5:  # Need at least 5 turns for meaningful windowed calculation
        return np.array([[0.0]])

    embeddings = np.stack(dialogue["embedding"].to_numpy(), axis=0)
    concepts = dialogue["concepts"].tolist()
    speakers = dialogue["speaker"].tolist()

    # Get unique speakers and create mapping
    unique_speakers = sorted(list(set(speakers)))
    k = len(unique_speakers)
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    
    # Initialize k×k CCI matrix
    cci_matrix = np.zeros((k, k))
    
    # For each speaker pair (j, i), calculate CCI score of j building upon i
    for j_speaker in unique_speakers:
        j_idx = speaker_to_idx[j_speaker]
        
        for i_speaker in unique_speakers:
            if j_speaker == i_speaker:
                continue  # Skip self-interactions
                
            i_idx = speaker_to_idx[i_speaker]
            cc_scores = []
            
            # Find all turns by speaker j
            for t in range(4, n_turns):
                if speakers[t] != j_speaker:
                    continue
                    
                current_embedding = embeddings[t]
                current_concepts = concepts[t]
                
                # Find window of previous 4 turns from speaker i only
                window_start = max(0, t - 4)
                i_window_turns = []
                
                for idx in range(t - 1, window_start - 1, -1):  # Go backwards from t-1
                    if speakers[idx] == i_speaker:
                        i_window_turns.append(idx)
                    if len(i_window_turns) >= 4:  # Limit to 4 turns from speaker i
                        break
                
                if len(i_window_turns) == 0:
                    continue  # Skip if no speaker i turns in window
                    
                # Reverse to get chronological order
                i_window_turns.reverse()
                
                # Get most recent speaker i turn for divergence calculation
                most_recent_i_idx = i_window_turns[-1]
                i_embedding = embeddings[most_recent_i_idx]
                D_t = divergence(current_embedding, i_embedding)
                
                # Collect concepts from speaker i's windowed turns
                i_windowed_concepts = set()
                for idx in i_window_turns:
                    i_windowed_concepts.update(concepts[idx])
                
                # Calculate novel concepts (in j's turn but not in i's windowed turns)
                n_t = current_concepts - i_windowed_concepts
                
                # Calculate grounded concepts (intersection with i's windowed concepts)
                g_t = current_concepts & i_windowed_concepts
                
                # Look ahead for incorporation - find next turns by speaker i
                future_i_concepts = set()
                look_ahead_limit = min(n_turns, t + 5)  # Look ahead up to 4 turns
                for future_idx in range(t + 1, look_ahead_limit):
                    if speakers[future_idx] == i_speaker:
                        future_i_concepts.update(concepts[future_idx])
                
                # Calculate incorporation and shared growth
                I_tp1 = incorporation(n_t, future_i_concepts)
                
                # For shared growth, compare grounding before and after
                union_context = i_windowed_concepts | current_concepts
                S_tp1 = shared_growth(g_t, future_i_concepts & current_concepts, union_context)
                
                cc_scores.append(cc_turn(D_t, I_tp1, S_tp1))
            
            # Set matrix entry C[j,i] = average CCI score of j building upon i
            if cc_scores:
                cci_matrix[j_idx, i_idx] = float(np.mean(cc_scores))
    
    return cci_matrix

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
    
    with mp.Pool(processes=N_PROCESSES) as pool:
        cci_matrices = list(tqdm(
            pool.imap(_compute_per_dialogue, grouped),
            total=n_dialogues,
            desc="Computing CCI"
        ))

    # Flatten matrices into speaker pair results
    speaker_pair_results = []
    all_cci_scores = []
    
    for group, matrix in zip(grouped, cci_matrices):
        dialogue_id = group["dialogue_id"].iloc[0]
        speakers = sorted(list(set(group["speaker"].tolist())))
        
        # Extract speaker pair scores from matrix
        for j_idx, from_speaker in enumerate(speakers):
            for i_idx, to_speaker in enumerate(speakers):
                if j_idx != i_idx:  # Skip self-interactions
                    cci_score = matrix[j_idx, i_idx]
                    
                    # Count number of interactions for this speaker pair
                    n_interactions = 0
                    for t in range(4, len(group)):
                        if group.iloc[t]['speaker'] == from_speaker:
                            # Check if to_speaker appears in the 4-turn window
                            window_start = max(0, t - 4)
                            for idx in range(t - 1, window_start - 1, -1):
                                if group.iloc[idx]['speaker'] == to_speaker:
                                    n_interactions += 1
                                    break  # Count each from_speaker turn only once
                    
                    speaker_pair_results.append({
                        'dialogue_id': dialogue_id,
                        'from_speaker': from_speaker,
                        'to_speaker': to_speaker,
                        'CCI_score': float(cci_score),
                        'n_interactions': n_interactions
                    })
                    
                    if cci_score != 0:  # Only include non-zero scores in statistics
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
    
    return result_df.sort_values(['dialogue_id', 'CCI_score'], ascending=[True, False]).reset_index(drop=True)