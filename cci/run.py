"""
CLI for batch CCI computation

Usage
-----
python -m cci.cli.run \
    --input dialogues.pkl \
    --output scores.csv \
    --alpha 0.75
"""
import argparse
import importlib
import pandas as pd
from pathlib import Path
import time

import cci

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Pickled DataFrame with dialogue data")
    p.add_argument("--output", default="data/cci_scores.csv",
                   help="Destination for CSV result")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override ALPHA weighting")
    return p.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    
    print("=" * 60)
    print("Creative Convergence Index (CCI) Computation")
    print("=" * 60)
    
    if args.alpha is not None:
        print(f"Using custom alpha value: {args.alpha}")
        importlib.reload(cci.config)
        cci.config.ALPHA = args.alpha
    else:
        print(f"Using default alpha value: {cci.config.ALPHA}")
    
    print()

    df = cci.i_o.load_dialogues(args.input)
    print()
    
    result_df = cci.pipeline.compute_cci(df)
    print()
    
    print("Saving results...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save speaker pair results to CSV
    result_df.to_csv(args.output, index=False)
    
    elapsed = time.time() - start_time
    print(f"SUCCESS: Results saved to {args.output}")
    print(f"SUCCESS: Processing completed in {elapsed:.1f} seconds")
    print()
    print("Summary:")
    print(f"  - Processed {result_df['dialogue_id'].nunique():,} dialogues")
    print(f"  - Total speaker pairs: {len(result_df):,}")
    
    if len(result_df) > 0:
        # Show top interactions
        top_scores = result_df[result_df['CCI_score'] > 0].head(3)
        print(f"  - Top CCI interactions:")
        for _, row in top_scores.iterrows():
            print(f"    {row['from_speaker']} → {row['to_speaker']}: {row['CCI_score']:.4f} (dialogue {row['dialogue_id']}, {row['n_interactions']} interactions)")
        
        # Show interaction distribution
        non_zero = result_df[result_df['CCI_score'] != 0]
        if len(non_zero) > 0:
            print(f"  - {len(non_zero)} non-zero interactions out of {len(result_df)} total pairs")
        else:
            print(f"  - No meaningful interactions found")
    print("=" * 60)

if __name__ == "__main__":
    main()