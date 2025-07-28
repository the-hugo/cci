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

import cci

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Pickled DataFrame with dialogue data")
    p.add_argument("--output", default="cci_scores.csv",
                   help="Destination for CSV result")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override ALPHA weighting")
    return p.parse_args()

def main():
    args = parse_args()
    if args.alpha is not None:
        importlib.reload(cci.config)
        cci.config.ALPHA = args.alpha

    df = cci.io.load_dialogues(args.input)
    scores = cci.pipeline.compute_cci(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"Wrote CCI scores to {args.output}")

if __name__ == "__main__":
    main()