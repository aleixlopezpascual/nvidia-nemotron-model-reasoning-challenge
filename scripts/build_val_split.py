"""One-shot script to build a stratified val split from train.csv.

Usage:
    uv run python scripts/build_val_split.py [--n 600] [--seed 42] [--out data/val_split_600.csv]

The output CSV (cols id, prompt, answer, label) is small and gets checked into the
repo for reproducibility of every CV run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nemotron_local_cv.splits import build_stratified_val_split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(ROOT / "train.csv"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(ROOT / "data" / "val_split_600.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    val = build_stratified_val_split(
        train_csv=args.train, n=args.n, seed=args.seed, out_path=args.out
    )
    print(f"Wrote {len(val)} rows to {args.out}")
    print()
    print("per-label counts:")
    print(val.groupby("label").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
