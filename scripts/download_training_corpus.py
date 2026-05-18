"""Download the kienngx CoT training corpus (Apache-2.0).

Usage:
    uv run python scripts/download_training_corpus.py [--out data/]

Saves ~13 MB CSV (`final_Nemotron_training_data.csv`) to the target dir.
Schema: id, prompt, answer, generated_cot, label (7 categories, 9500 rows).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "kienngx/nemotron-30b-competition-trainingdata-cot-labels"
ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(ROOT / "data"),
        help="Output directory (defaults to ./data/)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "final_Nemotron_training_data.csv"
    if target.exists():
        print(f"skip: {target} already exists ({target.stat().st_size / 1024 / 1024:.1f} MB)")
        return 0

    print(f"downloading {DATASET} -> {out}")
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "-p", str(out), "--unzip"],
        check=True,
    )
    if target.exists():
        print(f"done: {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("WARN: expected CSV not found after unzip; check the dataset structure")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
