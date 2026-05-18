"""Download a hand-picked list of additional notebooks for round-2 analysis.

Usage:
    uv run python scripts/download_specific_notebooks.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "notebooks"

# Round-2 targets: 0.86 LB notebooks + novel approaches we missed in round 1
REFS = [
    # === 0.86 LB (the user's explicit ask) ===
    "mohamedamr992/0-86-adapter-packaging-workflow",
    "afr1ste/nemotron-0-86-tinker-adapter-guide",
    "safar1/lb-score-0-86",
    # === 0.87 speculation ===
    "johnjanson/agi-for-medal-0-87-is-possible",
    # === Novel training methods ===
    "pkuszboi/0-85-lb-training-with-muon",       # Muon optimizer
    "waterjoe/nvidia-nemotron-mask-loss",         # Loss masking (gap in our round-1 set)
    "newduck/nvidia-nemotron-soft-balanced-sampling-sft",  # Soft-balanced sampling
    "hammadfarooq470/think-twice-self-correcting-reasoning",  # Self-correcting CoT
    "amanatar/nemotron-ultimate-sft-grpo-v3",     # SFT+GRPO v3
    "rauffauzanrambe/lora-adapter-fusion-and-rank-compression-pipeline",  # Adapter fusion
    # === Local CV (we explicitly want this) ===
    "jiazhuang/nemotron-local-cv",
    # === Per-puzzle-type solvers / data pipelines ===
    "kalyankkr/all-6-puzzle-types-decoded-sft-training-data",
    "mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers",
    "konbu17/bit-manipulation-solver-cot-generator",
    "kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset",
    "konbu17/nemotron-tong-style-cot-sft-updated-v2",  # konbu17 v2 follow-up
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def download(ref: str, dest: Path) -> None:
    if dest.exists():
        print(f"  · skip (already present): {dest.name}")
        return
    tmp = dest.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    run(["kaggle", "kernels", "pull", ref, "-p", str(tmp), "-m"])
    tmp.rename(dest)
    print(f"  ✓ downloaded: {dest.name}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(REFS)} round-2 notebooks into {OUT_DIR.relative_to(ROOT)}/ ...")
    for i, ref in enumerate(REFS, 1):
        dest = OUT_DIR / ref.replace("/", "__")
        print(f"[{i:>2}/{len(REFS)}] {ref}")
        try:
            download(ref, dest)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ failed: {e.stderr.strip() or e.stdout.strip()}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
