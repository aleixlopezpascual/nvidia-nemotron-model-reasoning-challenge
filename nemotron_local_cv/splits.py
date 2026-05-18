"""Stratified validation-split builder for local CV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labels import PUZZLE_TYPES, classify_puzzle_type


def build_stratified_val_split(
    train_csv: str | Path = "train.csv",
    n: int = 600,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Stratified-by-puzzle-type val sample from train.csv.

    Args:
        train_csv: path to the competition train.csv
        n: target total val size; per-type quota is n // 6 (rounded down).
        seed: pandas sample seed
        out_path: if set, write the val CSV (cols id, prompt, answer, label)

    Returns:
        DataFrame with columns id, prompt, answer, label. Index is reset.
    """
    df = pd.read_csv(train_csv)
    df["label"] = df["prompt"].apply(classify_puzzle_type)

    unknown = (df["label"] == "unknown").sum()
    if unknown:
        raise ValueError(
            f"{unknown} prompts did not match any puzzle-type lead. "
            "Inspect labels.py — the prompt format may have shifted."
        )

    per_type = n // len(PUZZLE_TYPES)
    parts = [
        df[df["label"] == label].sample(n=per_type, random_state=seed)
        for label in PUZZLE_TYPES
    ]
    val = pd.concat(parts).sort_values("id").reset_index(drop=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        val.to_csv(out_path, index=False)
    return val
