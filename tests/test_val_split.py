"""Tests for the stratified val split.

Two layers:
  1. The checked-in data/val_split_600.csv is the artifact downstream training
     and CV consume; its shape, label distribution, and uniqueness must hold.
  2. The builder is deterministic with seed=42 — if the train.csv ever shifts
     the builder will produce a different split, but for a fixed corpus the
     same seed must yield the same rows in the same order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nemotron_local_cv.labels import PUZZLE_TYPES
from nemotron_local_cv.splits import build_stratified_val_split

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = REPO_ROOT / "data" / "val_split_600.csv"
TRAIN_CSV = REPO_ROOT / "train.csv"


class TestCheckedInValSplit:
    @pytest.fixture(scope="class")
    def val_df(self):
        if not VAL_CSV.exists():
            pytest.skip("data/val_split_600.csv missing")
        return pd.read_csv(VAL_CSV)

    def test_required_columns(self, val_df):
        assert set(val_df.columns) >= {"id", "prompt", "answer", "label"}

    def test_row_count(self, val_df):
        assert len(val_df) == 600

    def test_ids_unique(self, val_df):
        assert val_df["id"].is_unique

    def test_per_label_count(self, val_df):
        counts = val_df["label"].value_counts().to_dict()
        for label in PUZZLE_TYPES:
            assert counts.get(label) == 100, f"{label}: {counts.get(label)}"

    def test_no_unknown_label(self, val_df):
        assert "unknown" not in set(val_df["label"])

    def test_prompt_and_answer_non_null(self, val_df):
        assert val_df["prompt"].notna().all()
        assert val_df["answer"].notna().all()
        assert (val_df["prompt"].str.len() > 0).all()


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not unzipped")
class TestBuilderDeterminism:
    def test_seed_42_matches_checked_in_split(self):
        # Rebuilding from the same train.csv + same seed must produce the
        # exact same row set as the checked-in artifact (modulo nothing).
        if not VAL_CSV.exists():
            pytest.skip("data/val_split_600.csv missing")
        rebuilt = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=42)
        on_disk = pd.read_csv(VAL_CSV)
        assert sorted(rebuilt["id"].tolist()) == sorted(on_disk["id"].tolist())

    def test_two_runs_same_seed_match(self):
        a = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=42)
        b = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=42)
        assert a["id"].tolist() == b["id"].tolist()

    def test_different_seed_yields_different_split(self):
        a = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=42)
        b = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=7)
        # Overwhelmingly likely to differ; if they don't, the seed is being ignored.
        assert a["id"].tolist() != b["id"].tolist()

    def test_per_type_quota_holds(self):
        out = build_stratified_val_split(train_csv=TRAIN_CSV, n=600, seed=42)
        counts = out["label"].value_counts().to_dict()
        for label in PUZZLE_TYPES:
            assert counts[label] == 100
