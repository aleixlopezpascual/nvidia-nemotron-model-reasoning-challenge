"""Tests for the 6-way puzzle-type classifier.

The classifier is leading-clause-match against deterministic prompt prefixes.
The strong invariant is: every row in the real train.csv must classify to a
known puzzle type — if any row returns 'unknown', either the prompt format
shifted or our lead table is stale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nemotron_local_cv.labels import (
    PUZZLE_TYPES,
    classify_puzzle_type,
)

TRAIN_CSV = Path(__file__).resolve().parents[1] / "train.csv"


class TestUnitClassification:
    def test_known_leads_each_resolve(self):
        cases = [
            ("In Alice's Wonderland, a secret bit manipulation rule says ...", "bit_manipulation"),
            ("In Alice's Wonderland, the gravitational constant is ...", "gravitational_constant"),
            ("In Alice's Wonderland, a secret unit conversion ...", "unit_conversion"),
            ("In Alice's Wonderland, secret encryption rules ...", "text_encryption"),
            ("In Alice's Wonderland, numbers are secretly converted ...", "numeral_conversion"),
            ("In Alice's Wonderland, a secret set of transformation rules is applied to equations ...", "equation_transformation"),
        ]
        for prompt, expected in cases:
            assert classify_puzzle_type(prompt) == expected, f"misclassified: {prompt!r}"

    def test_unknown_prompt_returns_unknown(self):
        assert classify_puzzle_type("Once upon a time...") == "unknown"

    def test_empty_prompt_returns_unknown(self):
        assert classify_puzzle_type("") == "unknown"

    def test_puzzle_types_tuple_is_six(self):
        # The val-split builder hardcodes n // 6 per type — if this grows we
        # need to revisit the split-builder math.
        assert len(PUZZLE_TYPES) == 6
        assert len(set(PUZZLE_TYPES)) == 6


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not unzipped")
class TestAgainstFullTrainCsv:
    @pytest.fixture(scope="class")
    def train_df(self):
        return pd.read_csv(TRAIN_CSV)

    def test_no_unknown_rows(self, train_df):
        labels = train_df["prompt"].apply(classify_puzzle_type)
        unknown = (labels == "unknown").sum()
        assert unknown == 0, f"{unknown} prompts did not match any lead"

    def test_every_puzzle_type_present(self, train_df):
        labels = set(train_df["prompt"].apply(classify_puzzle_type))
        assert set(PUZZLE_TYPES) <= labels

    def test_distribution_roughly_balanced(self, train_df):
        # Empirical: ~1555-1602 per type out of 9500. Any class < 1000 or
        # > 2500 would be a real signal of corpus drift.
        counts = train_df["prompt"].apply(classify_puzzle_type).value_counts()
        assert counts.min() > 1000, f"too few of some type: {counts.to_dict()}"
        assert counts.max() < 2500, f"too many of some type: {counts.to_dict()}"
