"""Tests for `python -m nemotron_train` argument wiring."""

from __future__ import annotations

import sys

from nemotron_train import __main__ as train_cli
from nemotron_train import recipe as train_recipe
import nemotron_train


def test_train_cli_forwards_memory_overrides(monkeypatch):
    captured = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(nemotron_train, "train", _fake_train)
    monkeypatch.setattr(train_recipe, "BASE_MODEL", "base-from-recipe")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_train",
            "--data",
            "/tmp/corpus.csv",
            "--output",
            "/tmp/out",
            "--max-steps",
            "10",
            "--max-length",
            "4096",
            "--no-packing",
        ],
    )

    rc = train_cli.main()
    assert rc == 0
    assert captured["data_csv"] == "/tmp/corpus.csv"
    assert captured["output_dir"] == "/tmp/out"
    assert captured["base_model"] == "base-from-recipe"
    assert captured["max_steps"] == 10
    assert captured["max_length"] == 4096
    assert captured["packing"] is False


def test_train_cli_forwards_distributed_overrides(monkeypatch):
    captured = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(nemotron_train, "train", _fake_train)
    monkeypatch.setattr(train_recipe, "BASE_MODEL", "base-from-recipe")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nemotron_train",
            "--data",
            "/tmp/corpus.csv",
            "--output",
            "/tmp/out",
            "--deepspeed",
            "configs/deepspeed_zero3_lora.json",
            "--per-device-train-batch-size",
            "1",
            "--grad-accum",
            "8",
            "--logging-steps",
            "5",
            "--save-strategy",
            "no",
            "--save-total-limit",
            "1",
        ],
    )

    rc = train_cli.main()
    assert rc == 0
    assert captured["base_model"] == "base-from-recipe"
    assert captured["extra_training_kwargs"] == {
        "deepspeed": "configs/deepspeed_zero3_lora.json",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "logging_steps": 5,
        "save_strategy": "no",
        "save_total_limit": 1,
    }
