"""Tests for scripts/materialize_merge_experiments.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from scripts.materialize_merge_experiments import materialize_merge_experiments


def _write_adapter(dir_path: Path, tensors: dict[str, np.ndarray]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model", "r": 32}) + "\n"
    )


def test_materialize_merge_experiments_writes_manifest_and_outputs(tmp_path):
    key = "base_model.model.layers.0.q_proj.lora_A.weight"
    base = tmp_path / "base"
    donor = tmp_path / "donor"
    _write_adapter(base, {key: np.array([[2.0]], dtype=np.float32)})
    _write_adapter(donor, {key: np.array([[6.0]], dtype=np.float32)})

    experiments = tmp_path / "experiments.csv"
    with experiments.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "base_adapter",
                "donor_adapter",
                "donor_weight",
                "include_modules",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "test_merge",
                "base_adapter": str(base),
                "donor_adapter": str(donor),
                "donor_weight": "0.5",
                "include_modules": "q_proj",
            }
        )

    out_root, manifest = materialize_merge_experiments(
        experiments_csv=experiments,
        output_root=tmp_path / "merged",
        manifest_out=tmp_path / "manifest.csv",
    )

    rows = list(csv.DictReader(manifest.open()))
    assert rows == [{"name": "test_merge", "adapter": str(out_root / "test_merge")}]

    merged = load_file(str(out_root / "test_merge" / "adapter_model.safetensors"))
    np.testing.assert_allclose(merged[key], np.array([[4.0]], dtype=np.float32))
