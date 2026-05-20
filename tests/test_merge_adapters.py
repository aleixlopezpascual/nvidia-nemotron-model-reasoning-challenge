"""Tests for scripts/merge_adapters.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from scripts.merge_adapters import merge_adapters


def _write_adapter(dir_path: Path, tensors: dict[str, np.ndarray], *, rank: int = 32) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model", "r": rank}) + "\n"
    )


def test_merge_adapters_full_soup_blends_overlap(tmp_path):
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    key = "base_model.model.layers.0.q_proj.lora_A.weight"
    keep = "base_model.model.layers.0.o_proj.lora_A.weight"

    _write_adapter(
        base_dir,
        {
            key: np.array([[1.0, 3.0]], dtype=np.float32),
            keep: np.array([[9.0, 9.0]], dtype=np.float32),
        },
    )
    _write_adapter(
        donor_dir,
        {
            key: np.array([[5.0, 7.0]], dtype=np.float32),
            keep: np.array([[1.0, 1.0]], dtype=np.float32),
        },
    )

    out = merge_adapters(
        base_dir=base_dir,
        donor_dir=donor_dir,
        output_dir=tmp_path / "merged",
        donor_weight=0.25,
    )

    merged = load_file(str(out / "adapter_model.safetensors"))
    np.testing.assert_allclose(merged[key], np.array([[2.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(merged[keep], np.array([[7.0, 7.0]], dtype=np.float32))

    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["merge_metadata"]["donor_weight"] == 0.25
    assert cfg["merge_metadata"]["merged_tensor_count"] == 2


def test_merge_adapters_module_filter_only_changes_selected_keys(tmp_path):
    base_dir = tmp_path / "base"
    donor_dir = tmp_path / "donor"
    q_key = "base_model.model.layers.0.q_proj.lora_A.weight"
    m_key = "base_model.model.layers.0.in_proj.lora_A.weight"

    _write_adapter(
        base_dir,
        {
            q_key: np.array([[2.0, 2.0]], dtype=np.float32),
            m_key: np.array([[4.0, 4.0]], dtype=np.float32),
        },
    )
    _write_adapter(
        donor_dir,
        {
            q_key: np.array([[6.0, 6.0]], dtype=np.float32),
            m_key: np.array([[10.0, 10.0]], dtype=np.float32),
        },
    )

    out = merge_adapters(
        base_dir=base_dir,
        donor_dir=donor_dir,
        output_dir=tmp_path / "merged",
        donor_weight=0.5,
        include_modules=("q_proj",),
    )

    merged = load_file(str(out / "adapter_model.safetensors"))
    np.testing.assert_allclose(merged[q_key], np.array([[4.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(merged[m_key], np.array([[4.0, 4.0]], dtype=np.float32))

    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["merge_metadata"]["include_modules"] == ["q_proj"]
    assert cfg["merge_metadata"]["merged_tensor_count"] == 1


def test_merge_adapters_uses_final_dir_fallback(tmp_path):
    base_root = tmp_path / "base"
    donor_root = tmp_path / "donor"
    key = "base_model.model.layers.0.q_proj.lora_A.weight"
    _write_adapter(base_root / "final", {key: np.array([[1.0]], dtype=np.float32)})
    _write_adapter(donor_root / "final", {key: np.array([[3.0]], dtype=np.float32)})

    out = merge_adapters(
        base_dir=base_root,
        donor_dir=donor_root,
        output_dir=tmp_path / "merged",
        donor_weight=0.5,
    )
    merged = load_file(str(out / "adapter_model.safetensors"))
    np.testing.assert_allclose(merged[key], np.array([[2.0]], dtype=np.float32))
