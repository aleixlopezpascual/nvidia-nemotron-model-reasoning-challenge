"""Tests for scripts/package_submission.py."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.package_submission import build_submission_zip


def _write_adapter(dir_path: Path, *, base_model: str = "old/base") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "r": 32}) + "\n"
    )


def test_build_submission_zip_rewrites_base_and_is_flat(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, base_model="legacy/model")

    output = tmp_path / "submission.zip"
    build_submission_zip(
        adapter_dir=adapter_dir,
        output_zip=output,
        base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )

    with zipfile.ZipFile(output) as zf:
        assert sorted(zf.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        cfg = json.loads(zf.read("adapter_config.json").decode("utf-8"))
        assert (
            cfg["base_model_name_or_path"]
            == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        )


def test_build_submission_zip_uses_final_subdir_fallback(tmp_path):
    adapter_root = tmp_path / "adapter_root"
    _write_adapter(adapter_root / "final")

    output = tmp_path / "submission.zip"
    build_submission_zip(adapter_dir=adapter_root, output_zip=output)

    with zipfile.ZipFile(output) as zf:
        assert "adapter_config.json" in zf.namelist()
        assert "adapter_model.safetensors" in zf.namelist()


def test_build_submission_zip_can_keep_original_base_model(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, base_model="custom/base")

    output = tmp_path / "submission.zip"
    build_submission_zip(
        adapter_dir=adapter_dir,
        output_zip=output,
        rewrite_base_model=False,
    )

    with zipfile.ZipFile(output) as zf:
        cfg = json.loads(zf.read("adapter_config.json").decode("utf-8"))
    assert cfg["base_model_name_or_path"] == "custom/base"
