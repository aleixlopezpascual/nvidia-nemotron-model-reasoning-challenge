"""Merge two compatible LoRA adapters without training.

Usage:
    uv run python scripts/merge_adapters.py \
        --base-dir /path/to/base_adapter \
        --donor-dir /path/to/donor_adapter \
        --output-dir /path/to/merged_adapter \
        --donor-weight 0.1 \
        [--include-module q_proj --include-module k_proj]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

try:
    from scripts.package_submission import REQUIRED, resolve_adapter_dir
except ModuleNotFoundError:
    from package_submission import REQUIRED, resolve_adapter_dir


def _matches_modules(key: str, include_modules: tuple[str, ...]) -> bool:
    return any(f".{module}." in key for module in include_modules)


def merge_adapters(
    *,
    base_dir: str | Path,
    donor_dir: str | Path,
    output_dir: str | Path,
    donor_weight: float,
    include_modules: tuple[str, ...] = (),
) -> Path:
    if not 0.0 <= donor_weight <= 1.0:
        raise ValueError("donor_weight must be within [0.0, 1.0]")

    base_root = resolve_adapter_dir(Path(base_dir))
    donor_root = resolve_adapter_dir(Path(donor_dir))
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads((base_root / "adapter_config.json").read_text())
    donor_cfg = json.loads((donor_root / "adapter_config.json").read_text())
    if base_cfg.get("r") != donor_cfg.get("r"):
        raise ValueError(
            f"adapter rank mismatch: base r={base_cfg.get('r')} donor r={donor_cfg.get('r')}"
        )

    base_tensors = load_file(str(base_root / "adapter_model.safetensors"))
    donor_tensors = load_file(str(donor_root / "adapter_model.safetensors"))

    merged: dict[str, np.ndarray] = {}
    selected_overlap = 0
    for key, base_tensor in base_tensors.items():
        should_merge = key in donor_tensors and (
            not include_modules or _matches_modules(key, include_modules)
        )
        if should_merge:
            donor_tensor = donor_tensors[key]
            if base_tensor.shape != donor_tensor.shape:
                raise ValueError(
                    f"shape mismatch for {key}: {base_tensor.shape} vs {donor_tensor.shape}"
                )
            merged[key] = ((1.0 - donor_weight) * base_tensor) + (
                donor_weight * donor_tensor
            )
            selected_overlap += 1
        else:
            merged[key] = base_tensor

    if include_modules and selected_overlap == 0:
        raise ValueError(
            f"no overlapping tensors matched include_modules={list(include_modules)}"
        )

    save_file(merged, str(out_root / "adapter_model.safetensors"))

    merge_note = {
        "base_adapter_dir": str(base_root),
        "donor_adapter_dir": str(donor_root),
        "donor_weight": donor_weight,
        "include_modules": list(include_modules),
        "merged_tensor_count": selected_overlap,
    }
    base_cfg["merge_metadata"] = merge_note
    (out_root / "adapter_config.json").write_text(json.dumps(base_cfg, indent=2) + "\n")

    for name in REQUIRED:
        if not (out_root / name).exists():
            raise RuntimeError(f"missing merged output file: {name}")

    return out_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, help="Base adapter directory")
    ap.add_argument("--donor-dir", required=True, help="Donor adapter directory")
    ap.add_argument("--output-dir", required=True, help="Directory for merged adapter")
    ap.add_argument(
        "--donor-weight",
        required=True,
        type=float,
        help="Interpolation weight for donor tensors in [0, 1]",
    )
    ap.add_argument(
        "--include-module",
        action="append",
        default=[],
        help="Restrict merging to tensor names containing .<module>.",
    )
    args = ap.parse_args()

    out = merge_adapters(
        base_dir=args.base_dir,
        donor_dir=args.donor_dir,
        output_dir=args.output_dir,
        donor_weight=args.donor_weight,
        include_modules=tuple(args.include_module),
    )
    print(f"wrote merged adapter to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
