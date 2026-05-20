"""Build a deterministic submission.zip from a local adapter directory.

Usage:
    uv run python scripts/package_submission.py \
        --adapter-dir /path/to/adapter \
        --output submission.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from nemotron_train.recipe import BASE_MODEL

REQUIRED = ("adapter_config.json", "adapter_model.safetensors")


def _resolve_adapter_dir(adapter_dir: Path) -> Path:
    missing = [name for name in REQUIRED if not (adapter_dir / name).is_file()]
    if not missing:
        return adapter_dir

    final_dir = adapter_dir / "final"
    if final_dir.is_dir():
        final_missing = [name for name in REQUIRED if not (final_dir / name).is_file()]
        if not final_missing:
            return final_dir

    raise FileNotFoundError(
        f"missing required adapter files in {adapter_dir}: {missing}"
    )


def build_submission_zip(
    *,
    adapter_dir: str | Path,
    output_zip: str | Path,
    rewrite_base_model: bool = True,
    base_model: str = BASE_MODEL,
) -> Path:
    source = _resolve_adapter_dir(Path(adapter_dir))
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(source / "adapter_model.safetensors", stage / "adapter_model.safetensors")

        cfg_path = source / "adapter_config.json"
        if rewrite_base_model:
            cfg = json.loads(cfg_path.read_text())
            cfg["base_model_name_or_path"] = base_model
            (stage / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        else:
            shutil.copy2(cfg_path, stage / "adapter_config.json")

        with zipfile.ZipFile(
            output,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as zf:
            for name in REQUIRED:
                zf.write(stage / name, arcname=name)

    with zipfile.ZipFile(output) as zf:
        names = sorted(zf.namelist())
    if names != sorted(REQUIRED):
        raise RuntimeError(
            f"submission.zip must contain only {list(REQUIRED)} at root; got {names}"
        )

    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", required=True, help="Directory with adapter files")
    ap.add_argument("--output", default="submission.zip", help="Path to output zip")
    ap.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model name to write to adapter_config.json",
    )
    ap.add_argument(
        "--keep-base-model",
        action="store_true",
        help="Do not rewrite base_model_name_or_path",
    )
    args = ap.parse_args()

    out = build_submission_zip(
        adapter_dir=args.adapter_dir,
        output_zip=args.output,
        rewrite_base_model=not args.keep_base_model,
        base_model=args.base_model,
    )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
