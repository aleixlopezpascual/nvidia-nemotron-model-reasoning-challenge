"""Materialize merged adapters from a CSV experiment plan.

Input CSV columns:
    name, base_adapter, donor_adapter, donor_weight, include_modules

`include_modules` is optional and comma-separated.
Outputs:
    - one merged adapter directory per row under --output-root
    - a manifest CSV (name,adapter) suitable for `python -m nemotron_local_cv --adapter-manifest`
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from scripts.merge_adapters import merge_adapters
except ModuleNotFoundError:
    from merge_adapters import merge_adapters


def materialize_merge_experiments(
    *,
    experiments_csv: str | Path,
    output_root: str | Path,
    manifest_out: str | Path,
) -> tuple[Path, Path]:
    experiments_path = Path(experiments_csv)
    output_root = Path(output_root)
    manifest_path = Path(manifest_out)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with experiments_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    manifest_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    for row in rows:
        name = row["name"].strip()
        include_modules = tuple(
            m.strip() for m in (row.get("include_modules") or "").split(",") if m.strip()
        )
        out_dir = output_root / name
        try:
            merge_adapters(
                base_dir=row["base_adapter"],
                donor_dir=row["donor_adapter"],
                output_dir=out_dir,
                donor_weight=float(row["donor_weight"]),
                include_modules=include_modules,
            )
        except Exception as exc:
            report_rows.append(
                {
                    "name": name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        manifest_rows.append({"name": name, "adapter": str(out_dir)})
        report_rows.append({"name": name, "status": "ok", "error": ""})

    if not manifest_rows:
        raise RuntimeError("no merge experiments completed successfully")

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "adapter"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    report_path = manifest_path.with_name(f"{manifest_path.stem}_report.csv")
    with report_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "status", "error"])
        writer.writeheader()
        writer.writerows(report_rows)

    return output_root, manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiments-csv",
        default="data/merge_experiments.csv",
        help="CSV describing merge experiments",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        help="Directory where merged adapter folders will be written",
    )
    ap.add_argument(
        "--manifest-out",
        required=True,
        help="Output manifest CSV for downstream batch evaluation",
    )
    args = ap.parse_args()

    out_root, manifest = materialize_merge_experiments(
        experiments_csv=args.experiments_csv,
        output_root=args.output_root,
        manifest_out=args.manifest_out,
    )
    print(f"wrote merged adapters under {out_root}")
    print(f"wrote manifest {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
