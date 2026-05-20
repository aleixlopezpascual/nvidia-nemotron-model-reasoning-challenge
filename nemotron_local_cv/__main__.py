"""CLI: evaluate one or many LoRA adapters against a cached val split.

Usage (on a GPU host):
    # Single adapter
    uv run python -m nemotron_local_cv \
        --adapter /path/to/adapter_dir \
        --val data/val_split_600.csv \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        [--debug]

    # Batch mode (manifest CSV with columns: adapter[, name])
    uv run python -m nemotron_local_cv \
        --adapter-manifest data/adapter_candidates.csv \
        --summary-out data/adapter_eval_summary.csv \
        --val data/val_split_600.csv \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import host_metric

REQUIRED_VAL_COLUMNS = {"id", "prompt", "answer", "label"}


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug or "adapter"


def _validate_val_df(val_df: pd.DataFrame) -> None:
    missing = REQUIRED_VAL_COLUMNS - set(val_df.columns)
    if missing:
        sys.exit(f"val CSV is missing columns: {missing}")


def _evaluate_adapter(
    *,
    val_df: pd.DataFrame,
    adapter_path: str,
    max_lora_rank: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    debug: bool,
) -> tuple[float, pd.DataFrame | None, pd.DataFrame | None]:
    solution = val_df[["id", "answer"]]
    submission_stub = pd.DataFrame([{"prediction": adapter_path}])

    accuracy = host_metric.score(
        solution=solution,
        submission=submission_stub,
        row_id_column_name="id",
        max_lora_rank=max_lora_rank,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        debug=debug,
    )

    debug_path = Path("debug_predictions.csv")
    if not debug or not debug_path.exists():
        return accuracy, None, None

    pred_df = pd.read_csv(debug_path, dtype="str").fillna(
        {"extracted_prediction": "NOT_FOUND"}
    )
    merged = val_df.astype({"id": "str"}).merge(
        pred_df.astype({"id": "str"}), on="id", how="left"
    )
    merged["is_correct"] = merged.apply(
        lambda r: host_metric.verify(str(r.answer), str(r.extracted_prediction)),
        axis=1,
    )
    per_label = (
        merged.groupby("label")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )
    return float(merged.is_correct.mean()), per_label, merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--adapter",
        help="Path to a directory containing adapter_config.json + adapter_model.safetensors",
    )
    mode.add_argument(
        "--adapter-manifest",
        help="CSV with columns adapter[, name] for batch evaluation",
    )
    ap.add_argument(
        "--summary-out",
        default="adapter_eval_summary.csv",
        help="Batch mode: output CSV path for ranked summary",
    )
    ap.add_argument(
        "--debug-dir",
        default="adapter_debug",
        help="Batch mode + --debug: directory for per-adapter debug CSV archives",
    )
    ap.add_argument(
        "--val",
        required=True,
        help="Path to a val CSV with cols id, prompt, answer, label",
    )
    ap.add_argument(
        "--model", required=True, help="HF model id or local path for the base model"
    )
    ap.add_argument("--max-tokens", type=int, default=7680)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument(
        "--debug", action="store_true", help="Write debug_predictions.csv with raw outputs"
    )
    args = ap.parse_args()

    val_df = pd.read_csv(args.val)
    _validate_val_df(val_df)

    # host_metric.score reads test.csv from DATA_PATH and joins on `solution[row_id_column_name]`.
    # We mirror that by staging the val rows as `test.csv` in a temp dir.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        val_df.to_csv(tmp_path / "test.csv", index=False)
        host_metric.configure(model_path=args.model, data_path=tmp_path)

        if args.adapter is not None:
            accuracy, per_label, merged = _evaluate_adapter(
                val_df=val_df,
                adapter_path=args.adapter,
                max_lora_rank=args.max_lora_rank,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                debug=args.debug,
            )

            if args.debug and per_label is not None and merged is not None:
                print()
                print(f"=== overall accuracy: {accuracy:.4f} ===")
                print()
                print("per-label accuracy:")
                print(per_label.set_index("label").to_string())
                archive = Path("debug_predictions_with_labels.csv")
                merged.to_csv(archive, index=False)
                print(f"\nwrote {archive}")
            else:
                print(f"\n=== overall accuracy: {accuracy:.4f} ===")
                print("(re-run with --debug for per-label breakdown)")
            return 0

        manifest_df = pd.read_csv(args.adapter_manifest)
        if "adapter" not in manifest_df.columns:
            sys.exit("adapter manifest must include column: adapter")
        if "name" not in manifest_df.columns:
            manifest_df["name"] = manifest_df["adapter"].astype(str).apply(
                lambda p: Path(p).name or p
            )

        summary_rows: list[dict[str, object]] = []
        per_label_rows: list[dict[str, object]] = []
        debug_root = Path(args.debug_dir)
        if args.debug:
            debug_root.mkdir(parents=True, exist_ok=True)

        total = len(manifest_df)
        for idx, item in enumerate(manifest_df.itertuples(index=False), 1):
            adapter_name = str(item.name)
            adapter_path = str(item.adapter)
            print(f"\n[{idx}/{total}] evaluating {adapter_name}")

            try:
                accuracy, per_label, merged = _evaluate_adapter(
                    val_df=val_df,
                    adapter_path=adapter_path,
                    max_lora_rank=args.max_lora_rank,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_num_seqs=args.max_num_seqs,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    debug=args.debug,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                summary_rows.append(
                    {
                        "name": adapter_name,
                        "adapter": adapter_path,
                        "overall_accuracy": float("nan"),
                        "status": "error",
                        "error": message,
                    }
                )
                print(f"    error: {message}")
                continue

            summary_rows.append(
                {
                    "name": adapter_name,
                    "adapter": adapter_path,
                    "overall_accuracy": accuracy,
                    "status": "ok",
                    "error": "",
                }
            )
            print(f"    overall accuracy: {accuracy:.4f}")

            if args.debug and per_label is not None and merged is not None:
                run_slug = _safe_slug(f"{idx:03d}_{adapter_name}")
                archive = debug_root / f"{run_slug}_debug_predictions_with_labels.csv"
                merged.to_csv(archive, index=False)

                for row in per_label.itertuples(index=False):
                    per_label_rows.append(
                        {
                            "name": adapter_name,
                            "adapter": adapter_path,
                            "label": row.label,
                            "accuracy": row.accuracy,
                            "count": int(row.count),
                        }
                    )

        summary = pd.DataFrame(summary_rows)
        if "status" in summary.columns:
            summary = summary.sort_values(
                by=["status", "overall_accuracy"],
                ascending=[True, False],
                na_position="last",
            )
        else:
            summary = summary.sort_values("overall_accuracy", ascending=False)
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

        print("\n=== ranked adapters ===")
        print(summary.to_string(index=False))
        print(f"\nwrote {summary_path}")

        if per_label_rows:
            per_label_path = summary_path.with_name(
                f"{summary_path.stem}_per_label.csv"
            )
            pd.DataFrame(per_label_rows).to_csv(per_label_path, index=False)
            print(f"wrote {per_label_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
