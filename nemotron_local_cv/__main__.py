"""CLI: run a LoRA adapter against a cached val split and print accuracy + per-label.

Usage (on a GPU host):
    uv run python -m nemotron_local_cv \\
        --adapter /path/to/adapter_dir \\
        --val data/val_split_600.csv \\
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        [--debug]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import host_metric


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="Path to a directory containing adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--val", required=True, help="Path to a val CSV with cols id, prompt, answer, label")
    ap.add_argument("--model", required=True, help="HF model id or local path for the base model")
    ap.add_argument("--max-tokens", type=int, default=7680)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--debug", action="store_true", help="Write debug_predictions.csv with raw outputs")
    args = ap.parse_args()

    val_df = pd.read_csv(args.val)
    required_cols = {"id", "prompt", "answer", "label"}
    missing = required_cols - set(val_df.columns)
    if missing:
        sys.exit(f"val CSV is missing columns: {missing}")

    # host_metric.score reads test.csv from DATA_PATH and joins on `solution[row_id_column_name]`.
    # We mirror that by staging the val rows as `test.csv` in a temp dir.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        val_df.to_csv(tmp_path / "test.csv", index=False)
        host_metric.configure(model_path=args.model, data_path=tmp_path)

        solution = val_df[["id", "answer"]]
        submission_stub = pd.DataFrame([{"prediction": args.adapter}])

        accuracy = host_metric.score(
            solution=solution,
            submission=submission_stub,
            row_id_column_name="id",
            max_lora_rank=args.max_lora_rank,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            debug=args.debug,
        )

        # Per-label breakdown — read back the debug CSV if present, otherwise re-run
        # verify against the merged predictions we just produced (debug=True writes
        # debug_predictions.csv into the CWD).
        debug_path = Path("debug_predictions.csv")
        if args.debug and debug_path.exists():
            pred_df = pd.read_csv(debug_path, dtype="str").fillna({"extracted_prediction": "NOT_FOUND"})
            merge = val_df.astype({"id": "str"}).merge(pred_df.astype({"id": "str"}), on="id", how="left")
            merge["is_correct"] = merge.apply(
                lambda r: host_metric.verify(str(r.answer), str(r.extracted_prediction)),
                axis=1,
            )
            print()
            print(f"=== overall accuracy: {merge.is_correct.mean():.4f} ===")
            print()
            print("per-label accuracy:")
            print(merge.groupby("label").is_correct.agg(["mean", "count"]).to_string())
            # Stash a copy alongside the debug CSV for archival.
            archive = debug_path.with_name("debug_predictions_with_labels.csv")
            merge.to_csv(archive, index=False)
            print(f"\nwrote {archive}")
        else:
            print(f"\n=== overall accuracy: {accuracy:.4f} ===")
            print("(re-run with --debug for per-label breakdown)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
