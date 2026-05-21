"""CLI for SFT training. Run on a GPU instance with the heavy deps installed:

    uv pip install torch transformers peft trl datasets accelerate

    uv run python -m nemotron_train \\
        --data /path/to/final_Nemotron_training_data.csv \\
        --output /path/to/adapter_out \\
        [--max-steps 1000] [--lr 2e-4] [--sample-n 1000] [--base <model>]
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to the CoT CSV (kienngx schema: id, prompt, answer, generated_cot, label)")
    ap.add_argument("--output", required=True, help="Directory to write adapter checkpoints to")
    ap.add_argument("--base", default=None, help="Override base model id/path (default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)")
    ap.add_argument("--max-steps", type=int, default=-1, help="Cap on optimizer steps; -1 = epochs-driven")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate (default 2e-4)")
    ap.add_argument("--epochs", type=float, default=None, help="Override num_train_epochs (default 1.0)")
    ap.add_argument("--sample-n", type=int, default=None, help="Subsample N rows for quick runs")
    ap.add_argument("--max-length", type=int, default=None, help="Override SFT max_length (default from recipe)")
    ap.add_argument(
        "--deepspeed",
        default=None,
        help="Path to a DeepSpeed config JSON (useful for multi-GPU ZeRO-3 on Vertex)",
    )
    ap.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Override per-device micro-batch size",
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient_accumulation_steps",
    )
    ap.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Override logging_steps",
    )
    ap.add_argument(
        "--save-strategy",
        choices=["no", "steps", "epoch"],
        default=None,
        help="Override save_strategy",
    )
    ap.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Override save_total_limit",
    )
    packing_group = ap.add_mutually_exclusive_group()
    packing_group.add_argument(
        "--packing",
        dest="packing",
        action="store_true",
        help="Override SFT packing=True",
    )
    packing_group.add_argument(
        "--no-packing",
        dest="packing",
        action="store_false",
        help="Override SFT packing=False",
    )
    ap.set_defaults(packing=None)
    args = ap.parse_args()

    # Lazy import — the package's top-level avoids pulling torch on import.
    from nemotron_train import train
    from nemotron_train.recipe import BASE_MODEL

    extra_training_kwargs: dict = {}
    if args.deepspeed is not None:
        extra_training_kwargs["deepspeed"] = args.deepspeed
    if args.per_device_train_batch_size is not None:
        extra_training_kwargs["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.grad_accum is not None:
        extra_training_kwargs["gradient_accumulation_steps"] = args.grad_accum
    if args.logging_steps is not None:
        extra_training_kwargs["logging_steps"] = args.logging_steps
    if args.save_strategy is not None:
        extra_training_kwargs["save_strategy"] = args.save_strategy
    if args.save_total_limit is not None:
        extra_training_kwargs["save_total_limit"] = args.save_total_limit

    train(
        data_csv=args.data,
        output_dir=args.output,
        base_model=args.base or BASE_MODEL,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        sample_n=args.sample_n,
        max_length=args.max_length,
        packing=args.packing,
        extra_training_kwargs=extra_training_kwargs or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
