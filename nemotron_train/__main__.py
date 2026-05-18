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
    args = ap.parse_args()

    # Lazy import — the package's top-level avoids pulling torch on import.
    from nemotron_train import train
    from nemotron_train.recipe import BASE_MODEL

    train(
        data_csv=args.data,
        output_dir=args.output,
        base_model=args.base or BASE_MODEL,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        sample_n=args.sample_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
