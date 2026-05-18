"""SFT training pipeline for a rank-32 LoRA on Nemotron-3-Nano-30B-A3B-BF16.

Implements the convergent public 0.85 recipe (see docs/notebook_learnings/04_*.md and
00_SYNTHESIS.md) plus `completion_only_loss=True` (the basic-SFT-group gap identified
in round 2).

Public entry points:
    LORA_CONFIG, TRAINING_DEFAULTS    # the recipe constants
    load_sft_dataset(csv_path, ...)   # CSV -> HF datasets.Dataset of {"text": ...}
    train(...)                         # main training function
"""

from .recipe import LORA_CONFIG, TRAINING_DEFAULTS
from .data import load_sft_dataset

__all__ = ["LORA_CONFIG", "TRAINING_DEFAULTS", "load_sft_dataset", "train"]


def train(*args, **kwargs):  # lazy import — pulls in torch/trl/peft
    from .trainer import train as _train

    return _train(*args, **kwargs)
