"""Local CV harness mirroring the host metric for the NVIDIA Nemotron Reasoning Challenge.

Public entry points:
    classify_puzzle_type(prompt) -> str         # 6-way labeler
    build_stratified_val_split(...)             # one-shot val builder
    score_adapter(adapter_path, val_df, ...)    # run vLLM + scoring (needs GPU)
"""

from .labels import PUZZLE_TYPES, classify_puzzle_type
from .splits import build_stratified_val_split

__all__ = [
    "PUZZLE_TYPES",
    "classify_puzzle_type",
    "build_stratified_val_split",
]
