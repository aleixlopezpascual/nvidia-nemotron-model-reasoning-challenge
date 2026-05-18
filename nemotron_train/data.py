"""Load + format a CoT corpus for SFT against Nemotron-3-Nano-30B.

Default schema is the `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
CSV: `id, prompt, answer, generated_cot, label`. The loader is forgiving of the
optional `label` column.

Output: an HF `datasets.Dataset` with one `text` column ready for TRL's
`SFTTrainer` (since `dataset_text_field="text"` is the default).

The chat template is applied here once, not at training time — this guarantees
the training prompt matches the host's inference prompt byte-for-byte (modulo
the assistant turn we're training on). The user-message suffix is the verified
`BOXED_INSTRUCTION` from `nemotron_local_cv.host_metric` (with backticks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from nemotron_local_cv.host_metric import BOXED_INSTRUCTION


def _build_assistant_content(cot: str, answer: str) -> str:
    """Assistant turn in the Nemotron 'thinking' format.

    The tokenizer's chat template with `enable_thinking=True` wraps this content
    in the appropriate `<think>...</think>` scaffolding at inference time. We
    therefore put the CoT body (without explicit think tags) then the boxed
    final answer on its own line. If the upstream CoT already contains
    `\\boxed{...}` we strip it so the only boxed value is the verified one.
    """
    body = cot or ""
    # Strip any stray boxed-answer claims in the CoT body so the verified ground
    # truth is the only boxed value the model is trained to emit.
    import re

    body = re.sub(r"\\boxed\{[^}]*\}", "", body).strip()
    return f"{body}\n\\boxed{{{answer}}}"


def _format_row(tokenizer, prompt: str, answer: str, cot: str) -> str:
    """Render one row through the tokenizer's chat template."""
    messages = [
        {"role": "user", "content": prompt + BOXED_INSTRUCTION},
        {"role": "assistant", "content": _build_assistant_content(cot, answer)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )


def load_sft_dataset(
    csv_path: str | Path,
    tokenizer,
    *,
    drop_labels: Iterable[str] = ("UNKNOWN_NEEDS_RECLASSIFY",),
    max_chars: int | None = None,
    sample_n: int | None = None,
    seed: int = 42,
):
    """Load a CoT CSV into an HF Dataset of {"text": str} rows.

    Args:
        csv_path: path to the CoT CSV (kienngx schema).
        tokenizer: a HuggingFace tokenizer with `apply_chat_template`.
        drop_labels: any rows whose `label` matches one of these are dropped.
            Pass an empty tuple to keep everything.
        max_chars: optional pre-filter — drop rows whose `generated_cot` length
            (in characters) exceeds this. Useful as a coarse proxy for
            8192-token cap; let TRL packing/truncation handle the fine end.
        sample_n: if set, take a random N-row subsample (after filtering).
        seed: pandas sample seed.
    """
    df = pd.read_csv(csv_path)
    required = {"id", "prompt", "answer", "generated_cot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"corpus is missing required columns: {missing}")

    df["generated_cot"] = df["generated_cot"].fillna("").astype(str)
    df["answer"] = df["answer"].astype(str)
    df["prompt"] = df["prompt"].astype(str)

    if "label" in df.columns:
        for label in drop_labels:
            n_before = len(df)
            df = df[df["label"] != label]
            print(f"[data] dropped {n_before - len(df)} rows with label={label!r}")

    if max_chars is not None:
        n_before = len(df)
        df = df[df["generated_cot"].str.len() <= max_chars]
        print(f"[data] dropped {n_before - len(df)} rows with cot > {max_chars} chars")

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        print(f"[data] subsampled to {sample_n} rows (seed={seed})")

    print(f"[data] formatting {len(df)} rows through chat template...")
    texts = [
        _format_row(tokenizer, p, a, c)
        for p, a, c in zip(df["prompt"], df["answer"], df["generated_cot"])
    ]

    from datasets import Dataset

    return Dataset.from_dict({"text": texts})
