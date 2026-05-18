"""Verbatim mirror of the host's metric kernel (kaggle.com/code/metric/nvidia-nemotron-metric).

Pulled 2026-05-18 from the host kernel; the only deltas vs. the host source are:
  - `MODEL_PATH` / `DATA_PATH` are runtime-injected via `configure(...)` rather than
    pulled via kagglehub (we run off Kaggle).
  - The Triton/ptxas env tar+chmod commands at the top of the host kernel are
    Kaggle-specific and intentionally omitted here.

Everything load-bearing for correctness — the user-prompt suffix (with backticks),
`apply_chat_template` call (with `enable_thinking=True`), vLLM SamplingParams,
`extract_final_answer`, `verify` — is byte-identical to the host source.

DO NOT EDIT functions in this file casually. If the host metric is updated, re-pull
and reconcile. The pure functions (extract_final_answer, verify) are pinned to the
2026-05-18 host version.
"""

from __future__ import annotations

import glob
import math
import multiprocessing
import os
import re
import time
from pathlib import Path

import pandas as pd

# Set via `configure(model_path=..., data_path=...)` before calling generate_predictions/score.
MODEL_PATH: Path | None = None
DATA_PATH: Path | None = None


def configure(model_path: str | Path, data_path: str | Path) -> None:
    """Set the base-model path and the data-path (for test.csv lookups)."""
    global MODEL_PATH, DATA_PATH
    MODEL_PATH = Path(model_path)
    DATA_PATH = Path(data_path)


# --- Verbatim from host metric.py: classes / pure functions --------------------


class ParticipantVisibleError(Exception):
    pass


def cache_model(
    path: str | Path,
    exts: tuple[str, ...] = (".bin", ".pt", ".safetensors"),
    num_workers: int | None = None,
    chunk_mb: int = 256,
) -> int:
    """Pre-read model weight files into the OS page cache to speed up later loads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath: Path) -> tuple[Path, int]:
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        try:
            with open(fpath, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    total += len(data)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
        return fpath, total

    path = Path(path)
    files: list[Path] = []
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file() and str(p).endswith(exts)]
        files.sort()
    else:
        files = [path] if path.exists() else []

    if not files:
        print(f"No model files found to cache at: {path}")
        return 0

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {fpath.name}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f"[cache_model] total read ≈ {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")
    return total_bytes


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Updated 2026-05-18 host version: walks `\boxed{` starts and takes everything up to
    the last `}` before the next `\boxed{` (or end of text). Handles answers that contain
    literal `}` and nested LaTeX like `\boxed{\frac{1}{2}}` correctly.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative
    tolerance (1e-2); otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# --- Verbatim from host: the boxed-instruction suffix --------------------------

# Note the BACKTICKS around `\boxed{}` and `\boxed{your answer}` — these are
# part of the host's prompt and must be reproduced verbatim during training.
BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def build_user_content(prompt: str) -> str:
    """Append the host's boxed-instruction suffix to a raw puzzle prompt."""
    return prompt + BOXED_INSTRUCTION


def generate_predictions(
    test_df: pd.DataFrame,
    lora_path: str,
    row_id_col: str,
    max_lora_rank: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    debug: bool = False,
) -> pd.DataFrame:
    """Load the base model + LoRA, run generation, extract answers.

    Requires `configure(model_path=...)` to have been called.

    Args:
        debug: if True, also writes debug_predictions.csv with raw outputs.
    """
    if MODEL_PATH is None:
        raise RuntimeError("Call nemotron_local_cv.host_metric.configure(...) first.")

    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Triton/ptxas path only needed under Kaggle's runner; harmless if unset.
    os.environ.setdefault("TRITON_PTXAS_PATH", "/tmp/triton/backends/nvidia/bin/ptxas")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_df.itertuples(index=False):
        user_content = build_user_content(item.prompt)
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path),
    )

    predictions = []
    debug_records = []
    for item, output in zip(test_df.itertuples(index=False), outputs):
        raw_text = output.outputs[0].text
        extracted_answer = extract_final_answer(raw_text)
        row_id_val = getattr(item, row_id_col)
        predictions.append({row_id_col: row_id_val, "prediction": extracted_answer})
        if debug:
            debug_records.append(
                {
                    row_id_col: row_id_val,
                    "raw_output": raw_text,
                    "extracted_prediction": extracted_answer,
                }
            )

    if debug and debug_records:
        debug_df = pd.DataFrame(debug_records)
        debug_df.to_csv("debug_predictions.csv", index=False)
        print("Debug data saved to debug_predictions.csv")

    return pd.DataFrame(predictions)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    *,
    # Defaults below mirror the HOST EVAL CONFIG (NOT the host source-code defaults,
    # which are different and wrong for this comp — see 04_*.md in docs).
    max_lora_rank: int = 32,
    max_tokens: int = 7680,
    top_p: float = 1.0,
    temperature: float = 0.0,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
    debug: bool = False,
) -> float:
    """Run generation + verification. Returns aggregate accuracy.

    Note: the keyword-only defaults above match the *competition's runtime config*
    rather than the host source-code's misleading function-signature defaults.
    """
    if DATA_PATH is None:
        raise RuntimeError("Call nemotron_local_cv.host_metric.configure(...) first.")

    lora_path = submission["prediction"].iloc[0]

    test_df = pd.read_csv(DATA_PATH / "test.csv", index_col=None)
    row_id_col = str(test_df.columns.to_list()[0])
    test_df = test_df[test_df[row_id_col].isin(solution[row_id_column_name])]

    submission = generate_predictions(
        test_df=test_df,
        lora_path=lora_path,
        row_id_col=row_id_column_name,
        max_lora_rank=max_lora_rank,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        debug=debug,
    )

    dataset = solution.merge(submission, on=row_id_column_name)
    num_correct = 0
    for item in dataset.itertuples(index=False):
        ground_truth = item.answer
        extracted_answer = item.prediction
        if verify(str(ground_truth), str(extracted_answer)):
            num_correct += 1

    return float(num_correct / len(solution))
