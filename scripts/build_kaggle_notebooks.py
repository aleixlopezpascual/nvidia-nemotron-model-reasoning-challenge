"""Generate the Kaggle Notebook .ipynb files + kernel-metadata.json from Python source.

Hand-writing .ipynb JSON is painful and hard to diff. Each cell here is just a
Python string; the script emits notebooks ready for `kaggle kernels push`.

Run after editing this file:
    uv run python scripts/build_kaggle_notebooks.py

Then push (one-time create, subsequent runs use the same dir):
    uv run kaggle kernels push -p kaggle/train_kernel
    uv run kaggle kernels push -p kaggle/cv_kernel
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_USER = "aleixlopez"
CODE_DATASET = f"{KAGGLE_USER}/nemotron-code"

# Base model published by the host. Kagglehub path used by every public notebook.
BASE_MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"

# ----------------------------------------------------------------------------
# Reusable cell snippets
# ----------------------------------------------------------------------------

CELL_ENV_BEFORE_TORCH = '''\
# Set memory env BEFORE torch import (cannot be changed after).
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Untar our code bundle into /kaggle/working so the package layout is preserved.
import subprocess, sys
subprocess.run(
    ["tar", "xzf", f"/kaggle/input/nemotron-code/nemotron-code.tar.gz", "-C", "/kaggle/working"],
    check=True,
)
sys.path.insert(0, "/kaggle/working")

# Sanity: our packages import.
import nemotron_local_cv  # noqa: F401
import nemotron_train  # noqa: F401
print("nemotron_local_cv + nemotron_train imported.")
'''

CELL_BLACKWELL_PREMODEL = '''\
# Apply pre-model-load Blackwell env. (post-load patches happen after model load.)
from kaggle.blackwell_setup import set_cuda_alloc_env, set_blackwell_ptxas_env  # noqa: E402
set_cuda_alloc_env()
set_blackwell_ptxas_env()
print("CUDA alloc env + Blackwell ptxas env set.")
'''

CELL_RESOLVE_PATHS = '''\
import kagglehub
from pathlib import Path

BASE_MODEL_PATH = Path(kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"))
print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")

DATA_DIR = Path("/kaggle/working/data")
VAL_SPLIT = DATA_DIR / "val_split_600.csv"
assert VAL_SPLIT.exists(), VAL_SPLIT
'''

# ----------------------------------------------------------------------------
# Training kernel cells
# ----------------------------------------------------------------------------

TRAIN_CELL_DEPS = '''\
# Training-only deps. Kaggle's base image ships torch + transformers + datasets;
# we add the exact trl/peft/accelerate versions that match our recipe.
import subprocess, sys
deps = [
    "trl>=0.12,<0.18",
    "peft>=0.13,<0.16",
    "accelerate>=1.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *deps])
'''

TRAIN_CELL_CORPUS = '''\
# Locate the kienngx CoT corpus dataset. Attach it to this kernel before running:
#   Dataset slug: kienngx/nemotron-30b-competition-trainingdata-cot-labels
from pathlib import Path
import glob

CORPUS_CANDIDATES = glob.glob("/kaggle/input/**/final_Nemotron_training_data.csv", recursive=True)
assert CORPUS_CANDIDATES, "Attach the kienngx CoT corpus dataset before running."
CORPUS_PATH = Path(CORPUS_CANDIDATES[0])
print(f"CORPUS_PATH: {CORPUS_PATH}")

OUTPUT_DIR = Path("/kaggle/working/adapter")
OUTPUT_DIR.mkdir(exist_ok=True)
'''

TRAIN_CELL_RUN = '''\
# Smoke run first; flip --sample-n / --max-steps for the full run.
from nemotron_train import train

train(
    data_csv=str(CORPUS_PATH),
    output_dir=str(OUTPUT_DIR),
    base_model=str(BASE_MODEL_PATH),
    sample_n=200,        # bump to None for full corpus
    max_steps=50,        # bump to -1 for epochs-driven full run (~1000 steps)
)
'''

TRAIN_CELL_SUBMIT_ZIP = '''\
# Package the adapter as submission.zip in /kaggle/working (the Kaggle "Submit" button
# zips notebook outputs; this matches the host's expected zip layout).
import zipfile, os
from pathlib import Path

ADAPTER_DIR = OUTPUT_DIR  # already a Path
ZIP_PATH = Path("/kaggle/working/submission.zip")

REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
present = set(os.listdir(ADAPTER_DIR))
missing = [f for f in REQUIRED if f not in present]
assert not missing, f"missing required adapter files: {missing}"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in present:
        zf.write(ADAPTER_DIR / f, arcname=f)
print(f"wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
'''

# ----------------------------------------------------------------------------
# CV kernel cells
# ----------------------------------------------------------------------------

CV_CELL_DEPS = '''\
# vLLM + peft for inference. Kaggle's image may already have a compatible
# vllm — we pin the minimum that supports Nemotron MoE+Mamba LoRA loading.
import subprocess, sys
deps = [
    "vllm>=0.6.3",
    "peft>=0.13,<0.16",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *deps])
'''

CV_CELL_LOCATE_ADAPTER = '''\
# Locate the trained adapter. Two ways to attach it to this kernel:
#   (a) Add the training kernel as a kernel-source: adapter lives at
#       /kaggle/input/<train-kernel-slug>/adapter/
#   (b) Upload it as a private Dataset and attach: /kaggle/input/<dataset>/...
# Both code paths are covered by the glob below.
import glob

cands = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
assert cands, "No adapter_config.json found in /kaggle/input/**. Attach the trained-adapter source."
ADAPTER_DIR = str(Path(cands[0]).parent)
print(f"ADAPTER_DIR: {ADAPTER_DIR}")
'''

CV_CELL_RUN = '''\
# Run local CV via the host-metric mirror.
import pandas as pd
import tempfile
from pathlib import Path
from nemotron_local_cv import host_metric

val_df = pd.read_csv(VAL_SPLIT)
print(f"val rows: {len(val_df)}")
print(val_df["label"].value_counts().to_dict())

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    val_df.to_csv(tmp_path / "test.csv", index=False)
    host_metric.configure(model_path=str(BASE_MODEL_PATH), data_path=tmp_path)

    solution = val_df[["id", "answer"]]
    submission_stub = pd.DataFrame([{"prediction": ADAPTER_DIR}])

    accuracy = host_metric.score(
        solution=solution,
        submission=submission_stub,
        row_id_column_name="id",
        debug=True,
    )

print(f"\\n=== overall accuracy: {accuracy:.4f} ===")
'''

CV_CELL_PER_LABEL = '''\
# Per-label breakdown joined from the debug CSV produced above.
from pathlib import Path

debug = Path("/kaggle/working/debug_predictions.csv")
if debug.exists():
    pred = pd.read_csv(debug, dtype="str").fillna({"extracted_prediction": "NOT_FOUND"})
    merge = val_df.astype({"id": "str"}).merge(pred.astype({"id": "str"}), on="id", how="left")
    merge["is_correct"] = merge.apply(
        lambda r: host_metric.verify(str(r.answer), str(r.extracted_prediction)),
        axis=1,
    )
    print(merge.groupby("label").is_correct.agg(["mean", "count"]).to_string())
    merge.to_csv("/kaggle/working/debug_predictions_with_labels.csv", index=False)
else:
    print("debug_predictions.csv not found — re-run with debug=True")
'''

# ----------------------------------------------------------------------------
# Notebook builders
# ----------------------------------------------------------------------------


def _code(src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return {
        "cell_type": "code",
        "source": lines,
        "metadata": {"trusted": True},
        "execution_count": None,
        "outputs": [],
    }


def _markdown(text: str) -> dict:
    lines = [l + "\n" for l in text.splitlines()]
    return {"cell_type": "markdown", "source": lines, "metadata": {}}


def _notebook(cells: list[dict], internet: bool) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {
                "accelerator": "nvidiaRtxPro6000",
                "dataSources": [],  # set in kernel-metadata.json, not here
                "isInternetEnabled": internet,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def _kernel_metadata(slug: str, title: str, internet: bool, dataset_sources: list[str],
                     competition_sources: list[str] | None = None) -> dict:
    return {
        "id": slug,
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true" if internet else "false",
        "dataset_sources": dataset_sources,
        "competition_sources": competition_sources or [],
        "kernel_sources": [],
        "model_sources": [BASE_MODEL_HANDLE],
    }


def build_train_kernel(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        _markdown(
            "# Nemotron SFT Training (Kaggle / RTX PRO 6000)\n\n"
            "Runs the convergent public 0.85 SFT recipe + `completion_only_loss=True`. "
            "Defaults to a **smoke run** (200 rows, 50 steps) — bump `sample_n` to `None` and "
            "`max_steps` to `-1` for a full ~1-epoch run on the kienngx CoT corpus.\n\n"
            "Data sources required: this code dataset (`aleixlopez/nemotron-code`), the base "
            "model (`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`), and the "
            "kienngx CoT corpus (`kienngx/nemotron-30b-competition-trainingdata-cot-labels`)."
        ),
        _code(CELL_ENV_BEFORE_TORCH),
        _code(CELL_BLACKWELL_PREMODEL),
        _code(TRAIN_CELL_DEPS),
        _code(CELL_RESOLVE_PATHS),
        _code(TRAIN_CELL_CORPUS),
        _code(TRAIN_CELL_RUN),
        _code(TRAIN_CELL_SUBMIT_ZIP),
    ]
    (out_dir / "notebook.ipynb").write_text(json.dumps(_notebook(cells, internet=True), indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(
        _kernel_metadata(
            slug=f"{KAGGLE_USER}/nemotron-sft-training",
            title="Nemotron SFT Training",
            internet=True,
            dataset_sources=[CODE_DATASET, "kienngx/nemotron-30b-competition-trainingdata-cot-labels"],
            competition_sources=["nvidia-nemotron-model-reasoning-challenge"],
        ),
        indent=2,
    ))
    print(f"built {out_dir}/notebook.ipynb + kernel-metadata.json")


def build_cv_kernel(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        _markdown(
            "# Nemotron Local CV (Kaggle / RTX PRO 6000)\n\n"
            "Runs `nemotron_local_cv.host_metric` against the 600-row stratified val split. "
            "Mirrors the host scorer's prompt + extraction + verification byte-for-byte.\n\n"
            "Attach: this code dataset, the base model, and the trained adapter (either as "
            "another Dataset or as a kernel-source pointing at the training kernel's output)."
        ),
        _code(CELL_ENV_BEFORE_TORCH),
        _code(CELL_BLACKWELL_PREMODEL),
        _code(CV_CELL_DEPS),
        _code(CELL_RESOLVE_PATHS),
        _code(CV_CELL_LOCATE_ADAPTER),
        _code(CV_CELL_RUN),
        _code(CV_CELL_PER_LABEL),
    ]
    (out_dir / "notebook.ipynb").write_text(json.dumps(_notebook(cells, internet=True), indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(
        _kernel_metadata(
            slug=f"{KAGGLE_USER}/nemotron-local-cv",
            title="Nemotron Local CV",
            internet=True,
            dataset_sources=[CODE_DATASET],
            competition_sources=["nvidia-nemotron-model-reasoning-challenge"],
        ),
        indent=2,
    ))
    print(f"built {out_dir}/notebook.ipynb + kernel-metadata.json")


def main() -> int:
    build_train_kernel(REPO_ROOT / "kaggle" / "train_kernel")
    build_cv_kernel(REPO_ROOT / "kaggle" / "cv_kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
