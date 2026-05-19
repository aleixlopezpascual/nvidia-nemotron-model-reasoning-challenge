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

# Base model published by the host. Pinned to version 1 (the only one as of 2026-05-18).
# Kagglehub uses the un-versioned slug below; kernel-metadata.json needs the versioned one.
BASE_MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
BASE_MODEL_KAGGLEHUB = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"

# ----------------------------------------------------------------------------
# Reusable cell snippets
# ----------------------------------------------------------------------------

CELL_ENV_BEFORE_TORCH = '''\
# Set memory env BEFORE torch import (cannot be changed after).
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Glob-discover instead of hardcoding — Kaggle's path conventions vary
# (`/kaggle/input/datasets/<owner>/<slug>/` for our newer dataset,
#  `/kaggle/input/<slug>/` for older public ones).
import glob, sys
matches = glob.glob("/kaggle/input/**/blackwell_setup.py", recursive=True)
assert matches, "code dataset not attached (no blackwell_setup.py found under /kaggle/input)"
CODE_ROOT = os.path.dirname(matches[0])
print(f"CODE_ROOT = {CODE_ROOT}")
sys.path.insert(0, CODE_ROOT)

import nemotron_local_cv  # noqa: F401
import nemotron_train  # noqa: F401
print("nemotron_local_cv + nemotron_train imported.")
'''

CELL_BLACKWELL_PREMODEL = '''\
# Pre-model-load Blackwell setup. The mamba_ssm stub MUST be installed before
# AutoModelForCausalLM.from_pretrained loads modeling_nemotron_h.py.
from blackwell_setup import (  # noqa: E402
    set_cuda_alloc_env,
    set_blackwell_ptxas_env,
    install_mamba_ssm_stub,
)
set_cuda_alloc_env()
set_blackwell_ptxas_env()
install_mamba_ssm_stub()
print("CUDA alloc env + Blackwell ptxas env + mamba_ssm stub ready.")
'''

CELL_RESOLVE_PATHS = '''\
from pathlib import Path

# Use the kernel-metadata.json-attached mount directly (faster than kagglehub.download).
BASE_MODEL_PATH = Path("/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
assert BASE_MODEL_PATH.is_dir(), f"base model not mounted at {BASE_MODEL_PATH}"
print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print("  files:", sorted(p.name for p in BASE_MODEL_PATH.iterdir())[:6], "...")

# val_split_600.csv ships inside the code dataset (see scripts/sync_code_to_kaggle.py).
VAL_SPLIT = Path(CODE_ROOT) / "data" / "val_split_600.csv"
assert VAL_SPLIT.exists(), VAL_SPLIT
'''

# ----------------------------------------------------------------------------
# Training kernel cells
# ----------------------------------------------------------------------------

TRAIN_CELL_DEPS = '''\
# Training-only deps installed from the public offline-packages dataset
# (`dennisfong/nvidia-nemotron-offline-packages`). Kaggle's base image already
# ships torch + transformers + datasets; we add trl/peft/accelerate for SFT.
# Internet must be OFF for competition compliance, so `--no-index --find-links`
# is mandatory.

import glob, os, subprocess, sys

# Walk /kaggle/input one level deep to surface what's actually attached.
print("== /kaggle/input layout ==")
for d in sorted(os.listdir("/kaggle/input")):
    sub = f"/kaggle/input/{d}"
    print(f"  {d}/")
    try:
        for f in sorted(os.listdir(sub))[:6]:
            print(f"    {f}/" if os.path.isdir(f"{sub}/{f}") else f"    {f}")
    except Exception as e:
        print(f"    <err: {e}>")

# Discover offline-packages dir by searching for a known-present wheel.
matches = glob.glob("/kaggle/input/**/offline_packages", recursive=True)
if not matches:
    # Some datasets mount the wheels at the dataset root, not a subdir.
    wheels = glob.glob("/kaggle/input/**/trl-*.whl", recursive=True)
    assert wheels, "no offline_packages dir AND no trl-*.whl wheel found"
    OFFLINE_DIR = os.path.dirname(wheels[0])
else:
    OFFLINE_DIR = matches[0]
print(f"\\nOFFLINE_DIR = {OFFLINE_DIR}")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-cache-dir",
    "--no-index", "--find-links", OFFLINE_DIR,
    "trl", "peft", "accelerate",
])
print("trl + peft + accelerate installed from offline packages.")
'''

TRAIN_CELL_CORPUS = '''\
# Locate the kienngx CoT corpus dataset. Attach it to this kernel before running:
#   Dataset slug: kienngx/nemotron-30b-competition-trainingdata-cot-labels
from pathlib import Path
import glob
import shutil

CORPUS_CANDIDATES = glob.glob("/kaggle/input/**/final_Nemotron_training_data.csv", recursive=True)
assert CORPUS_CANDIDATES, "Attach the kienngx CoT corpus dataset before running."
CORPUS_PATH = Path(CORPUS_CANDIDATES[0])
print(f"CORPUS_PATH: {CORPUS_PATH}")

OUTPUT_DIR = Path("/kaggle/working/adapter")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

# Free stale run artifacts to avoid papermill "no space left on device".
for stale in ("/kaggle/working/submission.zip", "/kaggle/working/debug_predictions.csv"):
    p = Path(stale)
    if p.exists():
        p.unlink()

total, used, free = shutil.disk_usage("/kaggle/working")
print(f"/kaggle/working free: {free / 1024**3:.2f} GB")
'''

TRAIN_CELL_RUN_TEMPLATE = '''\
# {label} run. Flip sample_n=None + max_steps=-1 for full corpus / ~1 epoch.
from nemotron_train import train

train(
    data_csv=str(CORPUS_PATH),
    output_dir=str(OUTPUT_DIR),
    base_model=str(BASE_MODEL_PATH),
    sample_n={sample_n},        # rows to subsample from the corpus
    max_steps={max_steps},      # cap on optimizer steps (-1 = epochs-driven)
    max_length={max_length},    # sequence length cap; lower first for OOM safety
    packing={packing},          # packing=False lowers average live tokens/step
    extra_training_kwargs={{"save_strategy": "no"}},  # avoid checkpoint bloat on Kaggle disk
)
'''

TRAIN_CELL_SUBMIT_ZIP = '''\
# Package the adapter as submission.zip in /kaggle/working (the Kaggle "Submit" button
# zips notebook outputs; this matches the host's expected zip layout).
import zipfile, os
from pathlib import Path

ADAPTER_DIR = OUTPUT_DIR  # trainer may save here or under OUTPUT_DIR/final
ZIP_PATH = Path("/kaggle/working/submission.zip")

REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
present = set(os.listdir(ADAPTER_DIR)) if ADAPTER_DIR.exists() else set()
missing = [f for f in REQUIRED if f not in present]
if missing and (ADAPTER_DIR / "final").is_dir():
    ADAPTER_DIR = ADAPTER_DIR / "final"
    present = set(os.listdir(ADAPTER_DIR))
    missing = [f for f in REQUIRED if f not in present]
assert not missing, f"missing required adapter files in {ADAPTER_DIR}: {missing}"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in present:
        zf.write(ADAPTER_DIR / f, arcname=f)
print(f"wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
'''

# ----------------------------------------------------------------------------
# CV kernel cells
# ----------------------------------------------------------------------------

CV_CELL_DEPS = '''\
# peft + vllm for adapter loading/inference. Some Kaggle images include vllm,
# others do not; install from offline packages to keep this kernel deterministic.
# Internet must be OFF for competition compliance, so `--no-index --find-links`
# is mandatory.
import glob, subprocess, sys
matches = glob.glob("/kaggle/input/**/offline_packages", recursive=True)
assert matches, "offline_packages dir not found under /kaggle/input"
OFFLINE_DIR = matches[0]
print(f"OFFLINE_DIR = {OFFLINE_DIR}")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--no-cache-dir",
    "--no-index", "--find-links", OFFLINE_DIR,
    "peft", "vllm",
])
print("peft + vllm installed from offline packages.")
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
                     competition_sources: list[str] | None = None,
                     kernel_sources: list[str] | None = None) -> dict:
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
        "kernel_sources": kernel_sources or [],
        "model_sources": [BASE_MODEL_HANDLE],
        "machine_shape": "NvidiaRtxPro6000",
    }


OFFLINE_PACKAGES_DATASET = "dennisfong/nvidia-nemotron-offline-packages"


def build_train_kernel(
    out_dir: Path,
    sample_n: int | None,
    max_steps: int,
    label: str,
    max_length: int,
    packing: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cell_run = TRAIN_CELL_RUN_TEMPLATE.format(
        label=label,
        sample_n="None" if sample_n is None else sample_n,
        max_steps=max_steps,
        max_length=max_length,
        packing=packing,
    )
    cells = [
        _markdown(
            f"# Nemotron SFT Training (Kaggle / RTX PRO 6000) — {label}\n\n"
            "Runs the convergent public 0.85 SFT recipe + `completion_only_loss=True`. "
            f"Current configuration: `sample_n={sample_n}`, `max_steps={max_steps}`, "
            f"`max_length={max_length}`, `packing={packing}`. "
            "Edit `scripts/build_kaggle_notebooks.py` and re-run with different flags to change.\n\n"
            "Data sources required: this code dataset (`aleixlopez/nemotron-code`), the base "
            "model (`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`), and the "
            "kienngx CoT corpus (`kienngx/nemotron-30b-competition-trainingdata-cot-labels`)."
        ),
        _code(CELL_ENV_BEFORE_TORCH),
        _code(CELL_BLACKWELL_PREMODEL),
        _code(TRAIN_CELL_DEPS),
        _code(CELL_RESOLVE_PATHS),
        _code(TRAIN_CELL_CORPUS),
        _code(train_cell_run),
        _code(TRAIN_CELL_SUBMIT_ZIP),
    ]
    (out_dir / "notebook.ipynb").write_text(json.dumps(_notebook(cells, internet=False), indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(
        _kernel_metadata(
            slug=f"{KAGGLE_USER}/nemotron-sft-training",
            title="Nemotron SFT Training",
            internet=False,
            dataset_sources=[
                CODE_DATASET,
                "kienngx/nemotron-30b-competition-trainingdata-cot-labels",
                OFFLINE_PACKAGES_DATASET,
            ],
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
    (out_dir / "notebook.ipynb").write_text(json.dumps(_notebook(cells, internet=False), indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(
        _kernel_metadata(
            slug=f"{KAGGLE_USER}/nemotron-local-cv",
            title="Nemotron Local CV",
            internet=False,
            dataset_sources=[CODE_DATASET, OFFLINE_PACKAGES_DATASET],
            competition_sources=["nvidia-nemotron-model-reasoning-challenge"],
            kernel_sources=[f"{KAGGLE_USER}/nemotron-sft-training"],
        ),
        indent=2,
    ))
    print(f"built {out_dir}/notebook.ipynb + kernel-metadata.json")


SMOKE_PROFILES = {
    "tiny":  {"sample_n": 20,   "max_steps": 5,    "label": "tiny smoke", "max_length": 2048, "packing": False},
    "smoke": {"sample_n": 200,  "max_steps": 50,   "label": "smoke", "max_length": 2048, "packing": False},
    "full":  {"sample_n": None, "max_steps": -1,   "label": "full", "max_length": 2048, "packing": False},
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(SMOKE_PROFILES), default="smoke",
                    help="Training run size: tiny (20 rows / 5 steps), smoke (200/50), full (epochs-driven)")
    args = ap.parse_args()
    p = SMOKE_PROFILES[args.profile]
    print(
        f"profile: {args.profile} → sample_n={p['sample_n']}, max_steps={p['max_steps']}, "
        f"max_length={p['max_length']}, packing={p['packing']}"
    )
    build_train_kernel(REPO_ROOT / "kaggle" / "train_kernel", **p)
    build_cv_kernel(REPO_ROOT / "kaggle" / "cv_kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
