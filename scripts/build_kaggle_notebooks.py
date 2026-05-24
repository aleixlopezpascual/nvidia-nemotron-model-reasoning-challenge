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

TRIAGE_MODEL_SOURCES = [
    "kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1",
    "huikang/nemotron-adapter/Transformers/default/20",
    "kienngx/nemotron-nano-30b-trained/triton/5e-5/1",
    "kienngx/nemotron-nano-30b-trained/triton/1e-5/1",
    "kienngx/nemotron-nano-30b-trained/triton/1e-5-batch-2-workers-8/1",
    "kienngx/nemotron-nano-30b-trained/triton/batch2-lr1e-workers8/1",
    "kienngx/nemotron-nano-30b-trained/triton/grad-accum1-workers8-maxlen-4096-lr1e-5/1",
]


def _model_handle_from_mount_path(path: str) -> str | None:
    """Convert '/kaggle/input/models/<handle>' mount path into Kaggle model handle."""
    prefix = "/kaggle/input/models/"
    if isinstance(path, str) and path.startswith(prefix):
        handle = path[len(prefix):].strip("/")
        return handle or None
    return None

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
    "trl", "peft", "accelerate", "bitsandbytes",
])
print("trl + peft + accelerate + bitsandbytes installed from offline packages.")
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
# {label} run.
from nemotron_train import train

train(
    data_csv=str(CORPUS_PATH),
    output_dir=str(OUTPUT_DIR),
    base_model=str(BASE_MODEL_PATH),
    sample_n={sample_n},        # rows to subsample from the corpus
    max_steps={max_steps},      # cap on optimizer steps (-1 = epochs-driven)
    max_length={max_length},    # sequence length cap
    packing={packing},          # packing=False (required: eager attn incompatible with packing)
    use_qlora={use_qlora},      # QLoRA: load base in NF4 4-bit to fit on 96 GB Blackwell
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

# Compatibility shim for newer protobuf runtimes where MessageFactory.GetPrototype
# was removed (some Kaggle/papermill-side deps still call it).
from google.protobuf import message_factory as _message_factory
if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    def _get_prototype(self, descriptor):
        return _message_factory.GetMessageClass(descriptor)
    _message_factory.MessageFactory.GetPrototype = _get_prototype
    print("Applied protobuf MessageFactory.GetPrototype compatibility shim.")
'''

CV_CELL_RUN_BATCH = '''\
# Run no-training adapter evaluation via the batch CLI.
import sys
from pathlib import Path
import pandas as pd
from nemotron_local_cv import __main__ as cv_cli

SCRIPTS_ROOT = Path(CODE_ROOT) / "scripts"
if SCRIPTS_ROOT.is_dir():
    sys.path.insert(0, str(SCRIPTS_ROOT))
from materialize_merge_experiments import materialize_merge_experiments

manifests = []
static_manifest = Path(CODE_ROOT) / "data" / "adapter_candidates.csv"
if static_manifest.exists():
    manifests.append(static_manifest)

merge_plan = Path(CODE_ROOT) / "data" / "merge_experiments.csv"
if merge_plan.exists():
    merged_root = Path("/kaggle/working/merged_adapters")
    merged_manifest = Path("/kaggle/working/merged_adapter_manifest.csv")
    print(f"merge plan: {merge_plan}")
    materialize_merge_experiments(
        experiments_csv=merge_plan,
        output_root=merged_root,
        manifest_out=merged_manifest,
    )
    manifests.append(merged_manifest)

assert manifests, "missing manifests: expected adapter_candidates.csv and/or merge_experiments.csv"
for m in manifests:
    assert m.exists(), f"missing manifest: {m}"

manifest = Path("/kaggle/working/adapter_manifest_combined.csv")
combined = pd.concat([pd.read_csv(m) for m in manifests], ignore_index=True)
combined = combined.drop_duplicates(subset=["name"], keep="first")
combined.to_csv(manifest, index=False)
print(f"combined {len(combined)} adapters from {len(manifests)} manifest(s)")
print(f"manifest: {manifest}")

summary_out = Path("/kaggle/working/adapter_eval_summary.csv")
debug_dir = Path("/kaggle/working/adapter_debug")
if debug_dir.exists():
    import shutil
    shutil.rmtree(debug_dir)

cmd = [
    "nemotron_local_cv",
    "--adapter-manifest",
    str(manifest),
    "--summary-out",
    str(summary_out),
    "--debug-dir",
    str(debug_dir),
    "--debug",
    "--val",
    str(VAL_SPLIT),
    "--model",
    str(BASE_MODEL_PATH),
]
print("$ " + " ".join(cmd))
sys.argv = cmd
rc = cv_cli.main()
assert rc == 0, f"nemotron_local_cv exited with {rc}"

print(f"wrote {summary_out}")
'''

CV_CELL_SHOW_RESULTS = '''\
import pandas as pd
from pathlib import Path

summary = Path("/kaggle/working/adapter_eval_summary.csv")
assert summary.exists(), "adapter_eval_summary.csv missing"
df = pd.read_csv(summary)
print(df.to_string(index=False))

per_label = Path("/kaggle/working/adapter_eval_summary_per_label.csv")
if per_label.exists():
    pl = pd.read_csv(per_label)
    top = pl.sort_values(["label", "accuracy"], ascending=[True, False]).groupby("label").head(1)
    print("\\nTop adapter per label:")
    print(top.to_string(index=False))
'''

# Merge submission kernel: evaluates all merges SEQUENTIALLY to avoid OOM
# NEW: Also materialize one merge at a time (not all 9 upfront)
MERGE_SUBMIT_CELL_RUN_EVAL = '''\
# Evaluate all merge experiments from merge_experiments.csv.
# ULTRA-EFFICIENT: Materialize and evaluate ONE merge at a time
# No batch operations, minimal disk usage, minimal RAM
import sys
from pathlib import Path
import pandas as pd
from nemotron_local_cv import __main__ as cv_cli
import gc
import shutil

SCRIPTS_ROOT = Path(CODE_ROOT) / "scripts"
if SCRIPTS_ROOT.is_dir():
    sys.path.insert(0, str(SCRIPTS_ROOT))
from materialize_merge_experiments import materialize_merge_experiments

merge_plan = Path(CODE_ROOT) / "data" / "merge_experiments.csv"
assert merge_plan.exists(), f"merge_experiments.csv not found at {merge_plan}"

# Read the merge plan
df_plan = pd.read_csv(merge_plan)
print(f"Merge plan loaded: {len(df_plan)} experiments")

summary_out = Path("/kaggle/working/merge_eval_summary.csv")
all_results = []

# Evaluate SEQUENTIALLY: materialize one at a time, then delete
for idx, (merge_name, base_adapter, donor_adapter, donor_weight, include_modules) in enumerate(
    zip(df_plan["name"], df_plan["base_adapter"], df_plan["donor_adapter"], 
        df_plan["donor_weight"], df_plan["include_modules"])
):
    print(f"\\n[{idx+1}/{len(df_plan)}] Processing {merge_name}...")
    
    # Create single-merge experiment CSV
    single_experiment = Path(f"/kaggle/working/single_experiment_{idx}.csv")
    pd.DataFrame({
        "name": [merge_name],
        "base_adapter": [base_adapter],
        "donor_adapter": [donor_adapter],
        "donor_weight": [donor_weight],
        "include_modules": [include_modules if pd.notna(include_modules) else ""],
    }).to_csv(single_experiment, index=False)
    
    # Materialize just this one merge
    merged_dir = Path(f"/kaggle/working/merged_{idx}")
    merged_manifest = Path(f"/kaggle/working/manifest_{idx}.csv")
    
    print(f"  Materializing {merge_name}...")
    materialize_merge_experiments(
        experiments_csv=single_experiment,
        output_root=merged_dir,
        manifest_out=merged_manifest,
    )
    
    assert merged_manifest.exists(), f"Manifest not created for {merge_name}"
    
    # Evaluate just this merge
    print(f"  Evaluating {merge_name}...")
    summary_single = Path(f"/kaggle/working/result_{idx}.csv")
    cmd = [
        "nemotron_local_cv",
        "--adapter-manifest",
        str(merged_manifest),
        "--summary-out",
        str(summary_single),
        "--val",
        str(VAL_SPLIT),
        "--model",
        str(BASE_MODEL_PATH),
    ]
    
    sys.argv = cmd
    rc = cv_cli.main()
    
    if rc == 0 and summary_single.exists():
        result = pd.read_csv(summary_single)
        all_results.append(result)
        acc = result["accuracy"].iloc[0]
        print(f"  ✓ {merge_name}: {acc:.4f}")
    else:
        print(f"  ⚠️  {merge_name} evaluation failed")
    
    # Delete materialized merge to free disk space
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        print(f"  Cleaned up {merged_dir}")
    
    # Clean up temp files
    for temp_file in [single_experiment, merged_manifest, summary_single]:
        if temp_file.exists():
            temp_file.unlink()
    
    # Free GPU memory
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  Memory freed")

# Combine all results
print(f"\\nCombining {len(all_results)} results...")
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(summary_out, index=False)
    print(f"✓ Evaluation complete: {summary_out}")
    print("\\nFinal Rankings:")
    print(combined.sort_values("accuracy", ascending=False).to_string(index=False))
else:
    print(f"❌ No evaluations completed!")
    raise RuntimeError("All merge evaluations failed")
'''

MERGE_SUBMIT_CELL_RANK_PICKS = '''\
import pandas as pd
from pathlib import Path

summary = Path("/kaggle/working/merge_eval_summary.csv")
assert summary.exists(), "merge_eval_summary.csv missing"
df = pd.read_csv(summary)
print("All merge evaluations:")
print(df.sort_values("accuracy", ascending=False).to_string(index=False))

best_row = df.loc[df["accuracy"].idxmax()]
best_name = best_row["name"]
best_score = best_row["accuracy"]
print(f"\\n✓ Best merge: {best_name} @ {best_score:.4f}")

# Save best name for packaging step
Path("/kaggle/working/best_merge_name.txt").write_text(best_name)
'''

MERGE_SUBMIT_CELL_PACKAGE = '''\
import zipfile
from pathlib import Path

best_name = Path("/kaggle/working/best_merge_name.txt").read_text().strip()
print(f"Packaging best merge: {best_name}")

merged_root = Path("/kaggle/working/merged_adapters_phase2")
adapter_dir = merged_root / best_name
assert adapter_dir.is_dir(), f"Best merge dir not found: {adapter_dir}"

REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
missing = [f for f in REQUIRED if not (adapter_dir / f).is_file()]
assert not missing, f"Missing required files in {adapter_dir}: {missing}"

out_zip = Path("/kaggle/working/submission.zip")
out_zip.parent.mkdir(parents=True, exist_ok=True)
if out_zip.exists():
    out_zip.unlink()

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in REQUIRED:
        zf.write(adapter_dir / fname, arcname=fname)

# Verify
with zipfile.ZipFile(out_zip) as zf:
    names = sorted(zf.namelist())
assert names == sorted(REQUIRED), f"Unexpected zip contents: {names}"

size_gb = out_zip.stat().st_size / 1024**3
print(f"wrote {out_zip}  ({size_gb:.2f} GB)")
print(f"contents: {names}")
print("✓ Ready → click Output > submission.zip > Submit to Competition")
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


def _kernel_metadata(
    slug: str,
    title: str,
    internet: bool,
    dataset_sources: list[str],
    competition_sources: list[str] | None = None,
    kernel_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
) -> dict:
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
        "model_sources": model_sources or [BASE_MODEL_HANDLE],
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
    use_qlora: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cell_run = TRAIN_CELL_RUN_TEMPLATE.format(
        label=label,
        sample_n="None" if sample_n is None else sample_n,
        max_steps=max_steps,
        max_length=max_length,
        packing=packing,
        use_qlora=use_qlora,
    )
    cells = [
        _markdown(
            f"# Nemotron SFT Training (Kaggle / RTX PRO 6000) — {label}\n\n"
            "Runs the convergent public 0.85 SFT recipe + `completion_only_loss=True`. "
            f"Current configuration: `sample_n={sample_n}`, `max_steps={max_steps}`, "
            f"`max_length={max_length}`, `packing={packing}`, `use_qlora={use_qlora}`. "
            f"Edit `scripts/build_kaggle_notebooks.py` and re-run with different flags to change.\n\n"
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
        _code(CV_CELL_RUN_BATCH),
        _code(CV_CELL_SHOW_RESULTS),
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
            model_sources=[BASE_MODEL_HANDLE, *TRIAGE_MODEL_SOURCES],
        ),
        indent=2,
    ))
    print(f"built {out_dir}/notebook.ipynb + kernel-metadata.json")


# ----------------------------------------------------------------------------
# Submit kernel cells  — no GPU needed, just zips a chosen adapter
# ----------------------------------------------------------------------------

SUBMIT_CELL = '''\
# Package a chosen merged adapter from the CV kernel output into submission.zip.
# No GPU required — this notebook runs in seconds.
import os, zipfile
from pathlib import Path

# ── choose which adapter to package ──────────────────────────────────────────
# Must match a name in the CV kernel's merged_adapter_manifest.csv.
ADAPTER_NAME = "tinker_plus_5e5_full_010"
# ─────────────────────────────────────────────────────────────────────────────

# Locate the CV kernel output.  It's mounted as a kernel-source input so it
# lands under /kaggle/input/<kernel-owner>/<kernel-slug>/.
import glob
cv_candidates = glob.glob(
    f"/kaggle/input/**/merged_adapters/{ADAPTER_NAME}/adapter_config.json",
    recursive=True,
)
assert cv_candidates, (
    f"Could not find adapter_config.json for {ADAPTER_NAME} under /kaggle/input. "
    "Make sure the CV kernel is attached as a kernel source."
)
adapter_dir = Path(cv_candidates[0]).parent
print(f"adapter_dir: {adapter_dir}")

REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
missing = [f for f in REQUIRED if not (adapter_dir / f).is_file()]
assert not missing, f"Missing required files: {missing}"

out_zip = Path("/kaggle/working/submission.zip")
out_zip.parent.mkdir(parents=True, exist_ok=True)
if out_zip.exists():
    out_zip.unlink()

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in REQUIRED:
        zf.write(adapter_dir / fname, arcname=fname)

# Verify
with zipfile.ZipFile(out_zip) as zf:
    names = sorted(zf.namelist())
assert names == sorted(REQUIRED), f"Unexpected zip contents: {names}"

size_gb = out_zip.stat().st_size / 1024**3
print(f"wrote {out_zip}  ({size_gb:.2f} GB)")
print(f"contents: {names}")
print("Ready → click Output > submission.zip > Submit to Competition")
'''


def build_submit_kernel(out_dir: Path, adapter_name: str = "tinker_plus_5e5_full_010") -> None:
    """Build a lightweight submission-packager kernel (no GPU, no inference)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Inject chosen adapter name
    cell_src = SUBMIT_CELL.replace(
        '"tinker_plus_5e5_full_010"', f'"{adapter_name}"', 1
    )
    cells = [
        _markdown(
            f"# Nemotron Submit Packager — {adapter_name}\n\n"
            "Lightweight notebook that picks the best merged adapter from the CV kernel output "
            "and packages it as `submission.zip` ready for **Submit to Competition**.\n\n"
            "**No GPU required.** Attach `aleixlopez/nemotron-local-cv` as a kernel source."
        ),
        _code(cell_src),
    ]
    (out_dir / "notebook.ipynb").write_text(
        json.dumps(_notebook(cells, internet=False), indent=1)
    )
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(
            _kernel_metadata(
                slug=f"{KAGGLE_USER}/nemotron-submit",
                title="Nemotron Submit Packager",
                internet=False,
                dataset_sources=[],
                competition_sources=["nvidia-nemotron-model-reasoning-challenge"],
                kernel_sources=[f"{KAGGLE_USER}/nemotron-local-cv"],
            ),
            indent=2,
        )
    )
    print(f"built {out_dir}/notebook.ipynb + kernel-metadata.json")


def build_single_merge_kernel(out_dir: Path, merge_name: str, base_adapter: str, 
                             donor_adapter: str, donor_weight: float, 
                             include_modules: str = "") -> None:
    """Build a single-merge evaluation + submission kernel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Escape module list for Python string
    modules_str = include_modules if include_modules else ""
    
    merge_eval_cell = f'''\
# Single merge: {merge_name}
import sys
from pathlib import Path
import csv
import pandas as pd
from nemotron_local_cv import __main__ as cv_cli
import zipfile

SCRIPTS_ROOT = Path(CODE_ROOT) / "scripts"
if SCRIPTS_ROOT.is_dir():
    sys.path.insert(0, str(SCRIPTS_ROOT))
from merge_adapters import merge_adapters

# Create single-merge experiment
merge_name = "{merge_name}"
base_adapter = r"{base_adapter}"
donor_adapter = r"{donor_adapter}"
donor_weight = {donor_weight}
include_modules = r"{modules_str}"

print(f"base_adapter exists={{Path(base_adapter).is_dir()}}: {{base_adapter}}")
print(f"donor_adapter exists={{Path(donor_adapter).is_dir()}}: {{donor_adapter}}")
print(f"include_modules={{include_modules or '(all)'}}")

merged_root = Path("/kaggle/working/merged_adapter")
merged_root.mkdir(parents=True, exist_ok=True)
adapter_dir = merged_root / merge_name

# Materialize directly so any failure is visible in the notebook traceback.
print(f"Materializing {{merge_name}}...")
try:
    adapter_dir = merge_adapters(
        base_dir=base_adapter,
        donor_dir=donor_adapter,
        output_dir=adapter_dir,
        donor_weight=donor_weight,
        include_modules=tuple(m.strip() for m in include_modules.split(",") if m.strip()),
    )
except ValueError as exc:
    if "no overlapping tensors matched include_modules=" not in str(exc):
        raise
    print(f"WARNING: {{exc}}")
    print("Falling back to full merge with include_modules=()")
    adapter_dir = merge_adapters(
        base_dir=base_adapter,
        donor_dir=donor_adapter,
        output_dir=adapter_dir,
        donor_weight=donor_weight,
        include_modules=(),
    )

merged_manifest = Path("/kaggle/working/manifest.csv")
with merged_manifest.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["name", "adapter"])
    writer.writeheader()
    writer.writerow({{"name": merge_name, "adapter": str(adapter_dir)}})

# Evaluate
print(f"Evaluating {{merge_name}}...")
summary_out = Path("/kaggle/working/eval_summary.csv")
cmd = [
    "nemotron_local_cv",
    "--adapter-manifest",
    str(merged_manifest),
    "--summary-out",
    str(summary_out),
    "--val",
    str(VAL_SPLIT),
    "--model",
    str(BASE_MODEL_PATH),
]
sys.argv = cmd
rc = cv_cli.main()
if rc != 0:
    raise RuntimeError(f"Evaluation failed with code {{rc}}")

# Read result
df = pd.read_csv(summary_out)
if "accuracy" in df.columns:
    accuracy = df["accuracy"].iloc[0]
elif "overall_accuracy" in df.columns:
    accuracy = df["overall_accuracy"].iloc[0]
else:
    raise KeyError(f"expected accuracy column in {{list(df.columns)}}")
print(f"✓ {{merge_name}}: {{accuracy:.4f}}")

# Package merge as submission.zip
REQUIRED = ["adapter_config.json", "adapter_model.safetensors"]
missing = [f for f in REQUIRED if not (adapter_dir / f).is_file()]
if missing:
    raise RuntimeError(f"Missing files: {{missing}}")

submission_zip = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in REQUIRED:
        zf.write(adapter_dir / fname, arcname=fname)

size_mb = submission_zip.stat().st_size / 1024 / 1024
print(f"✓ Packaged {{submission_zip}} ({{size_mb:.2f}} MB)")
print(f"✓ Ready for submission!")
'''
    
    cells = [
        _markdown(
            f"# Nemotron Merge Evaluation: {merge_name}\n\n"
            f"Single-merge evaluation kernel.\n\n"
            f"**Merge:** {merge_name}\n"
            f"**Base:** Tinker\n"
            f"**Donor:** {donor_adapter.split('/')[-2]}\n"
            f"**Weight:** {donor_weight}\n"
            f"**Modules:** {include_modules if include_modules else '(all)'}"
        ),
        _code(CELL_ENV_BEFORE_TORCH),
        _code(CELL_BLACKWELL_PREMODEL),
        _code(CV_CELL_DEPS),
        _code(CELL_RESOLVE_PATHS),
        _code(merge_eval_cell),
    ]
    
    (out_dir / "notebook.ipynb").write_text(json.dumps(_notebook(cells, internet=False), indent=1))
    
    # Ensure all model mounts referenced in notebook code are attached.
    model_srcs = [BASE_MODEL_HANDLE]
    for adapter_path in (base_adapter, donor_adapter):
        handle = _model_handle_from_mount_path(adapter_path)
        if handle:
            model_srcs.append(handle)
    model_srcs = sorted(set(model_srcs))
    
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(
            _kernel_metadata(
                slug=f"{KAGGLE_USER}/nemotron-merge-{merge_name.lower()}",
                title=f"Nemotron Merge: {merge_name}",
                internet=False,
                dataset_sources=[CODE_DATASET, OFFLINE_PACKAGES_DATASET],
                competition_sources=["nvidia-nemotron-model-reasoning-challenge"],
                model_sources=model_srcs,
            ),
            indent=2,
        )
    )
    print(f"  ✓ {out_dir.name}")



SMOKE_PROFILES = {
    "tiny":  {"sample_n": 20,   "max_steps": 5,    "label": "tiny smoke",          "max_length": 2048, "packing": False, "use_qlora": True},
    "smoke": {"sample_n": 200,  "max_steps": 50,   "label": "smoke",               "max_length": 2048, "packing": False, "use_qlora": True},
    "full":  {"sample_n": None, "max_steps": 1000, "label": "qlora-8192 baseline", "max_length": 8192, "packing": False, "use_qlora": True},
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=list(SMOKE_PROFILES),
        default="smoke",
        help="Training run size: tiny (20/5), smoke (200/50), full (QLoRA 8192: full data, 1000 steps, max_length 8192, QLoRA 4-bit base)",
    )
    args = ap.parse_args()
    p = SMOKE_PROFILES[args.profile]
    print(
        f"profile: {args.profile} → sample_n={p['sample_n']}, max_steps={p['max_steps']}, "
        f"max_length={p['max_length']}, packing={p['packing']}"
    )
    build_train_kernel(REPO_ROOT / "kaggle" / "train_kernel", **p)
    build_cv_kernel(REPO_ROOT / "kaggle" / "cv_kernel")
    build_submit_kernel(
        REPO_ROOT / "kaggle" / "submit_kernel",
        adapter_name="tinker_plus_5e5_full_005",
    )
    
    # Phase 2: Build individual merge kernels (no batch)
    print("\nBuilding 9 individual merge kernels...")
    merge_experiments_csv = REPO_ROOT / "data" / "merge_experiments.csv"
    df_merges = __import__("pandas").read_csv(merge_experiments_csv)
    
    for idx, row in df_merges.iterrows():
        merge_name = row["name"]
        base_adapter = row["base_adapter"]
        donor_adapter = row["donor_adapter"]
        donor_weight = row["donor_weight"]
        include_modules = row["include_modules"] if "include_modules" in row and __import__("pandas").notna(row["include_modules"]) else ""
        
        kernel_dir = REPO_ROOT / "kaggle" / f"merge_{merge_name}"
        build_single_merge_kernel(kernel_dir, merge_name, base_adapter, donor_adapter, donor_weight, include_modules)
    
    print("\n✓ All 9 merge kernels built!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
