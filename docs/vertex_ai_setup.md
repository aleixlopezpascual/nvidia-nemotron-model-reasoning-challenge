# Vertex AI Setup Guide

End-to-end recipe for running the training + local-CV pipeline on a Vertex AI Workbench Instance. Assumes you already have a GCP project with Vertex AI enabled.

Why Vertex AI Workbench (over a one-shot Custom Training Job): we want **iterative** sessions — train, run CV, inspect predictions, tweak, retrain. Workbench gives us a persistent JupyterLab + terminal with attached GPU and a re-attachable boot disk.

---

## 1. Pick the GPU SKU

The 30B base in bf16 needs ~60 GB just for weights. Add LoRA grads + activations + gradient checkpointing: **80 GB VRAM minimum**. Don't try anything smaller.

| Vertex AI machine type | GPU | VRAM | Approx. on-demand $/hr (us-central1) | Fit |
|---|---|---|---|---|
| `a2-ultragpu-1g` | 1× A100 80 GB | 80 GB | ~$5.0/hr | ✅ fits |
| `a3-highgpu-1g` | 1× H100 80 GB | 80 GB | ~$11.0/hr | ✅ fits, ~2× faster training |
| `g2-standard-12` | 1× L4 | 24 GB | ~$0.7/hr | ❌ won't fit |
| `a3-megagpu-8g` | 8× H100 80 GB | 8× 80 GB | ~$88/hr | ✅ overkill for a 32-rank LoRA |

**Default recommendation: `a3-highgpu-1g` for production training runs, `a2-ultragpu-1g` for iteration / CV.** The H100 lift on a 30B base in eager attention is roughly 2× wall-clock; the cost ratio is also ~2×, so they're equivalent per-experiment but H100 lets you iterate faster.

Disk: ~300 GB SSD persistent disk is plenty (base model is ~60 GB, plus adapters/checkpoints/data).

Region: pick one near you with `a3-highgpu-1g` availability. `us-central1`, `us-east4`, `europe-west4` typically have it.

> **Cost note**: a full training run is ~1000 steps at ~30 min on H100 = ~$5–10 per run. Local CV is ~30–90 min = ~$5–10 per CV. Total budget for the 4-week plan: order of $200–500 if you're disciplined. Idle-shutdown the instance when not in use.

---

## 2. Create the Workbench Instance

UI path: **Vertex AI → Workbench → Instances → Create new**.

Or via gcloud (preferred for reproducibility):

```bash
PROJECT=<your-gcp-project>
REGION=us-central1
ZONE=us-central1-a       # must match the region; a3-highgpu-1g is in -a/-b/-c
INSTANCE=nemotron-dev

gcloud workbench instances create $INSTANCE \
  --project=$PROJECT \
  --location=$ZONE \
  --machine-type=a3-highgpu-1g \
  --accelerator-type=NVIDIA_H100_80GB \
  --accelerator-core-count=1 \
  --install-gpu-driver \
  --boot-disk-size=300 \
  --boot-disk-type=PD_SSD \
  --data-disk-size=300 \
  --data-disk-type=PD_SSD \
  --idle-shutdown \
  --idle-shutdown-timeout=1800     # 30 min — auto-stop when idle
```

Once it shows **RUNNING**, open the **JupyterLab** link from the Workbench console. From inside JupyterLab → **Terminal**.

---

## 3. First-time environment setup (inside the instance terminal)

```bash
# Pin a working dir on the persistent boot disk so nothing is lost on stop/start.
mkdir -p /home/jupyter/work && cd /home/jupyter/work

# Install uv (used to manage our project's deps).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Clone the repo (use HTTPS + gcloud auth or your GH PAT).
git clone https://github.com/<your-username>/nvidia-nemotron-model-reasoning-challenge.git
cd nvidia-nemotron-model-reasoning-challenge

# Recreate the project venv from uv.lock.
uv sync

# Install the heavy training / inference deps INTO the project venv.
# We keep these out of pyproject.toml so the laptop side doesn't try to install
# (impossible on Apple Silicon for vllm/torch+CUDA).
uv pip install \
    "torch==2.5.*" \
    "transformers>=4.45,<4.50" \
    "peft>=0.13,<0.16" \
    "trl>=0.12,<0.18" \
    "datasets>=3.0" \
    "accelerate>=1.0" \
    "vllm>=0.6.3" \
    "bitsandbytes>=0.44"

# Sanity: GPU visible, torch sees it
uv run python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

> **Version pins**: TRL's `SFTConfig.completion_only_loss` kwarg landed in 0.12. PEFT supports Nemotron via `trust_remote_code`. vLLM ≥ 0.6.3 has LoRA support for Nemotron MoE+Mamba.

---

## 4. Stage credentials (one time per instance)

### Kaggle (for fetching datasets/models)

From your laptop:

```bash
cat ~/.kaggle/kaggle.json   # copy the contents
```

On the instance:

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"<your-kaggle-username>","key":"<your-kaggle-key>"}
EOF
chmod 600 ~/.kaggle/kaggle.json
uv run kaggle competitions list 2>&1 | head -3   # sanity
```

### HuggingFace (for the base model)

```bash
uv pip install huggingface_hub
huggingface-cli login   # paste a read token; the base model is gated until you accept the license once
```

---

## 5. Pre-stage the heavy artifacts

These get fetched **once** to the data disk; subsequent runs reuse them.

```bash
cd /home/jupyter/work/nvidia-nemotron-model-reasoning-challenge

# 1) Base model (~60 GB). Pull into a fixed local path so vLLM/HF transformers
#    use a stable filesystem location.
mkdir -p /home/jupyter/models
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir /home/jupyter/models/Nemotron-3-Nano-30B-A3B-BF16

# 2) Competition data (train.csv, test.csv).
uv run kaggle competitions download nvidia-nemotron-model-reasoning-challenge
unzip -o nvidia-nemotron-model-reasoning-challenge.zip

# 3) Training CoT corpus (13 MB, kienngx Apache-2.0).
uv run python scripts/download_training_corpus.py

# 4) Sanity-build the 600-row stratified val split (already checked into git,
#    re-deriving here for verification).
uv run python scripts/build_val_split.py
```

Verify the layout:

```
/home/jupyter/models/Nemotron-3-Nano-30B-A3B-BF16/   # base model
/home/jupyter/work/nvidia-nemotron-model-reasoning-challenge/
    data/val_split_600.csv                            # CV val split
    data/final_Nemotron_training_data.csv             # 9500-row CoT corpus
    train.csv, test.csv                               # competition data
```

---

## 6. Run training

```bash
cd /home/jupyter/work/nvidia-nemotron-model-reasoning-challenge

# Quick smoke run: 50 steps on 200 rows. Should finish in <10 min on H100.
uv run python -m nemotron_train \
    --data data/final_Nemotron_training_data.csv \
    --output /home/jupyter/adapters/smoke \
    --base /home/jupyter/models/Nemotron-3-Nano-30B-A3B-BF16 \
    --sample-n 200 \
    --max-steps 50

# Full reproduction run: ~1 epoch on the full corpus, ~1000 steps.
uv run python -m nemotron_train \
    --data data/final_Nemotron_training_data.csv \
    --output /home/jupyter/adapters/repro_v1 \
    --base /home/jupyter/models/Nemotron-3-Nano-30B-A3B-BF16 \
    --max-steps 1000
```

The adapter lands at `/home/jupyter/adapters/repro_v1/final/`. Two files we care about: `adapter_config.json`, `adapter_model.safetensors`.

---

## 7. Run local CV

```bash
cd /home/jupyter/work/nvidia-nemotron-model-reasoning-challenge

uv run python -m nemotron_local_cv \
    --adapter /home/jupyter/adapters/repro_v1/final \
    --val data/val_split_600.csv \
    --model /home/jupyter/models/Nemotron-3-Nano-30B-A3B-BF16 \
    --debug
```

Output is `overall accuracy: 0.XXXX` plus a per-label breakdown. With the convergent 0.85 recipe + `completion_only_loss=True` we expect ~0.84-0.87 local accuracy (range covers train/val noise + the public-LB bonus from off-Kaggle compute).

---

## 8. Submit to Kaggle

```bash
cd /home/jupyter/adapters/repro_v1/final
zip -j /home/jupyter/submissions/repro_v1.zip adapter_config.json adapter_model.safetensors

uv run kaggle competitions submit \
    -c nvidia-nemotron-model-reasoning-challenge \
    -f /home/jupyter/submissions/repro_v1.zip \
    -m "repro of public 0.85 recipe + completion_only_loss=True, run repro_v1"
```

You can also do this from the laptop after copying the adapter back:

```bash
# laptop
gcloud workbench instances get-credentials nemotron-dev --location=us-central1-a   # if needed
gcloud compute scp --recurse \
    --tunnel-through-iap \
    --zone=us-central1-a \
    nemotron-dev:/home/jupyter/adapters/repro_v1/final ./adapters/repro_v1
```

---

## 9. Day-to-day: stop / start

When done for the session:

```bash
gcloud workbench instances stop nemotron-dev --location=us-central1-a
```

Idle-shutdown also fires after 30 min of no activity. The boot + data disks persist, so next time:

```bash
gcloud workbench instances start nemotron-dev --location=us-central1-a
```

…and pick up where you left off — the venv, base model, datasets, adapters, all still there.

---

## Troubleshooting

- **OOM during training**: lower `--per-device-train-batch-size` (already 1; you'd need to enable `--gradient-checkpointing` if disabled — it's on by default in our recipe). Last resort: bump up to `a3-megagpu-2g` (2× H100, then enable `tensor_parallel_size=2` for inference; SFT would need DDP).
- **vLLM "tensor_parallel_size > GPUs visible"**: check `nvidia-smi`; the harness uses `tensor_parallel_size=1` by default — single-GPU.
- **`completion_only_loss` raises unknown kwarg**: TRL version too old. Upgrade to `trl>=0.12`.
- **Chat template missing**: the Nemotron-3-Nano tokenizer ships one — make sure you `huggingface-cli login` before downloading and that `trust_remote_code=True` is set (the tokenizer is custom).
- **Submission scored very low (~0.5)**: something in the training prompt template doesn't match the host's eval prompt. Run a single-row diff against `nemotron_local_cv.host_metric.BOXED_INSTRUCTION` and `apply_chat_template(..., enable_thinking=True)`.

---

## Cost discipline checklist

- [ ] `--idle-shutdown-timeout=1800` set on the instance.
- [ ] Stop the instance manually when leaving for >30 min.
- [ ] Don't leave Jupyter cells running model loads while iterating on code.
- [ ] Reuse the data disk across sessions (don't re-download the 60 GB base every time).
- [ ] Smoke runs first (`--sample-n 200 --max-steps 50`) before full ~1000-step runs.
- [ ] Local CV before every submission; never burn a daily submission on an untested change.
