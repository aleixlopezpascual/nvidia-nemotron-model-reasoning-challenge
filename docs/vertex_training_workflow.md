# Vertex-First Training Workflow (A100 40GB x up to 8)

This is the training path for maximum-score runs when Kaggle-only training is too memory-constrained.

Use Vertex for training, then use Kaggle kernels for CV parity and submission packaging.

---

## 1) Environment setup on Vertex VM

```bash
cd /workspace
git clone https://github.com/aleixlopezpascual/nvidia-nemotron-model-reasoning-challenge.git
cd nvidia-nemotron-model-reasoning-challenge

uv sync
uv pip install "torch==2.5.*" "transformers>=4.45,<4.50" "peft>=0.13,<0.16" \
  "trl>=0.12,<0.18" "datasets>=3.0" "accelerate>=1.0" "deepspeed>=0.15"
```

Prepare data and model paths:

```bash
# Competition + corpus
uv run kaggle competitions download nvidia-nemotron-model-reasoning-challenge
unzip -o nvidia-nemotron-model-reasoning-challenge.zip
uv run python scripts/download_training_corpus.py

# Base model local path (after HF login + license accept)
# Example:
# /workspace/models/Nemotron-3-Nano-30B-A3B-BF16
```

---

## 2) Baseline reproduction run (distributed, ZeRO-3)

Run with 8 GPUs (A100 40GB each):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m nemotron_train \
  --data data/final_Nemotron_training_data.csv \
  --output /workspace/adapters/baseline_repro \
  --base /workspace/models/Nemotron-3-Nano-30B-A3B-BF16 \
  --max-steps 1000 \
  --max-length 8192 \
  --packing \
  --deepspeed configs/deepspeed_zero3_lora.json \
  --per-device-train-batch-size 1 \
  --grad-accum 4 \
  --save-strategy epoch \
  --save-total-limit 2
```

Notes:
- `configs/deepspeed_zero3_lora.json` is included in this repo.
- Effective batch here is `8 * 1 * 4 = 32`, matching the current recipe target.
- If memory is unstable, reduce `--max-length` first (e.g., 6144 or 4096), then re-scale.

---

## 3) Targeted continuation run (equation-focused)

After building a focused dataset CSV:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m nemotron_train \
  --data data/equation_focused_training_data.csv \
  --output /workspace/adapters/equation_focus_v1 \
  --base /workspace/models/Nemotron-3-Nano-30B-A3B-BF16 \
  --max-steps 300 \
  --max-length 8192 \
  --packing \
  --deepspeed configs/deepspeed_zero3_lora.json \
  --per-device-train-batch-size 1 \
  --grad-accum 4 \
  --save-strategy steps \
  --save-total-limit 3
```

---

## 4) Optional Stage 2: GRPO (after SFT only)

Use this only after you have a strong SFT checkpoint. Treat GRPO as a small polish stage, not the primary training stage.

Why this ordering:
- SFT stabilizes output behavior/format and preserves the current baseline.
- GRPO is better for relative quality refinement once behavior is already stable.
- In this competition, sparse reward (final answer correctness) makes GRPO noisier than SFT.

Compute guidance:
- On smaller GPUs (e.g., L4), run GRPO only as a short experiment with strict rollout limits.
- Prefer targeted continuation SFT first for the weak category (`equation_transformation`).
- Promote GRPO candidates only if they beat the SFT-only checkpoint on overall CV and weak-label accuracy.

---

## 5) Move adapters into Kaggle CV + submission flow

1. Package the candidate adapter locally if needed:
```bash
uv run python scripts/package_submission.py --adapter-dir /workspace/adapters/<run>/final --output submission.zip
```

2. Prefer Kaggle-side parity scoring and submission packaging:
- CV kernel: `aleixlopez/nemotron-local-cv`
- submit packager: `aleixlopez/nemotron-submit-packager`

3. Promote only candidates that improve both:
- overall local CV score
- `equation_transformation` accuracy

---

## 6) Fast troubleshooting

- OOM: reduce `--max-length` (first), then lower `--grad-accum`.
- Divergence/regression: reduce `--max-steps` and checkpoint more frequently.
- No gain in equation category: improve focused data quality before increasing steps.
