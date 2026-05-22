"""Recipe constants for the convergent public 0.85 SFT recipe.

Sources:
- LoRA config: dgxchen / huikang, see docs/notebook_learnings/04_top_scoring_0_85_lb.md
- completion_only_loss=True: waterjoe (the basic-SFT-group gap)
- Optimizer: 8-bit AdamW (bitsandbytes) — saves ~5.5 GB vs fp32 on 96 GB GPU.
  Note: the public 0.85 recipe uses full-precision AdamW, but fitting 30B on
  a single Blackwell (96 GB) requires 8-bit optimizer states. A small quality
  degradation is expected; promote only if CV clears the 0.86 gate.
- LR=2e-4 linear, zero warmup
- Effective batch=32 (micro=1, grad_accum=32) — per dgxchen ablation
  (micro-batch=2 costs ~0.01 LB)
- ~1 epoch / ~1000 steps on the kienngx CoT corpus

QLoRA path: when VRAM is insufficient for bf16 base (63.6 GB), set use_qlora=True
in trainer.train(). This loads the base in NF4 4-bit (~15 GB), freeing ~48 GB of
VRAM for activations. Quality may degrade slightly vs bf16; only promote adapters
that clear the CV gate.
"""

from __future__ import annotations

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

LORA_CONFIG: dict = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",   # Mamba SSM fused projection — critical for hybrid arch
        "out_proj",  # Mamba SSM output projection
        "up_proj",
        "down_proj",
        "lm_head",   # Unsloth silently drops this for MoE — must include
    ],
}

# These align with the recipe described in 04_top_scoring_0_85_lb.md.
# Tune per-run via CLI args; this dict is the documented default.
TRAINING_DEFAULTS: dict = {
    # Data / sequence
    # NOTE: TRL renamed `max_seq_length` to `max_length` in 0.13. The offline
    # packages dataset ships a post-rename version, so we use `max_length`.
    "max_length": 8192,
    "packing": True,  # short prompts (~300 chars) benefit a lot from packing

    # Optimization
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "max_grad_norm": 0.0,  # 0 = no clipping (TRL respects this)
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "optim": "adamw_bnb_8bit",  # 8-bit AdamW: saves ~5.5 GB vs fp32; needed to fit 30B on 96 GB

    # Schedule
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,  # → effective batch 32
    "num_train_epochs": 1.0,
    "max_steps": -1,  # use epochs by default; override with --max-steps to cap

    # Loss: ONLY backprop on the assistant turn (waterjoe's free-win fix)
    "completion_only_loss": True,

    # Numerics
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,

    # I/O
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "report_to": "none",
    "remove_unused_columns": False,
    "dataloader_num_workers": 2,
}

# Model-loading kwargs for the base (passed to AutoModelForCausalLM.from_pretrained).
# bf16 (NOT 4-bit), eager attention (required for the hybrid Mamba+attention arch).
BASE_MODEL_KWARGS: dict = {
    "torch_dtype": "bfloat16",
    "attn_implementation": "eager",
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

# bitsandbytes 4-bit config for QLoRA on memory-constrained GPUs (e.g. single
# Blackwell 96 GB, where bf16 base uses 63.6 GB and leaves too little for
# Mamba SSM backward activations).
# Use NF4 (best quality for LoRA fine-tuning) with bf16 compute — gradient math
# stays in bf16, only the stored weights are 4-bit. Double quant saves ~0.5 GB.
QLORA_BNB_KWARGS: dict = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",  # string; trainer resolves to torch.bfloat16
    "bnb_4bit_use_double_quant": True,
}
