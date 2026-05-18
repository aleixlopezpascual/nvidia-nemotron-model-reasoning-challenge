"""Recipe constants for the convergent public 0.85 SFT recipe.

Sources:
- LoRA config: dgxchen / huikang, see docs/notebook_learnings/04_top_scoring_0_85_lb.md
- completion_only_loss=True: waterjoe (the basic-SFT-group gap)
- Optimizer: full-precision AdamW (NOT 8-bit), betas=(0.9, 0.95), eps=1e-8,
  weight_decay=0, no grad clip — per huikang's hand-rolled loop
- LR=2e-4 linear, zero warmup
- Effective batch=32 (micro=1, grad_accum=32) — per dgxchen ablation
  (micro-batch=2 costs ~0.01 LB)
- ~1 epoch / ~1000 steps on the kienngx CoT corpus
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
    "optim": "adamw_torch",  # full-precision AdamW (NOT 8-bit)

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
