"""Main training function: load base + LoRA, build dataset, run TRL SFTTrainer.

Designed for a single GPU instance (A100 80 GB / H100 / RTX PRO 6000 Blackwell).
30B base in bf16 needs ~60 GB just for the weights; LoRA + activations + grad
checkpointing easily fits in 80 GB.
"""

from __future__ import annotations

from pathlib import Path

from .recipe import BASE_MODEL, BASE_MODEL_KWARGS, LORA_CONFIG, TRAINING_DEFAULTS


def train(
    data_csv: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = BASE_MODEL,
    max_steps: int = -1,
    learning_rate: float | None = None,
    num_train_epochs: float | None = None,
    sample_n: int | None = None,
    extra_training_kwargs: dict | None = None,
) -> None:
    """Run SFT and write the LoRA adapter to ``output_dir``.

    Args:
        data_csv: path to the CoT CSV (kienngx schema).
        output_dir: where to save adapter checkpoints.
        base_model: HF model id or local path for the base.
        max_steps: -1 = epochs-driven, otherwise hard cap on optimizer steps.
        learning_rate: override the recipe default (2e-4).
        num_train_epochs: override the recipe default (1.0).
        sample_n: optional row subsample (for quick iteration runs).
        extra_training_kwargs: passed through to SFTConfig as overrides.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .data import load_sft_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trainer] loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # The base usually ships a chat template; if not, the user must supply one.
    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Base model tokenizer has no chat_template — this is required to format "
            "training rows with `enable_thinking=True` matching the host metric."
        )

    print(f"[trainer] loading base model {base_model} (bf16, eager attn)")
    model_kwargs = dict(BASE_MODEL_KWARGS)
    model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    # Required for LoRA training with checkpointing on PEFT-wrapped models.
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    _apply_nemotron_h_blackwell_patches()

    print("[trainer] building LoRA config")
    peft_config = LoraConfig(**LORA_CONFIG)

    print("[trainer] loading + formatting dataset")
    train_dataset = load_sft_dataset(data_csv, tokenizer, sample_n=sample_n)
    print(f"[trainer] {len(train_dataset)} train rows")

    print("[trainer] assembling SFTConfig")
    sft_kwargs: dict = dict(TRAINING_DEFAULTS)
    sft_kwargs["output_dir"] = str(output_dir)
    sft_kwargs["max_steps"] = max_steps
    if learning_rate is not None:
        sft_kwargs["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = num_train_epochs
    if extra_training_kwargs:
        sft_kwargs.update(extra_training_kwargs)

    sft_config = SFTConfig(**sft_kwargs)

    print("[trainer] starting training")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"[trainer] saving final adapter to {output_dir / 'final'}")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print("[trainer] done.")


def _apply_nemotron_h_blackwell_patches() -> None:
    """Post-model-load patches needed on Blackwell SMs (no-op elsewhere).

    Replaces the Triton rmsnorm_fn in modeling_nemotron_h with a pure-PyTorch
    fallback and disables Mamba's fast path. Triggered every train() call so
    it's reliably applied regardless of how the caller wired up imports.
    """
    import sys

    try:
        from blackwell_setup import (  # type: ignore
            patch_rmsnorm_with_pure_pytorch,
            disable_nemotron_fast_path,
        )
    except ImportError:
        return  # blackwell_setup not on sys.path — assume not running on Blackwell

    n_rms = patch_rmsnorm_with_pure_pytorch()
    n_fp = disable_nemotron_fast_path()
    print(f"[trainer] blackwell patches: rmsnorm_fn={n_rms}, is_fast_path_available=False x{n_fp}")
