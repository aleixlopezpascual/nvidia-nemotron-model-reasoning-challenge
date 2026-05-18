# 02 - Basic SFT Training Notebooks

This document analyzes five "basic SFT" (supervised fine-tuning) training notebooks from the NVIDIA Nemotron Model Reasoning Challenge. The competition asks for a LoRA adapter (rank <= 32) for the **Nemotron-3-Nano-30B (A3B BF16)** base, evaluated under vLLM with greedy decoding and answer extraction from `\boxed{...}`. All five notebooks target this same competition contract, but they vary in framework choice, LoRA configuration, data formatting, and the depth to which they engage with the underlying model architecture (a hybrid Mamba/Transformer MoE).

Three of the five are "real" SFT training pipelines that actually fine-tune LoRA adapters on the provided `train.csv`:

- `dennisfong` and `amanatar` are the same TRL `SFTTrainer` lineage — `amanatar` is a "research-backed" fork of `dennisfong` with extra knobs (NEFTune, packing, CoT augmentation, router freezing).
- `johnnyhyland` is a manual `torch` training loop using plain PEFT + `AutoModelForCausalLM` (no TRL).

The remaining two are *not* legitimate SFT runs:

- `jek1wantaufik` skips training entirely and writes a random-initialized LoRA adapter to disk (it is a synthetic-data-generation experiment that never actually trains a model).
- `mirzayasirabdullah07` is a packaging / SVD-conversion script that takes someone else's already-trained Tinker adapter and reshapes it for submission — it contains zero training code.

These two are included anyway because the prompt asks for all five; they primarily illuminate what a *non-training* baseline looks like, and the SVD-conversion logic in `mirzayasirabdullah07` is interesting in its own right (it shows how the Mamba `in_proj` matrix is decomposed into `gate_proj` + `x_proj` when training a Tinker adapter that is later remapped to PEFT format).

---

## Comparison Table

| Notebook (rank, votes) | Real SFT? | Framework | LoRA r | lora_alpha | target_modules | LR | Eff. batch | Epochs | Samples | Max seq | Packing | Quant | LB score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dennisfong (#2, 772) | Yes | TRL `SFTTrainer` + PEFT | 32 | 16 | `"all-linear"` | 2e-4 | 4 (1x4) | 1 | 600 | 1024 | False | none (bf16) | not stated |
| amanatar (#15, 176) | Yes | TRL `SFTTrainer` + PEFT | 32 | 64 | `"all-linear"` | 1e-4 | 8 (1x8) | 3 | 1000 (x2 aug -> 2000) | 1024 | True | none (bf16) | not stated |
| johnnyhyland (#12, 244) | Yes | Manual PyTorch loop + PEFT | 32 | 16 | regex `.*\.(in_proj\|out_proj\|up_proj\|down_proj)$` | 2e-4 | 4 (1x4) | 1 | 600 | 1024 | False | none (bf16) | not stated |
| jek1wantaufik (#17, 148) | No (random init) | none (writes random weights) | 32 | 64 | `[q,k,v,o,gate,up,down]_proj` | n/a | n/a | n/a | n/a (10k synth generated, unused) | n/a | n/a | n/a | not stated |
| mirzayasirabdullah07 (#20, 99) | No (packager) | n/a (SVD remap of someone else's adapter) | inherits from source adapter | inherits | rewritten to `[k,o,in,q,up,v,down,out]_proj + lm_head` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | references "0.85 LB" lineage of source adapter |

**Convergent defaults across the real SFT notebooks (dennisfong, amanatar, johnnyhyland):**
- Base model: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (the Kaggle mirror of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
- LoRA rank: 32 (max allowed by competition).
- Precision: bf16 throughout, **no bitsandbytes quantization** in any notebook.
- Max seq length: 1024.
- Subsample of `train.csv` (600-1000 rows out of 9500) — none use the full training set.
- gradient_checkpointing enabled.
- Identical Triton/PTXAS workarounds for Kaggle Blackwell GPUs (B200/B100 era).

---

## dennisfong - NVIDIA Nemotron - SFTTrainer training (rank #2, 772 votes)

This is the canonical, minimal "TRL SFTTrainer" baseline that the rest of the leaderboard's basic-SFT notebooks fork. It is intentionally bare-bones: get something training end-to-end on Kaggle's Blackwell GPUs.

- **Approach**: Standard SFT via TRL `SFTTrainer`. Loss is computed over the *full* formatted text (prompt + answer joined with chat template) — no completion-only masking, no `DataCollatorForCompletionOnlyLM`, no `formatting_func` that distinguishes prompt vs response. This means the model is trained to *also* model the prompt tokens, not just the answer.
- **Framework**: HuggingFace `transformers` + PEFT + TRL `SFTTrainer` / `SFTConfig`. No Accelerate config, no DeepSpeed, no Unsloth.
- **LoRA config**:
  ```python
  lora_config = LoraConfig(
      r=32,
      lora_alpha=16,            # alpha/r = 0.5
      target_modules="all-linear",
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
  )
  ```
  Notably `lora_alpha=16` is *half* the rank — a low-magnitude adaptation. `target_modules="all-linear"` lets PEFT auto-discover every `nn.Linear` in the model (covers attention `q/k/v/o`, MoE expert MLPs `up/down/gate`, *and* Mamba `in_proj/out_proj` since those are also `nn.Linear`). No `modules_to_save`.
- **Quantization**: None. Model is loaded directly in `torch.bfloat16` via `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`. No `BitsAndBytesConfig`.
- **Base model + tokenizer**: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` downloaded via `kagglehub`. Tokenizer loaded with `trust_remote_code=True`. `pad_token` set to `eos_token` if missing. The model is patched to force the "slow path" by setting `mod.is_fast_path_available = False` on any imported `modeling_nemotron_h` module — required to bypass broken CUDA kernels for the Mamba layers on the Kaggle Blackwell GPU.
- **Data formatting**:
  ```python
  def build_training_text(example):
      prompt = example["prompt"]
      answer = example["answer"]
      user_msg = prompt + "\nPut your final answer inside \\boxed{}."
      assistant_msg = f"\\boxed{{{answer}}}"
      messages = [
          {"role": "user", "content": user_msg},
          {"role": "assistant", "content": assistant_msg},
      ]
      text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      return {"text": text}
  ```
  - **No system prompt**.
  - The user message instructs `Put your final answer inside \boxed{}.`
  - The assistant message *is just* `\boxed{ANSWER}` — no chain-of-thought, no `<think>` tags.
  - Falls back to a manual `<|im_start|>user ... <|im_end|>` template if `apply_chat_template` raises.
  - **No train/val split** — all 600 subsampled rows go to training.
- **Training hyperparams**:
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=4` (effective batch 4).
  - `num_train_epochs=1`.
  - `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`.
  - `optim="adamw_torch"`, `max_grad_norm=1.0`. No `weight_decay` set explicitly.
  - `bf16=True`, `packing=False`, `max_length=1024`.
  - `gradient_checkpointing=True` with `use_reentrant=True` (the "forgiving" backend that tolerates the patched slow path).
  - `dataset_text_field="text"`.
  - `save_strategy="no"`, `report_to="none"`.
- **Hardware**: Kaggle GPU (Blackwell-class — implied by the `ptxas-blackwell` workarounds). The Triton/PTXAS fix copies `ptxas-blackwell` to `/tmp` with exec permissions because `/kaggle/input` is read-only.
- **Reported score / LB**: Not stated in the notebook.
- **Notable design choices**:
  - Extensive Kaggle environment plumbing (Triton patches, PTXAS exec-bit fix, `rmsnorm_fn` monkey-patch in `mamba_ssm` to avoid Triton kernels entirely).
  - 600-sample subsample (about 6% of available data) for "rapid iteration" — *not* a full-data run.
  - `packing=False` — trains one example per sequence, padded to 1024.
  - Saves adapter via `trainer.model.save_pretrained()` and zips it flat (files at zip root, no subdirectory) — matches the competition packaging contract.

---

## amanatar - NVIDIA Nemotron - SFTTrainer training (rank #15, 176 votes)

A direct fork of `dennisfong` that piles on every "research-backed" knob: NEFTune, packing, CoT data augmentation, system prompt, MoE router freezing, higher alpha, more epochs. Same overall TRL + PEFT scaffolding.

- **Approach**: Same TRL `SFTTrainer` SFT (full-text loss, no completion masking) but with two important differences from `dennisfong`:
  1. The data is **2x augmented** — each training row is rendered both as a "direct answer" example (`<think></think> \boxed{ANS}`) and as a "CoT trace" example (`<think>...reasoning...</think> \boxed{ANS}`), then concatenated and shuffled.
  2. **MoE router parameters are explicitly frozen** before applying PEFT, to prevent expert-routing collapse:
     ```python
     for name, param in model.named_parameters():
         if "router" in name.lower():
             param.requires_grad = False
     ```
- **Framework**: TRL `SFTTrainer` / `SFTConfig`, PEFT, HF Transformers. Same stack as `dennisfong`.
- **LoRA config**:
  ```python
  LORA_RANK   = 32
  LORA_ALPHA  = 64    # alpha/r = 2.0 (NVIDIA-recommended, per comment)
  LORA_DROPOUT = 0.0  # NEFTune provides regularization
  lora_config = LoraConfig(
      r=LORA_RANK,
      lora_alpha=LORA_ALPHA,
      target_modules="all-linear",
      lora_dropout=LORA_DROPOUT,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
  )
  ```
  - **Key divergence from dennisfong/johnnyhyland**: `lora_alpha=64` (alpha/r=2.0) vs their `lora_alpha=16` (alpha/r=0.5). This is a 4x higher effective adaptation magnitude.
  - **`lora_dropout=0.0`** (vs 0.05 in dennisfong/johnnyhyland) — the comment explicitly justifies this by saying NEFTune provides the regularization.
- **Quantization**: None. bf16 direct-load, same as dennisfong.
- **Base model + tokenizer**: Same `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`. `trust_remote_code=True`, pad=eos fallback. Same `modeling_nemotron_h.is_fast_path_available = False` patch.
- **Data formatting**: Uses a **system prompt** (unlike dennisfong):
  ```python
  SYSTEM_PROMPT = (
      "You are an expert at pattern recognition and reasoning. "
      "Analyze the examples carefully, identify the transformation rule, "
      "apply it to the new input, and put your final answer inside \\boxed{}."
  )
  ```
  Two assistant message formats:
  - Direct (Version A): `assistant_msg = f"<think>\n</think>\n\\boxed{{{answer}}}"` — empty think tags, just the boxed answer.
  - CoT (Version B): `assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"` where `cot` is a synthesized trace built by `_build_cot_trace(prompt, answer)`. The trace is generated by a heuristic classifier (`_classify`) that buckets puzzles into binary / numeric / text / mixed based on regex on the prompt, then emits a templated 3-5 line "reasoning" block citing the first few `->` examples from the prompt.
  - Note: the synthesized CoT is **not real reasoning** — it is a templated string that includes the answer at the end. It exists to teach the *structure* of `<think>...</think>\boxed{}` rather than actual problem solving.
  - **No explicit train/val split**.
  - The final dataset is `concatenate_datasets([ds_direct, ds_cot]).shuffle(seed=42)` -> 2000 examples from a 1000-row subsample.
- **Training hyperparams**:
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=8` (effective batch 8, double dennisfong).
  - `num_train_epochs=3`, `learning_rate=1e-4` (half dennisfong, justified as "lower LR for multi-epoch stability").
  - `lr_scheduler_type="cosine"`, `warmup_ratio=0.05` (vs 0.1).
  - `optim="adamw_torch"`, `weight_decay=0.01`, `max_grad_norm=1.0`.
  - `bf16=True`, **`packing=True`** (different from dennisfong's `packing=False`), `max_length=1024`.
  - **`neftune_noise_alpha=5.0`** — adds noise to embeddings during training for regularization. This is the headline "research-backed" knob.
  - `gradient_checkpointing=True` with `use_reentrant=True`.
- **Hardware**: Same Kaggle Blackwell setup, same Triton/PTXAS plumbing.
- **Reported score / LB**: Not stated.
- **Notable design choices**:
  - Aggressive use of TRL features: NEFTune + packing + multi-epoch.
  - Templated synthetic CoT is more of a "format teacher" than a reasoning teacher.
  - MoE router freezing is a real architectural insight — Nemotron-3-Nano-30B is an MoE model, and LoRA on the router can collapse expert utilization. Even though LoRA wouldn't normally touch the router, this notebook defensively freezes it before `get_peft_model`.
  - `lora_alpha=64` is the spiciest divergence — combined with no dropout and NEFTune, this is a "trust the regularization tricks, push adaptation magnitude" bet.

---

## johnnyhyland - NVIDIA Nemotron - training (rank #12, 244 votes)

A pre-TRL fork: same hyperparameters as `dennisfong`, but throws out the TRL `SFTTrainer` in favor of a hand-rolled PyTorch training loop with PEFT + plain `AutoModelForCausalLM`. The author flags it as "pretty janky" in the notebook header.

- **Approach**: Standard SFT (causal LM loss over the *entire* tokenized sequence; padding masked via `labels[mask == 0] = -100`). No completion masking — the prompt tokens contribute to the loss equally with the answer tokens. Loss computed as `outputs.loss` from `model(**batch)` with `labels` field.
- **Framework**: Pure `transformers` + PEFT + manual `torch.utils.data.DataLoader` and `torch.optim.AdamW`. **No TRL, no Accelerate, no DeepSpeed, no Unsloth.**
- **LoRA config**:
  ```python
  lora_config = LoraConfig(
      r=32,
      lora_alpha=16,            # alpha/r = 0.5, same as dennisfong
      target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
  )
  ```
  - **Key divergence**: target_modules is an **explicit regex** that hits only `in_proj`, `out_proj`, `up_proj`, `down_proj`. This **excludes** attention `q/k/v/o_proj` and the MoE `gate_proj`. Effectively:
    - Mamba layers: `in_proj`, `out_proj` (the two main Mamba block linears).
    - MLP/MoE experts: `up_proj`, `down_proj` (but **not** `gate_proj`).
    - Attention: none of `q_proj`, `k_proj`, `v_proj`, `o_proj`.
  - This is a substantially narrower adaptation than `"all-linear"`. It is also closer to how Tinker-trained adapters are structured (gate_proj + x_proj get merged into in_proj during conversion).
  - Same `lora_dropout=0.05` and `lora_alpha=16` as dennisfong.
- **Quantization**: None. bf16 direct load.
- **Base model + tokenizer**: Same Kaggle mirror, same `trust_remote_code=True`, same pad=eos, same `is_fast_path_available=False` patch.
- **Data formatting**:
  ```python
  def build_training_text(prompt: str, answer: str) -> str:
      user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."
      assistant_msg = f"\\boxed{{{answer}}}"
      messages = [
          {"role": "user", "content": user_msg},
          {"role": "assistant", "content": assistant_msg},
      ]
      text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      return text
  ```
  - Almost identical to dennisfong (the only diff is "Please put" vs "Put").
  - **No system prompt**, no `<think>` tags, plain `\boxed{ANSWER}` assistant message.
  - Tokenization happens up-front in a custom `SFTDataset.__init__` (precomputed `input_ids`, `attention_mask`, `labels`), with `truncation=True`, `max_length=1024`, `padding="max_length"` — this means **every sequence is padded to 1024** (wasteful, but simple). Pad positions have `labels=-100` so they are masked out of the loss.
  - No train/val split.
- **Training hyperparams**:
  - `per_device_train_batch_size=1`, `GRAD_ACCUM=4` (effective 4).
  - `NUM_EPOCHS=1`, `LR=2e-4` — identical to dennisfong.
  - Optimizer: `torch.optim.AdamW(..., lr=2e-4, weight_decay=0.01)` over only `param.requires_grad` parameters (so only LoRA + non-frozen).
  - No LR scheduler. No warmup. No early stopping. Manual gradient clipping `torch.nn.utils.clip_grad_norm_(..., 1.0)` before each `optimizer.step()`.
  - `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` — note **`use_reentrant=False`**, different from dennisfong/amanatar's `True`. Justified by comment as needed to avoid "tensor-count mismatch with our patched slow path" in this manual-loop setup.
  - No `packing`. No NEFTune. No `weight_decay` annealing.
- **Hardware**: Same Kaggle Blackwell setup, same Triton/PTXAS plumbing, but the Triton fixes are split across two cells (the `rmsnorm_fn` patch happens in cell 3, *after* model load).
- **Reported score / LB**: Not stated.
- **Notable design choices**:
  - Manual training loop — explicitly an "is anything training at all?" experiment per the notebook header. The author asks for community feedback.
  - Narrower `target_modules` regex (drops `q/k/v/o_proj` and `gate_proj`) — the most architecturally opinionated choice of the three real SFT notebooks. May reflect a belief that the Mamba `in_proj/out_proj` and expert `up/down_proj` are the load-bearing adaptation surfaces.
  - `use_reentrant=False` for gradient checkpointing — opposite of dennisfong/amanatar.
  - No LR scheduler / warmup — pure constant 2e-4 across the run.

---

## jek1wantaufik - NVIDIA Nemotron Model Reasoning Challenge (rank #17, 148 votes)

This notebook is **not actually a training notebook**. It generates synthetic puzzle data (10k rows of binary/numeric/text rule-induction problems), then writes a **randomly-initialized** LoRA adapter to disk and zips it. No model is ever loaded or fine-tuned. It is included here for completeness and as a contrast — what a "submit something, anything" baseline looks like.

- **Approach**: No training. No SFT objective. The synthetic data generation pipeline classifies the real training answers by regex (`binary` / `numeric` / `text` / `mixed`), then generates proportional synthetic puzzles (`gen_binary` does XOR/AND/OR/reverse on random 8-bit integers; `gen_numeric` does `a*b+b` and `a^2+b`; `gen_text` does Caesar-shift word ciphers). The synthetic data is concatenated with the real training prompts and... never used.
- **Framework**: Plain `torch` to save random tensors. No PEFT, no transformers, no TRL. The adapter is hand-built as a dict.
- **LoRA config** (written to `adapter_config.json` manually, no actual training):
  ```python
  adapter_config = {
      "peft_type": "LORA",
      "task_type": "CAUSAL_LM",
      "r": 32,
      "lora_alpha": 64,
      "lora_dropout": 0.05,
      "bias": "none",
      "target_modules": [
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"
      ],
      "inference_mode": True,
  }
  ```
  - r=32, alpha=64 (alpha/r=2.0) — matches `amanatar`'s ratio.
  - target_modules is the **standard transformer set** (`q/k/v/o_proj` for attention + `gate/up/down_proj` for MLP). It **omits Mamba's `in_proj`/`out_proj`** entirely — meaning even if this adapter were applied, it would only touch the attention/MLP fraction of the hybrid model.
- **Quantization**: n/a.
- **Base model + tokenizer**: Never loaded. The adapter is constructed without ever instantiating Nemotron.
- **Data formatting**:
  ```python
  def format_prompt(p):
      return f"""
  You are a reasoning system.
  Discover the rule from the examples
  and apply it to the final input.
  {p}
  Return the final result inside \\boxed{{}}.
  """.strip()
  ```
  Applied to both real and synthetic prompts. There is a system-prompt-flavored preamble, and the `\boxed{}` instruction is in the user message. No assistant template, no chat template, no tokenization.
- **Training hyperparams**: None — there is no training step.
- **Adapter weights**: Random Gaussian noise scaled by 0.01:
  ```python
  rank = 32; hidden = 128
  for module in target_modules:
      A = torch.randn(rank, hidden) * 0.01
      B = torch.randn(hidden, rank) * 0.01
      weights[f"{module}.lora_A.weight"] = A
      weights[f"{module}.lora_B.weight"] = B
  ```
  - `hidden=128` is a *fake* hidden dim — the Nemotron base model's actual hidden size is much larger, so these weights would not load against a real base model. The adapter is dimensionally incompatible with the actual `metric/nemotron-3-nano-30b-a3b-bf16` weights.
  - The shape `(rank, hidden)` for `lora_A` is also reversed vs PEFT convention (PEFT expects `lora_A.weight` of shape `(rank, in_features)`).
- **Hardware**: Runs on CPU (no GPU needed since nothing is trained).
- **Reported score / LB**: Not stated. Given the random weights and dimension mismatch, this would either fail to load or score at chance.
- **Notable design choices**:
  - The *synthetic data generator* is the only interesting part — it shows a partial taxonomy of puzzle types and a templated approach to generating proxy data. Could be a starting point for legitimate data augmentation if connected to a real training loop.
  - Demonstrates a failure mode in the "low-bar baseline" approach: producing a `submission.zip` that satisfies the file-format contract but does not load against the actual base model.

---

## mirzayasirabdullah07 - NVIDIA Nemotron Model Notebook (rank #20, 99 votes)

Also **not a training notebook**. This is a "packaging + format-conversion" script: it picks up a pre-trained Tinker-style adapter (from `kienngx/nemotron-nano-30b-trained/triton/tinker-adapter`), rewrites the keys/config to match what the competition evaluator wants, runs an **SVD-based merge** to collapse Mamba's `gate_proj`+`x_proj` into a single `in_proj` LoRA pair, and zips the result. Included here because the SVD merge logic teaches us about the Mamba LoRA structure.

- **Approach**: No SFT. The pipeline starts from `kienngx`'s Tinker adapter (which was trained with a different LoRA target layout — Tinker treats Mamba `in_proj` as two separate logical projections, `gate_proj` and `x_proj`) and converts it to the PEFT format the competition's vLLM evaluator expects (single `in_proj` projection).
- **Framework**: `safetensors` + `torch.linalg.svd` for the key remap. No training framework involved.
- **LoRA config** (rewritten on top of the source adapter):
  ```python
  cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  cfg["inference_mode"] = True
  cfg["lora_dropout"] = 0.0
  cfg["target_modules"] = [
      "k_proj", "o_proj", "in_proj", "q_proj",
      "up_proj", "v_proj", "down_proj", "out_proj", "lm_head",
  ]
  ```
  - `r` / `lora_alpha` are inherited from the source adapter's `adapter_config.json` — not set here.
  - `target_modules` now includes `lm_head` — only notebook in this set to do so (since `lm_head` is a `nn.Linear` that PEFT can wrap; including it raises the trainable param count).
  - `target_modules` is the **full union** of attention `(q,k,v,o)`, MoE expert `(up,down)`, and Mamba `(in_proj, out_proj)` linears, *plus* `lm_head`. This is the most comprehensive target set in any notebook we read.
  - `lora_dropout` forced to 0.0 for inference.
- **Quantization**: n/a.
- **Base model**: Sets `base_model_name_or_path` to the official `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` HF repo name in the config (regardless of what was originally there).
- **Key remap logic**:
  - Keys are renamed: `base_model.model.model -> base_model.model.backbone`.
  - MoE experts in the source adapter are stored as `experts.w1` / `experts.w2` (Tinker convention); these are *unfused* into per-expert PEFT entries `experts.{i}.up_proj` and `experts.{i}.down_proj`. The unfusing expands `lora_A` along the expert dimension if it has shape `(1, ...)`.
  - For Mamba blocks, the source adapter has separate LoRA factors on `gate_proj` and `x_proj`. These are concatenated and SVD-truncated back into a single `in_proj` LoRA pair, with adaptive rank selection keeping 98% of the singular-value variance (capped at original rank).
  ```python
  A_cat = torch.cat([gate_A, x_A], dim=0)
  B_block = torch.zeros(in_proj_dim, 2 * rank)
  B_block[:gate_B.shape[0], :rank] = gate_B
  B_block[gate_B.shape[0]:gate_B.shape[0] + x_B.shape[0], rank:] = x_B
  merged_matrix = B_block.float() @ A_cat.float()
  U, S, Vh = torch.linalg.svd(merged_matrix, full_matrices=False)
  cumsum = torch.cumsum(S, dim=0) / S.sum()
  k = torch.searchsorted(cumsum, 0.98).item() + 1
  k = min(k, rank)
  new_B = U[:, :k] * S[:k].unsqueeze(0)
  new_A = Vh[:k, :]
  ```
- **Reported score / LB**: The notebook references "0.85 LB" lineage (the Tinker-trained adapter source) but does not show a fresh evaluation.
- **Notable design choices**:
  - Reveals that Tinker uses a different LoRA layout from PEFT for hybrid Mamba models. Anyone who trains in Tinker has to do this kind of SVD conversion before submitting.
  - Adaptive-rank SVD truncation (98% variance) is a clever bandwidth-saver: rather than keeping the full `2*rank` after merging, it picks the smallest `k` that captures most of the energy, then caps `k <= rank` to stay within the competition limit.
  - The expert `w1 -> up_proj` and `w2 -> down_proj` mapping is the standard Mixtral/MoE renaming convention.

---

## Synthesis

A few patterns are clearly converging across the actual SFT notebooks (dennisfong / amanatar / johnnyhyland), and a few are clearly experimental.

**Strong defaults (likely safe to copy):**

1. **Base model loading: bf16 + trust_remote_code + slow path patch.** Every real SFT notebook uses `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, trust_remote_code=True)` and patches `mod.is_fast_path_available = False` on `modeling_nemotron_h`. This is non-negotiable on Kaggle Blackwell — the fast CUDA path is broken there.

2. **No bitsandbytes quantization in any notebook.** Despite the 30B model size (MoE A3B active-3B-per-token, so VRAM is more tractable than dense 30B), nobody falls back to 4-bit. Direct bf16 load + LoRA fits.

3. **LoRA rank = 32, the competition max.** All five "use" 32. Nobody experiments with lower ranks here.

4. **Greedy-eval-friendly answer template: `\boxed{ANSWER}`.** Every notebook ends the assistant message with `\boxed{...}` because that is what the evaluator regex extracts. The user-message instruction "Put your final answer inside `\boxed{}`" is also universal.

5. **Chat template via `tokenizer.apply_chat_template(messages, ..., add_generation_prompt=False)`** with a manual `<|im_start|>` fallback. The fallback is identical text across notebooks (copy-pasted lineage).

6. **gradient_checkpointing + per_device_train_batch_size=1.** Memory is the binding constraint; nobody fits batch > 1 per GPU.

7. **No completion-only loss masking.** Every real SFT notebook computes loss over the entire formatted text (prompt + answer). Not one of them uses `DataCollatorForCompletionOnlyLM` or a TRL `response_template`. This is a significant collective choice — the model is being trained to "predict the prompt" as well as the answer, which on this dataset (where prompts are long few-shot puzzles and answers are short) means most of the loss signal is on prompt tokens, not the boxed answer.

8. **Identical Kaggle plumbing: PTXAS exec-bit fix + `rmsnorm_fn` monkey-patch + `TRITON_PTXAS_PATH`.** All three real SFT notebooks copy the same 30-line setup block.

**Active divergences worth investigating:**

9. **`lora_alpha`: 16 vs 64.** dennisfong and johnnyhyland use `alpha=16` (alpha/r=0.5, a *low-magnitude* adaptation). amanatar uses `alpha=64` (alpha/r=2.0). This is a 4x difference in effective LoRA magnitude. Convention in the broader LoRA literature is split: many recipes use `alpha=2*r`, others use `alpha=r` or `alpha=r/2`. Worth A/B testing on real LB.

10. **`target_modules`: `"all-linear"` vs explicit regex vs explicit list.** dennisfong/amanatar use `"all-linear"` (lets PEFT discover every Linear, including Mamba `in_proj`/`out_proj`, attention `q/k/v/o`, and MoE expert `up/down/gate`). johnnyhyland uses a narrower regex that drops attention `q/k/v/o` and the `gate_proj`. jek1wantaufik (random) uses the standard transformer set and omits Mamba. mirzayasirabdullah07 (packager) uses the full set + `lm_head`. There is no consensus on whether to adapt Mamba, attention, or both.

11. **`lora_dropout`: 0.05 vs 0.0.** dennisfong/johnnyhyland use 0.05. amanatar uses 0.0, justifying it via NEFTune. The "use NEFTune instead of dropout" claim is plausible but unverified on this dataset.

12. **Packing: off vs on.** dennisfong/johnnyhyland: `packing=False` (one example per sequence, padded to 1024). amanatar: `packing=True` (concatenates examples to fill 1024-token sequences, claimed 2-3x speedup). For short answers + variable-length prompts, packing should give a real wall-clock win and is probably under-used.

13. **Epochs / LR / effective batch.** dennisfong & johnnyhyland: 1 epoch, LR=2e-4, eff. batch 4. amanatar: 3 epochs, LR=1e-4, eff. batch 8. The amanatar choice ("lower LR for multi-epoch stability") is the standard recipe; the 1-epoch + 2e-4 baseline is aggressive and may not converge.

14. **NEFTune + MoE router freezing — amanatar-only.** These are real techniques. Router freezing in particular is architecturally motivated (LoRA on `"all-linear"` would otherwise wrap the router gate linear; freezing it prevents routing collapse during fine-tune). Anyone using `target_modules="all-linear"` should freeze routers.

15. **CoT data augmentation: templated, not real reasoning.** amanatar's `_build_cot_trace` produces a templated `<think>...</think>` block that names the puzzle category and the visible examples, then states the answer. It is *not* a real reasoning trace. Its likely benefit is teaching the *output structure* (`<think>...</think>\boxed{...}`) so that during eval the model's reasoning tokens are well-bounded — but it doesn't add genuine reasoning supervision. A real CoT dataset (e.g. from `kienngx/nemotron-sft-reasoning-trajectories-dataset`, rank #19) would be a strict improvement.

16. **Subsampling: nobody uses the full 9500-row training set.** dennisfong/johnnyhyland use 600 rows, amanatar uses 1000 (2x augmented). All three are at < 11% of available data. The "real" winning notebooks (huikang, kienngx, dgxchen) referenced in the notebook index almost certainly use the full set — basic-SFT folks are leaving signal on the table.

17. **Tinker vs PEFT layout for Mamba blocks.** The `mirzayasirabdullah07` SVD-conversion machinery shows that Tinker treats Mamba's `in_proj` as two logical projections (`gate_proj`, `x_proj`) and trains separate LoRA factors on each. The competition's vLLM evaluator expects a single `in_proj` LoRA. If you train in Tinker, you must SVD-merge before submission. If you train in PEFT (as all three real notebooks here do), the `"all-linear"` or `in_proj` target hits the unified projection directly and no conversion is needed.

18. **Submission packaging is identical and trivial.** All five notebooks produce a *flat* `submission.zip` (files at zip root, no subdirectory prefix) containing `adapter_config.json` + `adapter_model.{bin,safetensors}`. This part is solved.
