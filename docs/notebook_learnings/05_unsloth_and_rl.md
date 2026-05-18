# 05 — Unsloth SFT & GRPO/RL Notebook Learnings

This file analyses two complementary Kaggle notebooks for the NVIDIA Nemotron-3-Nano-30B
Reasoning Challenge:

1. **`llkh0a/nemotron-unsloth-sft-training-3-30-2`** — a heavily-iterated Unsloth + vLLM SFT
   baseline that produces a LoRA r=32 adapter. This is the largest notebook in our pool
   (187 KB), almost entirely because of seven hand-written per-category trace generators
   (deterministic CoT synthesisers) plus a long vLLM validation cell.

2. **`johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster`** — the only notebook in our set
   that runs **GRPO (RL)**. It uses TRL's `GRPOTrainer` on top of an SFT warm-start with
   three custom reward functions (cosine-scaled accuracy + format + length). The author is
   honest that "time will tell results from GRPO".

Both notebooks share several design choices that are clearly hardening into community
consensus for this competition: rank-32 LoRA, `\boxed{}`-aligned SFT targets,
`adamw_8bit` optimiser, deterministic puzzle solvers used to *synthesise* training CoT,
exact mirroring of the eval-time `BOXED_INSTRUCTION` suffix, and gradient checkpointing.
They diverge on (a) Unsloth vs vanilla `transformers`, (b) LoRA target-module strategy,
and (c) whether to bother with RL at all.

---

## Notebook 1 — `llkh0a / nemotron-unsloth-sft-training-3-30-2` (Unsloth SFT)

- **Path**: `notebooks/llkh0a__nemotron-unsloth-sft-training-3-30-2/nemotron-unsloth-sft-training-3-30-2.ipynb`
- **Author**: `llkh0a`
- **Strategy** (verbatim from the title cell): "Supervised Fine-Tuning (SFT) on the full
  training set using LoRA"
- **Iteration index**: `3-30-2` — this is the author's 30th-ish iteration on the same
  pipeline, version 2. The notebook also tracks named pretrained adapter dirs from earlier
  iterations (`3-23`, `3-28`, `3-30-1`) for warm-start.
- **Votes / LB**: vote count not stored in the local repo metadata, and no reported LB
  score is committed inside the notebook (training outputs were cleared before
  publishing). Treat as a high-effort baseline, not a verified leaderboard recipe.

### 1.1 What's special about this Unsloth setup?

The model is loaded **without quantisation** — both `load_in_4bit` and `load_in_8bit` are
`False`, and the dtype is explicit bfloat16. The author leaves a comment noting they are
on an "RTX PRO 6000 Blackwell" (so plenty of VRAM headroom; 4-bit is not required).

Critically the author forces `attn_implementation = "eager"` and
`unsloth_force_compile = False`, presumably to avoid Flash-Attention / torch.compile
incompatibilities with the hybrid Mamba+attention Nemotron architecture.

Verbatim `FastLanguageModel.from_pretrained` call (cell 40):

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LEN, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    trust_remote_code = True,
    unsloth_force_compile = False,
    attn_implementation = "eager",
    dtype = torch.bfloat16,  # explicit bf16; RTX PRO 6000 Blackwell supports bf16
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)
```

Notable absences vs. the canonical "Unsloth GRPO" recipe: `fast_inference = True` and
`gpu_memory_utilization` are **not** set on `from_pretrained` because this author runs
vLLM as a separate engine (in a later cell) rather than using Unsloth's built-in vLLM
adapter. That separation is why they need an explicit aggressive GPU cleanup cell before
spinning up vLLM (cell 59) — Unsloth + a fresh `LLM(...)` on the same GPU is OOM-prone.

`MAX_SEQ_LEN = 3500` (training-time) while the eval-time `MAX_MODEL_LEN = 8192`. That
gap is deliberate: training trace lengths fit in ~3.5 k tokens, but inference must
tolerate the full 8 k input + `MAX_NEW_TOKENS = 7680`.

### 1.2 LoRA config

```python
LORA_RANK    = 32       # the competition cap
LORA_ALPHA   = 16       # alpha < rank (so effective scaling = 0.5)
LORA_DROPOUT = 0.1

target_modules = [
    'out_proj', 'v_proj', 'q_proj', 'down_proj', 'embed_tokens',
    'k_proj', 'in_proj', 'up_proj', 'o_proj', 'lm_head', 'gate_proj',
]
```

The target list is unusually broad: it includes both Mamba projections
(`in_proj`, `out_proj`), GQA attention (`q_proj`/`k_proj`/`v_proj`/`o_proj`), the MLP
expert projections (`gate_proj`/`up_proj`/`down_proj`) **and** the embedding/head
(`embed_tokens`, `lm_head`). Combined with rank 32 across the whole 30B model the author
estimates "~800 M trainable params" in the comment of cell 42.

This is the *opposite* of johnnyhyland's "sensitive-layers-only" philosophy (see
Notebook 2). It will use noticeably more VRAM but should give the LoRA more capacity to
shift the output distribution — particularly important because tying `lm_head` lets you
correct logit biases that pure attention/MLP LoRA cannot.

The wrapper is created via `FastLanguageModel.get_peft_model(...)` with
`use_gradient_checkpointing = "unsloth"` (Unsloth's custom recomputation, more memory
efficient than the HF default), `bias = "none"`, and `random_state = 42`.

There is also a careful warm-start path that **always** creates a fresh trainable LoRA
then loads weights from a previous adapter using `peft.load_peft_weights` +
`set_peft_model_state_dict`. The notebook calls out the trap explicitly:

> "Root-cause: `PeftModel.from_pretrained()` defaults to `is_trainable=False`, which
> freezes ALL LoRA params -> 0 trainable params."

This pattern is worth stealing: do not warm-start via `PeftModel.from_pretrained`;
instantiate a trainable adapter and *then* `set_peft_model_state_dict`.

### 1.3 Data

- **Source**: official `train.csv` (9 500 rows) plus a custom hold-out file
  `splited.csv` (column `HOLD_OUT == True` defines the validation set).
- **Per-category sampling caps** (cell 9):

  ```python
  TRAIN_SAMPLES_PER_CATEGORY = {
      "bit_manipulation": 1400,   # max 1602
      "text_decryption" : 1300,   # max 1576
      "unit_conversion" : 200,    # max 1594  # de-emphasised
      "numeral_system"  : 200,    # max 1576
      "gravity_physics" : 200,    # max 1597
      "symbol_transform": 10,     # max 823   # almost dropped
      "numeric_equation": 600,    # max 732
  }
  ```

  Total training set ≈ 3 710 rows (a deliberate ~60 % subsample of the available data).
  The author concentrates capacity on bit-manipulation and text-decryption (the two
  hardest categories) and almost ignores `symbol_transform` (10 rows).

- **Format**: chat-template with `enable_thinking=True`, response = `<think>{trace}</think>\n\n\boxed{answer}`.
- **Trace synthesis**: per-category deterministic solvers in cells 18–30
  (`solve_numeral_with_trace`, `solve_gravity_with_trace`,
  `solve_unit_conversion_with_trace`, `generate_text_decryption_trace`,
  `generate_bit_manipulation_trace`, `generate_symbol_transform_trace`,
  `generate_numeric_equation_trace`). These are 2 KB – 9 KB of hand-written code per
  category and are the reason the notebook is 187 KB.
- **Custom trace overrides** (cell 9): for `bit_manipulation` and `numeric_equation` the
  author also loads pre-computed CoT traces from a Kaggle dataset
  (`bit_manipulation_traces_v4.csv`, `numeric_equation_traces_new.csv`). These appear to
  be distilled from a stronger model (the column is named `trace_from_model`). When a
  sample id is present in the lookup it overrides the deterministic solver. This is a
  *teacher-distilled* trace upgrade on top of the deterministic baseline.
- **Boxed instruction** (cell 16) — mirrors the eval metric exactly:

  ```python
  BOXED_INSTRUCTION = (
      "\nPlease put your final answer inside `\\boxed{}`. "
      "For example: `\\boxed{your answer}`"
  )
  ```

- **Packing**: `packing=False` — disabled. (Compare johnnyhyland who uses `packing=True`.)
- **Response-only loss**: not used. The whole sequence is supervised. *However* there is
  a custom `BOXED_LOSS_WEIGHT = 5.0` that upweights every token from the last `\boxed{`
  onward by 5×. See §1.4.

### 1.4 Training

- **Optimiser**: `optim="adamw_8bit"` (bitsandbytes paged 8-bit AdamW).
- **Scheduler**: `lr_scheduler_type="cosine"`, `WARMUP_RATIO=0.03`.
- **LR**: `1e-4`.
- **Batch**: `per_device_train_batch_size=2`, `gradient_accumulation_steps=1`
  (so effective batch size 2 — very small, the author is trading throughput for fresh
  gradients on a single high-VRAM GPU).
- **Epochs**: `NUM_EPOCHS=2`, `MAX_STEPS=-1` (full epochs).
- **Precision**: bf16 when available, else fp16.
- **Gradient checkpointing**: `gradient_checkpointing=False` in `TrainingArguments`,
  because Unsloth already enabled it inside `get_peft_model(use_gradient_checkpointing="unsloth")`.
  Enabling it twice would double-recompute.
- **Eval / save**: `eval_strategy="steps"`, `eval_steps=300`, `save_strategy="no"`
  (i.e. the author runs eval during training for visibility but does not checkpoint —
  only the final adapter is saved).
- **Custom weighted loss** (cell 47): if `BOXED_LOSS_WEIGHT > 1`, the trainer is given a
  `compute_loss_func` that finds the last `\boxed{` token position in `labels` and
  multiplies the cross-entropy weight of every subsequent token by 5×. This is a clever
  way to bias the gradient toward the *answer* tokens without changing the dataset.
  Watch-outs: the implementation does a Python loop per sample per step
  (`for bi in range(batch_size): for i in range(seq_len-marker_len+1): ...`) — it will be
  slow on long sequences and is O(batch × seq_len × marker_len). Worth porting to a
  vectorised version if you adopt it.

### 1.5 Tricks unique to (or fully leveraged by) Unsloth

- `use_gradient_checkpointing="unsloth"` — replaces HF's default checkpointing with
  Unsloth's lower-overhead path.
- `attn_implementation="eager"` plus `unsloth_force_compile=False` — keeps Unsloth from
  trying to JIT a kernel that does not support the hybrid Mamba+attention layout.
- `embed_tokens` and `lm_head` are **trainable** LoRA targets. Unsloth specifically
  supports this without exploding VRAM via its tied-embedding optimisations.
- Save method: plain `model.save_pretrained(ADAPTER_DIR)` (LoRA delta only) — *not*
  `save_method="merged_16bit"`. The competition wants the LoRA adapter, so merging
  would defeat the point. Followed by a tokenizer save and an explicit assertion that
  `adapter_config.json` exists.
- Inference path: vLLM (not Unsloth's `FastLanguageModel.for_inference`) with
  `enable_lora=True`, `max_lora_rank=32`, and a robust 3-attempt fallback ladder on
  `gpu_memory_utilization` / `max_num_seqs` / `enable_chunked_prefill` in case the first
  init OOMs. The vLLM call mirrors the competition environment exactly
  (`enable_prefix_caching=True`, `enforce_eager=False`).

### 1.6 Reported LB / training loss

None recorded in the committed notebook — all output cells were cleared before publish.
The notebook contains a local weighted-CV computation (cell 61) that mirrors the
competition's per-category weighting:

```python
weights = {
    "bit_manipulation": 1/6, "text_decryption": 1/6, "unit_conversion": 1/6,
    "numeral_system":   1/6, "gravity_physics": 1/6,
    "symbol_transform": 1/12, "numeric_equation": 1/12,
}
```

This is the implicit metric the author optimises against. The fact that two categories
(`symbol_transform`, `numeric_equation`) carry half the weight of the others matches the
sampling decision in §1.3 — the author is implicitly aware those are worth less per
correct answer.

### 1.7 Why is this notebook 187 KB?

Three reasons, in order of magnitude:

1. **Seven big trace-generator cells (cells 18, 20, 22, 24, 26, 28, 30)** total ~36 KB of
   raw Python source. The `text_decryption` and `bit_manipulation` generators are each
   ~9 KB. Cell 26 is an accidental duplicate of cell 24 (both contain the *same*
   `generate_text_decryption_trace` function — probably a copy-paste left in by mistake).
2. **The vLLM validation cell (cell 61)** is ~9 KB and contains the 3-attempt fallback
   ladder, prompt re-tokenisation, empty-output retry, per-category accuracy, and
   weighted-CV computation.
3. **JSON escape overhead.** Total raw source across all cells is only ~81 KB; the rest
   of the 187 KB is `.ipynb` JSON overhead (escaped backslashes for `\\boxed`, large
   multi-element `source` arrays etc.). All `outputs` arrays are empty.

There is **no** training-loss plot, no dataset preview screenshot, and no model card —
this is a pure pipeline notebook, not a writeup.

---

## Notebook 2 — `johnnyhyland / nvidia-nemotron-sft-grpo-colab-faster` (SFT → GRPO)

- **Path**: `notebooks/johnnyhyland__nvidia-nemotron-sft-grpo-colab-faster/nvidia-nemotron-sft-grpo-colab-faster.ipynb`
- **Author**: `johnnyhyland`
- **Strategy**: SFT warm-up on deterministic-solver CoT, then **GRPO** RL refinement on a
  smaller subset with custom reward functions. Both phases use the same rank-32 LoRA
  adapter.
- **Why "Colab faster"**: the author moved off Kaggle to Colab "Much easier there to
  control the kernel". This is a process speed win, not a training speed win. There is no
  Unsloth in this notebook — it uses **vanilla `transformers` + TRL** because TRL's
  `GRPOTrainer` is the framework of record for GRPO at the time of writing.
- **Reported LB / votes**: also not committed. The header cell says only "time will tell
  results from GRPO" — so as of upload this was an *experimental* pipeline whose author
  did not yet have a verified score.

### 2.1 GRPO setup

- **Framework**: `trl.GRPOTrainer` + `trl.GRPOConfig`. The trainer is constructed in
  `make_grpo_trainer(...)` (cell 8). No Unsloth GRPO, no custom RL loop.
- **Reward signals (summed by GRPOTrainer)**: three functions passed as a list:
  `[cosine_reward, format_reward, length_reward]`. Verbatim source in §2.3.
- **Verifier**: pure-Python answer comparison via `_normalize_answer` (coerces "42.00"
  ↔ "42") and `_extract_boxed` (regex pull of `\boxed{...}`, with a fallback for an
  unescaped `boxed{...}` shape). No reward model and no LLM-as-judge.

### 2.2 Pipeline: SFT first, then GRPO

```
Full SFT warm-up on 600 samples
        │
        ▼
LoRA adapter (rank 32 on sensitive layers + shared experts)
        │
        ▼  (same model object — no save/reload between phases)
GRPO refinement on 220 samples × 4 generations/prompt
        │
        ▼
submission.zip (intermediate snapshots also saved every 50 steps)
```

Rollouts come from `model.generate` inside `GRPOTrainer`, with `temperature=0.7` and
`max_completion_length=2048`. The model is wrapped once with PEFT — when the GRPO cell
runs after SFT, it detects `isinstance(model, PeftModel)` and **reuses the same
adapter** rather than double-wrapping. This is essential; double-wrapping creates a
second untrained LoRA on top of the SFT LoRA and silently destroys the warm-start.

Note: this notebook is **prompt-only** for GRPO (the `ground_truth` is passed as a side
column, not as a target sequence). Each prompt produces `num_generations=4` completions,
and the cosine/format/length rewards are applied to the model's *own* output.

### 2.3 Reward function code (verbatim)

The most important and reusable artefact in either notebook. Copy-pasted as-is from
cell 7:

```python
_reward_debug_counter = {"calls": 0}

def _normalize_answer(s):
    """Normalize answer for comparison (42.00 == 42, whitespace, etc.)."""
    s = s.strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s

def _extract_boxed(content):
    """Extract answer from \\boxed{...}, handling multiline and edge cases."""
    match = re.search(r'\\boxed\{([^}]*)\}', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'boxed\{([^}]*)\}', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _get_content(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion

def cosine_reward(completions, ground_truth, **kwargs):
    """Cosine-scaled accuracy reward (from Light-R1).

    Correct answers: reward decays from 1.0 → 0.1 as length approaches max.
    Incorrect answers: reward decays from -0.1 → -1.0 as length approaches max.
    No \boxed{}: 0.0.

    Key insight: even when all completions are correct, different lengths produce
    different rewards → non-zero advantage → actual gradient signal.
    """
    max_len = GRPO_MAX_COMPLETION
    rewards = []
    _reward_debug_counter["calls"] += 1
    show_debug = _reward_debug_counter["calls"] <= 2

    for i, (completion, gt) in enumerate(zip(completions, ground_truth)):
        content = _get_content(completion)
        extracted = _extract_boxed(content)
        clen = len(content)
        # Cosine scaling factor: 1.0 at length 0, 0.0 at max_len
        progress = min(clen / max(max_len, 1), 1.0)
        cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0

        if extracted is not None and _normalize_answer(extracted) == _normalize_answer(gt):
            # Correct: [1.0, 0.1] — short correct = high reward
            reward = 0.1 + 0.9 * cos_scale
        elif extracted is not None:
            # Wrong but has \boxed{}: [-0.1, -1.0] — long wrong = worse
            reward = -0.1 - 0.9 * (1.0 - cos_scale)
        else:
            # No \boxed{} — penalize proportional to length (shorter fail = less bad)
            reward = -0.5 * progress  # 0.0 at start, -0.5 at max_len

        rewards.append(reward)

        if show_debug and i < 2:
            tail = content[-120:] if len(content) > 120 else content
            print(f"  [COSINE REWARD batch={_reward_debug_counter['calls']}] "
                  f"extracted={extracted!r}, gt={gt.strip()!r}, reward={reward:.3f}, "
                  f"len={clen}, tail={tail!r}", flush=True)

    return rewards

def format_reward(completions, **kwargs):
    """Binary format reward: 1.0 if \boxed{} present, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        content = _get_content(completion)
        rewards.append(1.0 if _extract_boxed(content) is not None else 0.0)
    return rewards

def length_reward(completions, **kwargs):
    """Length penalty: encourages concise reasoning (arxiv:2501.12599).

    Returns 0.0 for empty, scales to -1.0 at max_completion_length.
    """
    max_len = GRPO_MAX_COMPLETION
    rewards = []
    for completion in completions:
        content = _get_content(completion)
        progress = min(len(content) / max(max_len, 1), 1.0)
        rewards.append(-progress)  # 0.0 to -1.0
    return rewards
```

The author's own sanity check (also from cell 7):

```python
_test_cos = cosine_reward(
    completions=["The answer is \\boxed{42}.",
                 "No boxed answer here.",
                 "Wrong: \\boxed{99}." + " " * 800],
    ground_truth=["42", "42", "42"],
)
# expect ~[1.0, 0.0, ~-0.7]
```

**Key insight from the author**: the reason for using *cosine scaling* (not a flat
{+1/-1}) is the **zero-advantage problem**. If all four rollouts for a prompt are
correct, a flat reward would give them all the same reward → zero advantage → zero
gradient. By scaling reward against completion length, even all-correct rollouts get
varying rewards (shorter is better), so the policy still learns. This is borrowed
from the Light-R1 paper / discussion and arxiv:2501.12599.

Summed reward range, per completion: `cosine ∈ [-1.0, +1.0]`, `format ∈ {0, 1}`,
`length ∈ [-1.0, 0.0]`. So combined reward ≈ `[-2, +2]` (correct short with `\boxed{}` ≈
2.0; long wrong without `\boxed{}` ≈ -1.5).

### 2.4 Generation config during RL

From cell 4 + cell 8:

```python
GRPO_NUM_GENERATIONS = 4      # completions per prompt
GRPO_MAX_COMPLETION = 2048    # eval allows 7680 / 3584; 1024 had truncation
GRPO_LR = 5e-6                # 20× lower than SFT LR (warm-started model)
GRPO_EPOCHS = 1
GRPO_TEMPERATURE = 0.7        # exploration; eval is greedy

args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=GRPO_NUM_GENERATIONS,
    generation_batch_size=GRPO_NUM_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=GRPO_EPOCHS,
    learning_rate=GRPO_LR,
    temperature=GRPO_TEMPERATURE,
    beta=0.0,                 # no KL penalty → no reference model loaded
    loss_type="grpo",
    mask_truncated_completions=False,
    logging_steps=2,
    max_grad_norm=0.1,        # aggressive clipping, stabilises small-dataset GRPO
    weight_decay=0.1,
    optim="adamw_8bit",       # ~50% less optimiser memory vs adamw_torch
    lr_scheduler_type="cosine",
    warmup_steps=5,
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    remove_unused_columns=False,   # keep ground_truth column for reward func
)
```

The single most consequential setting is `beta=0.0`. With β=0 there is no KL term
against a reference policy, which means **TRL does not load a second copy of the
30B model** as the frozen reference. On Colab / Kaggle hardware that is the
difference between "fits" and "OOM". The cost is that the policy can drift
arbitrarily far from the SFT distribution; the author leans on
(a) `max_grad_norm=0.1`, (b) `weight_decay=0.1`, and (c) only 1 epoch over 220 prompts
to keep drift bounded.

Other notes:
- `tokenizer.padding_side = "left"` is set inside `make_grpo_trainer` — required for
  causal-LM generation in batched rollouts. Forgetting this is a common GRPO bug.
- `generation_batch_size == num_generations` means each optimisation step uses one prompt
  worth of generations. Coupled with `gradient_accumulation_steps=1` you get exactly one
  optimiser step per prompt — 220 steps over the GRPO subset.
- `mask_truncated_completions=False`: truncated rollouts still contribute to the loss.
  Combined with the length-penalising reward, this nudges the model toward
  `max_completion_length`-friendly trajectories without throwing them out.
- `remove_unused_columns=False` is mandatory because the reward functions need the
  `ground_truth` column from the dataset.

### 2.5 Compute claims — what makes it "Colab faster"?

The "faster" claim is **operational**, not algorithmic. Reading the header cell
("Much easier there to control the kernel") it is clear the author moved off Kaggle
because Kaggle's session limits and OOM-recovery behaviour were the bottleneck. Things
that *do* help compute:

- `beta=0.0` (no reference model) — biggest single win, ~50% less VRAM.
- `optim="adamw_8bit"` — paged 8-bit optimiser, ~50% less optimiser VRAM.
- `gradient_checkpointing=True` with `use_reentrant=True` — standard.
- LoRA on **only 12 sensitive layers + shared experts** (see §2.7) — fewer trainable
  parameters than llkh0a's broad target list.
- `mamba_ssm` and `causal_conv1d` imports — needed for the Nemotron hybrid arch.
- `offload_folder="/tmp/offload"` on `from_pretrained` — allows CPU offload if VRAM is
  tight (won't actually offload on a T4-class GPU, but acts as a safety valve).

GPU not stated explicitly. Colab A100-40 GB is the implied target.

### 2.6 SFT phase config

For completeness — the SFT warm-up (cell 8 / `make_trainer`):

```python
SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # GRAD_ACCUM
    num_train_epochs=1,              # NUM_EPOCHS — "fresh data beats repeated passes"
    learning_rate=5e-5,              # LR — 2× lower than llkh0a (1e-4)
    logging_steps=5,
    bf16=True,
    max_grad_norm=0.1,
    weight_decay=0.1,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=30,                 # WARMUP_STEPS
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=2048,                 # MAX_SEQ_LEN
    packing=True,                    # ↔ llkh0a packing=False
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)
```

Effective batch size = 2 × 4 = 8 (vs. llkh0a's 2). `packing=True` here is important —
with only 600 unique samples it compresses to ~462 sequences = ~58 optimiser steps,
which the author flagged as borderline and is why they use only 1 epoch.

### 2.7 LoRA target-module strategy

The notebook follows **NVIDIA's own quantisation-sensitivity analysis from §4.2 of the
Nemotron 3 Nano paper** (arXiv:2512.20848). Quoting the comments verbatim:

> - Only 12/52 layers are sensitive (6 attention + 6 pre-attention Mamba)
> - Remaining 40 Mamba layers tolerate weight perturbations (low LoRA ROI)
> - Shared experts always active → good LoRA targets on all layers
> - Router weights frozen (§3.2.5), routable experts excluded (sparse traffic)

Implementation: parse `model.config.hybrid_override_pattern` (a string where `*` denotes
attention, `M` denotes Mamba, `E` denotes MoE/FFN), find the attention layer indices,
then also include `index - 1` for the immediately-preceding Mamba layer. Build the
target-module list by filtering all `nn.Linear` layers down to those whose layer index
is in this "sensitive" set, plus all `shared_expert` modules (which are routed-always
and thus see every forward pass).

```python
BEST_GROUPS = ["attention", "mamba", "shared_expert"]
BEST_RANK = 32
target_modules = build_target_modules(BEST_GROUPS, sensitive_only=True)
lora_config = LoraConfig(
    r=BEST_RANK,
    lora_alpha=BEST_RANK,              # alpha == rank (scaling = 1.0)
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Compared to llkh0a's broad target list (§1.2):

| Aspect              | llkh0a (Unsloth SFT)                     | johnnyhyland (SFT+GRPO)               |
|---------------------|------------------------------------------|---------------------------------------|
| Rank                | 32                                       | 32                                    |
| Alpha               | 16 (scale 0.5)                           | 32 (scale 1.0)                        |
| Dropout             | 0.1                                      | 0.05                                  |
| Mamba projections   | All layers (`in_proj`, `out_proj`)       | Only 6 sensitive pre-attn layers      |
| Attention           | All layers                               | All 6 attention layers                |
| MLP                 | `gate_proj`, `up_proj`, `down_proj`      | Excluded (treated as routable expert) |
| Shared experts      | Excluded (not enumerated)                | All layers                            |
| Embedding / lm_head | **Trainable** LoRA                       | Excluded                              |
| Trainable params    | ~800 M (author's estimate)               | Much smaller (~50–150 M, est.)        |

### 2.8 Failure modes the author mentions

Scattered through the comments — these are valuable because they are the author's
*post-hoc fixes* from previous broken runs:

1. **Zero-advantage problem** — flat reward + all-correct rollouts → no gradient. Fixed
   with cosine length scaling. (See §2.3.)
2. **Truncated completions at `max_completion_length=1024`** — model's CoT spilled over,
   so the `\boxed{}` answer was never emitted and *every* rollout got format reward 0.
   Fixed by bumping to 2048.
3. **Double-LoRA bug** — applying a fresh `get_peft_model` after the SFT phase had
   already wrapped the model would silently create an outer LoRA on top of an inner one.
   Fixed with `isinstance(model, PeftModel)` check before re-wrapping.
4. **Reference-model OOM** — TRL by default loads a frozen reference policy for KL.
   Setting `beta=0.0` skips this load.
5. **Padding-side bug** — `tokenizer.padding_side` defaults to `"right"` (correct for
   training) but rollouts need `"left"` (so generated tokens append correctly). Set
   inside `make_grpo_trainer`.
6. **`remove_unused_columns=True`** (the default) silently drops the `ground_truth`
   column before it reaches the reward function. Must be `False`.
7. **Reward-hacking** is not explicitly called out in the notebook, but the cosine
   reward's "no `\boxed{}` → small negative scaled by length" branch is itself a hedge
   against the model learning to *not* emit `\boxed{}` to dodge the wrong-answer
   penalty.

The header cell openly says: *"time will tell results from GRPO"* — i.e. the author has
not verified the LB lift from the RL phase at upload time.

### 2.9 SFT data formatting

Worth highlighting because it mirrors llkh0a's strategy but is less hand-tuned per
category:

- 600-sample random subsample of `train.csv` (seed 42).
- Per-puzzle deterministic solvers (`solve_gravity`, `solve_unit_conversion`,
  `solve_base_conversion`, `solve_text_encryption`, `solve_bit_manipulation`,
  `solve_equation_transformation`) emit both a predicted answer and a CoT trace.
- Fallback CoT templates (one per puzzle type) used when the solver fails.
- Same `BOXED_INSTRUCTION` suffix as llkh0a, applied to user content.
- Chat template applied with `enable_thinking=True`.
- `solver_stats` tracks solved-vs-fallback rate per dataset.

The GRPO subset (220 prompts, seed 99) is sampled *from the same 600 SFT samples*. This
is deliberate — the model has already seen the gold CoT during SFT, so GRPO is purely
about reinforcing the reasoning *process* rather than introducing new puzzles.

---

## Synthesis

### (a) Is Unsloth a clear win for this competition's compute envelope?

**Conditionally yes, but the win is modest.** Unsloth's biggest documented wins
(2× faster training, 50% less VRAM via Triton kernels) require the kernels to actually
fire. With `attn_implementation="eager"` and `unsloth_force_compile=False` —
both required for the hybrid Mamba+attention Nemotron arch — you are giving up most of
Unsloth's speedup and keeping only the gradient-checkpointing improvements and the
`get_peft_model` convenience wrapper.

That said:
- llkh0a's setup *does* let them put LoRA on `embed_tokens` + `lm_head` without OOM,
  which is hard in vanilla `transformers` at this model scale. That's a real, exclusive
  Unsloth capability.
- Unsloth's grad-checkpointing is ~10–15% more memory-efficient than HF's default, which
  matters on a 30B model.
- Unsloth's `FastLanguageModel.from_pretrained` is friendlier than `AutoModelForCausalLM`
  for getting a working LoRA pipeline on the first try.

If you are on a 24 GB or smaller GPU, Unsloth is probably necessary to fit rank-32 LoRA
on 30B at all. If you have ≥48 GB (RTX 6000 Blackwell, A100-80, H100), the marginal
benefit shrinks and you may prefer vanilla `transformers` + TRL for compatibility with
GRPO (TRL's GRPOTrainer is the path of least resistance, and is what johnnyhyland uses).

**Combined Unsloth + GRPO is not demonstrated by either notebook.** That's an open
question — Unsloth ships its own `unsloth.RL` GRPO support but neither notebook here
uses it.

### (b) What does GRPO add over SFT for this task?

In principle: SFT teaches the model to *imitate* the synthetic CoT traces, while GRPO
lets the model *explore* different reasoning paths and reinforces ones that land on the
right answer. This matters because the deterministic solvers in §1.3 / §2.9 produce a
single, formulaic trace per puzzle. SFT-only training risks the model memorising the
trace template and over-fitting to its surface form, which is brittle when the test set
contains puzzles with different surface structure (e.g. the held-out evaluation set
likely shuffles parameter names, operator symbols, etc.).

GRPO with a verifier reward can theoretically:
1. Free the model to use shorter / different reasoning when it works.
2. Encourage robust answer extraction (the `format_reward` directly pushes for
   `\boxed{}`).
3. Discover "shortcut" reasoning patterns the deterministic solvers don't know.

In *practice* on this competition, with johnnyhyland's exact setup, there is no
recorded evidence the GRPO phase improves LB. The author themselves say "time will
tell". Key constraints:
- 220 prompts × 4 generations × 1 epoch = ~220 optimiser steps, which is a tiny RL run.
  GRPO papers typically use 10× this volume.
- The verifier is exact-match against the gold answer. For puzzle types where the model
  can guess the right number ~10% of the time, the reward signal is noisy.
- `beta=0.0` means no KL anchor → policy can drift away from the SFT-distilled trace
  style and start producing degenerate completions if the run is too long.

A pragmatic read: **GRPO is plausibly useful as a final polish (50–200 steps, β small
but non-zero) on top of a strong SFT adapter, not as a replacement for it.**

### (c) Are RL-trained adapters competitive with SFT-only 0.85-class adapters?

**Not in our sample.** Of the notebooks we've inspected so far:
- The highest-scoring approaches in this repo (the published 0.85-class adapters from
  earlier write-ups) are pure SFT with carefully curated CoT and the broad-target LoRA
  philosophy that llkh0a is iterating toward.
- The GRPO notebook here is explicitly an experiment by an author who has not yet
  verified the lift.

So: **as of these two notebooks, RL is experimental and there is no demonstrated public
LB win from GRPO over a well-tuned SFT recipe.** This matches the broader competitive
ML pattern — RL helps when the SFT data is the bottleneck, but here the bottleneck for
most teams is *trace quality* and *category coverage*, which are SFT-side problems
addressed by hand-rolled solvers and teacher distillation.

That said, if someone *did* get GRPO working at scale (e.g. 5 000 prompts × 8 generations
× 3 epochs with a small β), it would plausibly push past the SFT plateau because the
synthetic traces are inherently template-y and the test set rewards genuine reasoning.

### (d) Practical gotchas for either approach

Reusable across any future attempt at this competition:

**SFT (Unsloth or vanilla):**
- Mirror the `BOXED_INSTRUCTION` suffix *exactly* in training prompts. Both notebooks
  do this — if you omit it the model will never learn to expect that instruction at
  inference.
- Use `apply_chat_template(enable_thinking=True)` to get `<think>...</think>` regions
  to be supervised consistently. Nemotron-3-Nano is trained with this scaffold.
- If warm-starting from a previous adapter, do **not** use
  `PeftModel.from_pretrained` — it returns a frozen adapter. Instead create a fresh
  trainable adapter then call `set_peft_model_state_dict`. (llkh0a §1.2.)
- Targeting `embed_tokens` + `lm_head` with LoRA is feasible under Unsloth at this
  model scale and gives the LoRA more capacity to shift output bias. Skip it under
  vanilla `transformers` unless you have a lot of VRAM.
- Consider per-category sampling caps if some puzzle types dominate the dataset — both
  notebooks downsample the easy categories.
- Boxed-token loss upweighting (llkh0a's `BOXED_LOSS_WEIGHT=5.0`) is a cheap way to
  bias gradients toward the answer; the implementation is O(B × T × marker_len) — port
  to vectorised tensor ops if you adopt it.
- Train at `max_seq_length` smaller than eval's `max_model_len` to save memory; both
  notebooks do this (3500 / 2048 vs 8192).

**GRPO (TRL):**
- Set `beta=0.0` to avoid loading a 30B reference policy. Compensate with
  `max_grad_norm=0.1` + `weight_decay=0.1` + a tiny LR + a small number of epochs.
- Set `tokenizer.padding_side = "left"` before constructing the trainer.
- Set `remove_unused_columns=False` so reward functions can see `ground_truth`.
- `max_completion_length` must comfortably fit the longest expected trace + `\boxed{}`.
  1024 was too short for johnnyhyland's recipe; 2048 worked. For full competition
  alignment you may need 3584+ since eval allows 7680.
- Use length-scaled rewards (`cosine_reward` above) to avoid zero-advantage when all
  rollouts agree. Flat {+1, -1} rewards waste compute on most steps.
- Sum reward signals carefully: if your `format_reward` is binary {0, 1} and your
  `cosine_reward` is in [-1, 1] and your `length_reward` is in [-1, 0], the relative
  magnitudes determine which signal dominates. Tune by scaling the weaker signal up
  rather than tuning the dominant one down — you want the format signal to dominate
  early then fade.
- After SFT + GRPO, the saved adapter is the *combined* adapter (single
  `model.save_pretrained`). It is *not* two adapters stacked. You submit one zip.

**Both:**
- vLLM is the inference engine that matches the eval environment. Test locally with
  the same `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`,
  `enable_prefix_caching=True` settings as the competition metric notebook to catch
  prompt-format mismatches early. llkh0a's 3-attempt fallback ladder for vLLM init is
  worth copying — vLLM is OOM-sensitive on cold start.
- `save_method="merged_16bit"` is **wrong** for this competition — you must submit the
  LoRA adapter, not a merged model. Use plain `model.save_pretrained` and verify
  `adapter_config.json` exists in the zip root.
- Both authors run validation against the *competition-weighted* metric (1/6 for the
  five "main" categories, 1/12 for `symbol_transform` and `numeric_equation`) — not
  raw accuracy. Adopt the same weighting locally or you will optimise for the wrong
  thing.

