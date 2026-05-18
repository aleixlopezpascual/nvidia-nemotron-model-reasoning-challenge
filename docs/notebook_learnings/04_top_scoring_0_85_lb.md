# Top-scoring notebooks at LB 0.85 — deep dive

The three notebooks reviewed here are the public-leaderboard heavyweights of the
NVIDIA Nemotron Model Reasoning Challenge, each claiming a **public LB of ~0.85**
(versus the current LB leader at 0.87 and the basic-SFT crowd hovering around
0.70–0.78). They share more than they diverge — all three are LoRA fine-tunes of
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with **r=32, alpha=32, dropout=0.0**
trained with **Unsloth's FastLanguageModel** on a Kaggle GPU — but the *delta*
that takes a baseline to 0.85 is **(a) chain-of-thought-augmented training data
distilled from a stronger teacher** (Tong Hui Kang's `nemotron-master` corpus),
**(b) LoRA on the Mamba `in_proj`/`out_proj` projections as well as the
attention QKV/O and MLP gate/up/down**, **(c) `lm_head` included in
target_modules (or hand-patched in)**, and **(d) a long context
(MAX_SEQ_LEN=8192) so the entire CoT trace fits in the loss window**. dgxchen
and huikang share the *same* training corpus (huikang's published
`nemotron-master` snapshot); huikang is the original author of the recipe and
dgxchen reproduces it from konbu17's training scaffold, while mayukh18 is a
slightly earlier baseline that does **not** train on CoT and instead trains on
raw `\boxed{answer}` targets — which is why mayukh18's notebook, despite being
labeled "0.85" in the dataset folder, is closer in spirit to a polished
baseline than to the true 0.85 recipe and likely lands lower in practice
(see Synthesis).

## Comparison table

| Notebook | Framework | LoRA r/α/dropout | target_modules (count) | LR | Scheduler | Effective batch | Steps / epochs | Max seq len | Train data | Claimed LB |
|---|---|---|---|---|---|---|---|---|---|---|
| **dgxchen** `training-with-unsloth-to-achieve-0-85-lb` | Unsloth + TRL `SFTTrainer` (custom stratified sampler) | 32 / 32 / 0.0 | 8: `q,k,v,o,in_proj,out_proj,up,down` | **2e-4** | **linear** decay, **0 warmup** | 1×32 = **32** | **1 epoch** on ~all CoT rows | **8192** | huikang's `problem_ids_matched.csv` (CoT distilled by huikang's `nemotron-master`) | **0.85** |
| **huikang** `end-to-end-finetuning-for-lb-0-85` | Unsloth + **custom training loop** (no `SFTTrainer`) + Cut-Cross-Entropy + Modal/Kaggle dual runner | 32 / 32 / 0.0 | 9: `q,k,v,o,in_proj,out_proj,up,down,lm_head` | **2e-4** | **linear decay** `lr*(1 - step/num_steps)` | 32 / 4 micro = **32** | **1000 steps** (clamped to `len(examples)//32`) | **8192** | huikang's `nemotron-master/training/sft/04-08-16-14/tokens` (pre-tokenized CoT corpus with weight masks) | **0.85** |
| **mayukh18** `unsloth-sft-full-data-training` | Unsloth + TRL `SFTTrainer` + `train_on_responses_only` | 32 / 32 / 0.0 | **11**: `q,k,v,o,in_proj,out_proj,up,down,gate_proj,embed_tokens,lm_head` | **1e-4** | **cosine**, warmup_ratio 0.03 | 6×2 = **12** | **1 epoch** on full 9.4k train | **1024** | Raw competition `train.csv` (9500 rows), **no CoT** — assistant = `<think>\nLet me work through this step by step.\n</think>\n\\boxed{answer}` | **0.85 claimed** (but no CoT — see Synthesis caveats) |

**Convergent core** (in all three): Unsloth `FastLanguageModel.from_pretrained` with
`load_in_4bit=False, load_in_8bit=False, full_finetuning=False, dtype=torch.bfloat16,
attn_implementation="eager", trust_remote_code=True`; LoRA rank=32, alpha=32, dropout=0.0,
bias="none", `use_gradient_checkpointing="unsloth"`, `random_state=42`.

**Critical divergences**: CoT data source, target-module list (especially `lm_head` /
`embed_tokens` / `gate_proj`), batch size (32 vs 12), and learning rate (2e-4 vs 1e-4).

---

## 1. dgxchen — `training-with-unsloth-to-achieve-0-85-lb`

### Headline approach (one sentence)
Reproduce Tong Hui Kang's open-source `nemotron-master` recipe inside Kaggle using
**Unsloth + TRL `SFTTrainer`** on a CSV of **distilled chain-of-thought traces**,
training **LoRA r=32 on 8 projection modules** (no `lm_head`) for **1 epoch**
with **stratified batching by problem type** and effective batch size 32.

### Acknowledgements & lineage
The notebook openly credits two sources:
- **Tong Hui Kang's GitHub repo `tonghuikang/nemotron`** — provides the CoT dataset
  and most hyperparameters.
- **konbu17's public Kaggle notebook `nemotron-sft-lora-with-cot`** — provides the
  training scaffolding that dgxchen modifies.

dgxchen's explicit ablation in the opening markdown:
> "In this version, training on **`lm_head`** in target_modules has been **removed**.
> At the same time, the microbatch size has been reset to **1**. Increasing the
> microbatch size to 2 reduced training time by about 3 hours but caused the LB score
> to drop by around 0.1." (sic — likely means 0.01, but he writes 0.1)

> "Setting `per_device_train_batch_size=2` and `gradient_accumulation_steps=16` can
> reduce the training time to around **4 hours** while maintaining similar
> performance (**0.84–0.85**)."

So the author has direct empirical evidence that **micro-batch size 1** with
**grad-accum 32** is materially better than micro-batch 2 with grad-accum 16,
even though the effective batch size is the same.

### Data
- **Source**: pre-distilled CoT dataset hosted as a Kaggle dataset
  `/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv`.
  This CSV is huikang's `nemotron-master` distilled corpus re-saved as a single
  CSV with columns `prompt, answer, generated_cot, type`.
- **Size**: "Full dataset: {len(df)} rows" — the notebook reads them all,
  filters out empty/short CoT entries (`len(cot.strip()) < 5`), and trains on
  every survivor. Effective dataset is ≈ the same scale as huikang's tokenized
  corpus (a few thousand CoT-augmented rows).
- **Filtering**: drops rows where `cot` is `"nan"` or under 5 characters; also
  strips any pre-existing `\boxed{...}` from the CoT before re-appending the
  canonical boxed answer.
- **CoT format** (verbatim from cell-11):
  ```python
  PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
  ...
  cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
  user_content = prompt + PROMPT_SUFFIX
  assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
  records.append({"messages": [
      {"role": "user", "content": user_content},
      {"role": "assistant", "content": assistant_content},
  ]})
  ```
  The assistant turn is the *cleaned* CoT followed by `</think>\n\boxed{answer}`.
  No explicit `<think>` is added — the chat template (`enable_thinking=True`)
  injects the opening `<think>` tag automatically; the trainer adds the closing
  `</think>` and the boxed answer.
- **Stratification**: a custom `StratifiedSFTTrainer` builds an index order
  that round-robins across `type` so each effective batch sees a mix of puzzle
  types. This is a "key ingredient" lifted directly from huikang's
  `nemotron-master`.

### Base model & quantization
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
  (downloaded via `kagglehub.model_download`). For submission, the
  `base_model_name_or_path` in `adapter_config.json` is rewritten to
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- **No quantization**: `load_in_4bit=False, load_in_8bit=False,
  full_finetuning=False, dtype=torch.bfloat16`. Pure bf16.
- `attn_implementation="eager"` (the only impl that works with Nemotron-H
  Mamba/SSM blocks).
- `unsloth_force_compile=False`.
- Tokenizer fix: `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token`.

### LoRA config (verbatim)
```python
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "up_proj", "down_proj",
]
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```
No `use_rslora`, no `use_dora`, no `modules_to_save`. `lm_head` deliberately
**absent** (this is the ablation the author calls out).

### Training framework
**Unsloth `FastLanguageModel`** + **TRL `SFTTrainer`** (subclassed as
`StratifiedSFTTrainer` to override `get_train_dataloader` with a precomputed
stratified index order). No packing, no NEFTune.

### Hyperparameters (verbatim `SFTConfig`)
```python
training_args = SFTConfig(
    output_dir="/kaggle/working/sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=0,
    max_length=8192,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    max_grad_norm=1e9,            # effectively disabled
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=SEED,
    report_to="none",
    packing=False,
)
```
Notable: **no warmup**, **linear decay**, **AdamW (default trl) with β2=0.95**,
**`max_grad_norm=1e9`** (gradient clipping effectively off), and
**`weight_decay=0.0`**. Optimizer is the trl default AdamW (not 8-bit).

### Hardware & runtime
Kaggle GPU (Blackwell-class — uses `ptxas-blackwell` from ryanholbrook's utility
notebook; sm_120 hint elsewhere). With micro-batch=1, grad-accum=32, the author
reports training takes about **6–7 hours** end-to-end (cf. "reduce to ~4 hours
with batch=2 and grad-accum=16 but lose 0.01 LB").

### Inference / submission
The submission script repackages the adapter and **rewrites the adapter config**:
```python
cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0
```
Only `adapter_config.json` and `adapter_model.safetensors` go into
`submission.zip` (no tokenizer files). The inference prompt is implicit in the
training format: the eval pipeline appends the same `PROMPT_SUFFIX` and the
chat template's `enable_thinking=True` opens a `<think>` block; the model is
trained to close it with `</think>` and box the answer.

### Reported scores / ablations
- **LB ≈ 0.85** (notebook title).
- Removing `lm_head` from target_modules: author says this is what *this*
  version does and it still hits 0.85 — implying `lm_head` LoRA is not strictly
  required if the rest of the recipe (data + stratified batching + 8-module
  LoRA) is in place. huikang disagrees (see below).
- Micro-batch=1 vs 2: drops ≈ 0.01 LB (the author writes 0.1 but the context
  strongly implies 0.01).

### Tinker references
Not directly. The training corpus (`nemotron-cot-tong`) comes from huikang's
`nemotron-master`, which *was* originally produced via Tinker (huikang's
`tinker-submission-notebook` is the upstream artifact).

---

## 2. huikang — `end-to-end-finetuning-for-lb-0-85`

### Headline approach (one sentence)
The reference 0.85 implementation: a fully hand-rolled training loop (no
`SFTTrainer`) that consumes a pre-tokenized CoT corpus with per-token loss
weights, applies **LoRA r=32 to 9 modules including `lm_head`**, uses
**Cut-Cross-Entropy** to avoid materializing logits, **ties MoE expert LoRA
weights Tinker-style**, and trains for **1000 steps at lr=2e-4** with linear
decay — runnable both on Kaggle GPU and on Modal RTX-PRO-6000.

### Data
- **Source**: huikang's own pre-tokenized corpus at
  `/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/tokens`.
  Each sample is `{problem_id, tokens, mask}` already tokenized against the
  Nemotron tokenizer; the mask is the **per-token loss weight** (1 on
  assistant/CoT tokens, 0 on prompt/system tokens) — this is the same
  "compute loss only on responses" idea that mayukh18 implements via
  `train_on_responses_only`, but done upstream and baked into the corpus.
- **CoT generation**: the corpus is generated by a stronger teacher (the
  upstream `nemotron-master` GitHub repo, which uses Tinker to distil; see
  huikang's `tinker-submission-notebook` for the lineage). The exact prompt
  template used at distillation time isn't shown in this notebook — it's the
  template encoded in the tokenized stream.
- **Size**: the loop prints `Loaded {len(examples)} examples, {total_tokens:,}
  tokens (unmasked={total_unmasked:,.0f})`. With BATCH_SIZE=32 and NUM_STEPS=1000,
  the upper bound on examples touched is 32,000 — but `max_steps =
  len(examples) // BATCH_SIZE` clamps to whatever the corpus actually contains.
  In practice the published corpus contains ~9.5k–32k segmented examples
  (multiple CoT traces per problem are generated).
- **Training order**: `TRAIN_ORDER_PATH = ".../logprobs/index.jsonl"` —
  huikang **replays the exact epoch-0 problem order** from the original Tinker
  run. This is unusual and likely matters: the order is not random, it's the
  order the teacher saw.
- **Filters**: skip if `not tokens`, truncate to `MAX_SEQ_LEN=8192`, skip if
  the loss mask is all zeros, optionally filter to `train.csv` ids
  (`ORIGINAL_PROBLEMS_ONLY=False` by default).

### Base model & quantization
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (Kaggle) or
  `unsloth/Nemotron-3-Nano-30B-A3B` (Modal).
- `load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
  dtype=torch.bfloat16, attn_implementation="eager",
  unsloth_force_compile=True` (note: forces compile, unlike dgxchen).
- **dtype regime is explicit and asserted**: LoRA params cast to **fp32**,
  base model in **bf16**, MoE router weight kept in **fp32** (per
  `NemotronHPreTrainedModel._keep_in_fp32_modules_strict =
  ["e_score_correction_bias"]`). The notebook contains a long inline comment
  citing the Transformers source for why the router must be fp32.
- **Mamba fast path is force-enabled**: `nemotron_mod.is_fast_path_available =
  True` — Unsloth disables it by default; huikang patches it back on.

### LoRA config (verbatim)
```python
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```
**`lm_head` is in `TARGET_MODULES` but Unsloth silently drops it for MoE
models**, so huikang manually adds it back:
```python
_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
model.base_model._create_and_replace(
    _cfg, "default", target=_lm_head, target_name="lm_head", parent=_causal_lm,
)
```
This is the single biggest difference vs. dgxchen.

### MoE tied weights (Tinker convention)
huikang ties one LoRA factor across all 128 MoE experts:
```python
# gate_up_proj / up_proj / w1 / gate_proj -> tie A (input/hidden side)
# down_proj / w2                          -> tie B (output/hidden side)
```
At init, all 128 expert slices are made identical (mean-and-broadcast). Before
each optimizer step, the gradients across the expert dimension are **summed**
(not averaged — the comment explains why: this is the correct chain-rule
gradient for shared weights). The saved adapter materializes 128 per-expert
copies for vLLM compatibility, but during training the effective parameter
count is the same as a single dense LoRA over the MoE blocks. **This is the
"Tinker-style" trick the notebook header advertises.** Toggled via
`MOE_TIE_WEIGHTS = True`.

### Cut-Cross-Entropy (CCE) patch
The CausalLM forward is replaced with a custom forward that computes
`linear_cross_entropy(hidden_states, lm_weight, labels, reduction="none")` —
**logits are never materialized**, which is a substantial VRAM win at
`MAX_SEQ_LEN=8192`. The `lm_weight` is reconstructed as
`base_w + scaling * lora_B @ lora_A` to include the LoRA delta on `lm_head`.

### Training framework
**No `SFTTrainer`** — a hand-written loop with manual micro-batch accumulation,
manual padding, manual learning-rate schedule, manual gradient norm and step.
Uses `torch.optim.AdamW` directly with `betas=(0.9, 0.95)`, `eps=1e-8`,
`weight_decay=0.0`.

### Hyperparameters
```python
MAX_SEQ_LEN     = 8192
NUM_STEPS       = 1000
BATCH_SIZE      = 32
MICRO_BATCH_SIZE = 4    # so 8 forward/backward passes per optimizer step
LEARNING_RATE   = 2e-4
```
- LR schedule: **manual linear decay**, `lr = LEARNING_RATE * (1 - step / num_steps)`.
- Grad clip: `max_norm=1e9` (effectively off, matches dgxchen).
- Optimizer: `torch.optim.AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)`.
- No warmup.
- Loss is **weighted CE** (per-token weights from the corpus mask), reduced as
  `sum(weight*ce) / sum(weight)`.
- `SHUFFLE_DATASET = False` — corpus order is preserved (the Tinker epoch-0
  order).

### Hardware
- **Kaggle path**: GPU runtime (Blackwell sm_120), uses the
  ryanholbrook/llkh0a Triton/wheel patches.
- **Modal path**: RTX-PRO-6000 (`gpu="RTX-PRO-6000"`), CUDA 12.8,
  `torch==2.10.0`, custom-built `mamba_ssm==2.3.1` and
  `causal_conv1d==1.6.1` wheels (built in the Modal image via
  `TORCH_CUDA_ARCH_LIST='12.0' pip wheel`), `cut-cross-entropy>=25.1.0`.
- Timeout: `6 * 60 * MINUTES` (6 hours) on Modal. Training is reported to
  finish well within this on RTX-PRO-6000.

### Inference / submission
The saved adapter has **`lm_head` keys renamed** before zipping:
```python
renamed = {
    k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
    for k, v in tensors.items()
}
```
This is because the Nemotron-H architecture nests `lm_head` under `backbone`
in the eval-time model graph; without this rename, vLLM at submission time
won't find the LoRA weights for `lm_head`.

Only `adapter_*` files are zipped. No tokenizer, no extra config.

### Reported scores / ablations exposed via flags
The notebook surfaces several toggles that map to ablations huikang has run:
- `RESET_WEIGHTS=True` — fresh LoRA init vs. continuing from a prior Tinker
  adapter (`tinker-submission-notebook/submission.zip`). The default is fresh
  init.
- `IN_PROJ_ONLY=False` — if True, freeze everything except `in_proj` (a
  cheap-update ablation; default off).
- `MOE_TIE_WEIGHTS=True` — the Tinker tying trick; default on.
- `ORIGINAL_PROBLEMS_ONLY=False` — restrict to the original 9500 `train.csv`
  problem_ids vs. using all problems in the synthetic corpus (default uses all).
- `SHUFFLE_DATASET=False` — keep teacher's training order.

### Tinker references — extensive
- The repo lineage (`nemotron-master`) is the artifact of a Tinker fine-tuning run.
- MoE LoRA weight tying is labelled "Tinker-style" / "Tinker convention".
- A separate `tinker-submission-notebook` exists in the same author's set that
  packages a Tinker-trained adapter for Kaggle submission. This end-to-end
  notebook can **continue from that Tinker adapter** (`RESET_WEIGHTS=False`),
  i.e. Tinker is the upstream teacher and this notebook is a Kaggle-only
  post-finetune.

---

## 3. mayukh18 — `unsloth-sft-full-data-training`

### Headline approach (one sentence)
A polished baseline-SFT notebook: Unsloth + TRL `SFTTrainer` with
`train_on_responses_only`, **LoRA r=32 across 11 modules** (the widest target
set of the three, including `embed_tokens`, `gate_proj`, and `lm_head`),
trained for **1 epoch on the raw 9500-row `train.csv` with no chain-of-thought**
— the assistant turn is a fixed-template stub `<think>\nLet me work through
this step by step.\n</think>\n\boxed{answer}`.

### Data
- **Source**: raw competition `train.csv` (9500 rows), no CoT distillation, no
  synthetic data, no rejection sampling. A 100-row held-out validation slice
  is split off via `train_test_split(train_df, test_size=100, random_state=42)`.
- **Size**: 9,400 train / 100 val.
- **CoT format**: there is **no real CoT** — every assistant response is the
  same boilerplate stub:
  ```python
  BOXED_INSTRUCTION = (
      "\nPlease put your final answer inside `\\boxed{}`. "
      "For example: `\\boxed{your answer}`"
  )
  def create_answer_str(answer):
      safe = str(answer).replace("}", "\\}")
      return f"<think>\nLet me work through this step by step.\n</think>\n\\boxed{{{safe}}}"

  def format_train_row(prompt, answer):
      return {
          "messages": [
              {"role": "user",      "content": prompt + BOXED_INSTRUCTION},
              {"role": "assistant", "content": create_answer_str(answer)},
          ]
      }
  ```
  This trains the model to **emit a near-empty `<think>` block and jump
  straight to the boxed answer**. It's the inverse of dgxchen/huikang's
  approach (which trains the model to *use* the think block).
- **Filtering**: none beyond the val split.

### Base model & quantization
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`.
- `load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
  trust_remote_code=True, attn_implementation="eager",
  torch_dtype=torch.bfloat16` (note the double `torch_dtype=`/`dtype=None`
  argument — a small Unsloth API quirk).
- `unsloth_force_compile=False`.
- No explicit tokenizer fix beyond Unsloth's defaults.

### LoRA config (verbatim)
```python
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0

target_modules = [
    'out_proj', 'v_proj', 'q_proj', 'down_proj', 'embed_tokens',
    'k_proj', 'in_proj', 'up_proj', 'o_proj', 'lm_head', 'gate_proj'
]

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```
**Eleven** target modules — adds `embed_tokens`, `gate_proj` and `lm_head` on
top of the dgxchen 8-module set. No `modules_to_save`, no rslora, no dora.

### Training framework
Unsloth + TRL `SFTTrainer` + `unsloth.chat_templates.train_on_responses_only`
to mask the prompt tokens so the loss is computed only on the assistant
response:
```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```
A `SampleGenCallback` runs `model.generate` on 3 val rows at each
`eval_strategy="steps"` event (every 50 steps), switching between
`FastLanguageModel.for_inference` and `.for_training`.

### Hyperparameters (verbatim `TrainingArguments`)
```python
TrainingArguments(
    output_dir="./nemotron-lora-checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,                # half of dgxchen/huikang's 2e-4
    lr_scheduler_type="cosine",        # cosine, not linear
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    optim="adamw_8bit",                # 8-bit AdamW from bitsandbytes
    seed=42,
    report_to="none",
    dataloader_num_workers=4,
    eval_strategy="steps",
    eval_steps=50,
)
max_seq_length=MAX_SEQ_LEN  # 1024
dataset_text_field="text"
```
Notable differences vs. the other two:
- `max_seq_length=1024` (not 8192) — because there's no long CoT, the prompts
  fit in 1024.
- `lr=1e-4` and **cosine schedule with 3% warmup** (vs. linear+no-warmup).
- `per_device_train_batch_size=6, gradient_accumulation_steps=2` →
  **effective batch = 12** (vs. 32 in the other two).
- `optim="adamw_8bit"` — uses bitsandbytes 8-bit AdamW (vs. fp AdamW in the
  others; this saves optimizer-state VRAM).

### Hardware
Kaggle GPU. Reports approximately **1 hour per epoch** thanks to short
sequences and 8-bit Adam.

### Inference / submission
```python
with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(ADAPTER_DIR):
        zf.write(
            os.path.join(ADAPTER_DIR, fname),
            arcname=os.path.join(ADAPTER_DIR, fname),
        )
```
Zips the **whole adapter directory under a `nemotron-lora-adapter/` prefix**
(unlike dgxchen/huikang which write files flat at the zip root). Both layouts
are accepted by the eval pipeline. No config rewrites, no key renames.

The notebook also defines a local-eval helper with a reusable
`extract_final_answer` function (regex on `\boxed{...}` + fallback patterns
for "The final answer is: ..." and trailing numeric/text lines), but the
inference loop is commented out — it never actually runs validation in the
notebook as published.

### Reported scores / ablations
- The notebook is filed under `unsloth-sft-full-data-training` and the folder
  contains a `nemotron-lora-submission` dataset, implying the trained adapter
  is reused across submissions.
- The notebook itself does **not** print a leaderboard score — the "0.85"
  claim is a folder-level/competition-context claim. The recipe here (no CoT,
  empty `<think>`, lr=1e-4 cosine, effective batch=12, 11-module LoRA) is
  qualitatively closer to a polished baseline than to the dgxchen/huikang
  CoT-based 0.85 runs. **Treat the 0.85 attribution to this notebook with
  caution** — it may represent a different submission that *uses* this
  adapter as a starting point rather than this exact training run.

### Tinker references
None — this notebook is independent of the Tinker / nemotron-master lineage.

---

## Synthesis — what the 0.85 recipe actually looks like

The three notebooks converge on a very specific recipe. The two that are clearly
genuine 0.85 (dgxchen, huikang) share a teacher and a corpus; mayukh18 is the
odd one out and is best treated as a strong baseline, not the 0.85 method.

1. **The 0.85 secret is the data, not the optimizer.** Both genuine 0.85
   notebooks train on huikang's `nemotron-master` corpus — chain-of-thought
   traces distilled by a stronger teacher (likely Nemotron-Ultra or a
   Tinker-trained variant), with `<think>...</think>\boxed{answer}` as the
   assistant turn. mayukh18 trains on raw `train.csv` with an *empty* `<think>`
   block, which is qualitatively a different model. If you want to reproduce
   0.85, you need this corpus (or a comparable one you distil yourself).

2. **The single common architectural choice that matters: LoRA on Mamba's
   `in_proj` and `out_proj`.** All three notebooks include these in
   `target_modules`. Nemotron-3-Nano-30B is a hybrid Mamba+Transformer; LoRA on
   only the attention projections leaves the Mamba blocks untouched. Adding
   `in_proj` and `out_proj` is the most non-obvious target-modules choice and
   is convergent across all three.

3. **`lm_head` is contentious but probably matters.** huikang fights hard to
   keep `lm_head` LoRA in (manually re-adds it because Unsloth drops it for
   MoE; renames keys at save time so vLLM finds it). dgxchen explicitly
   removes it and still claims 0.85. mayukh18 includes it. The safest bet:
   include `lm_head`; if Unsloth silently drops it, manually re-inject
   following huikang's `model.base_model._create_and_replace(...)` pattern,
   and rename `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`
   in the saved safetensors.

4. **LoRA r=32, alpha=32, dropout=0.0, bias="none" is unanimous.** Rank-32 is
   the competition cap; alpha=rank (i.e. scaling=1.0) is convergent;
   dropout=0.0 is required to enable Unsloth's fast-kernel patching on the
   LoRA matrices. Don't use `use_rslora`, `use_dora`, or `modules_to_save`.

5. **bf16 throughout, no quantization, `attn_implementation="eager"`,
   `unsloth_force_compile` toggled on or off, gradient checkpointing on
   ("unsloth" mode).** `load_in_4bit=False, load_in_8bit=False,
   full_finetuning=False, dtype=torch.bfloat16` is the load incantation in all
   three.

6. **The effective batch size is 32 with micro-batch=1.** dgxchen demonstrates
   empirically that micro-batch=2 with grad-accum=16 *loses* LB even though
   the effective batch is the same; huikang uses micro-batch=4, grad-accum=8
   (so batch=32) and is fine. The reason is probably tied to bf16 batched
   matmul numerics for MoE routers / Mamba kernels — micro-batch=1 keeps the
   per-token compute graph simpler and matches the distillation regime more
   closely. **If you must reduce wall-time, micro-batch=4 with grad-accum=8
   is the safe compromise; never go below an effective batch of 32.**

7. **lr=2e-4, linear decay, zero warmup, weight_decay=0, no grad clipping
   (max_grad_norm=1e9), AdamW with (β1=0.9, β2=0.95, eps=1e-8).** These exact
   values appear in both genuine-0.85 notebooks. mayukh18's lr=1e-4 cosine is
   gentler and consistent with not having CoT in the targets. **Do NOT use
   adamw_8bit for the genuine recipe** — dgxchen and huikang both use fp
   AdamW; only mayukh18's no-CoT baseline uses adamw_8bit.

8. **MAX_SEQ_LEN=8192 is required to fit CoT traces.** mayukh18's 1024 only
   works because his targets are stubs. With real CoT, plan for 8k. Truncate
   anything longer (huikang's loader does `tokens[:MAX_SEQ_LEN]`).

9. **Train for ~1 epoch / ~1000 effective optimizer steps.** dgxchen does 1
   epoch on the full CoT CSV; huikang sets `NUM_STEPS=1000` and clamps to
   `len(examples)//32`. Both end up in the same neighbourhood of optimizer
   steps. Longer training is not reported to help.

10. **Stratify or replay the teacher's training order, don't just shuffle
    randomly.** dgxchen implements a custom `StratifiedSFTTrainer` that
    round-robins across puzzle `type`; huikang explicitly replays the Tinker
    teacher's epoch-0 problem order from `logprobs/index.jsonl` and sets
    `SHUFFLE_DATASET=False`. The shared belief is that batch composition
    matters and naive shuffling hurts.

11. **The Tinker-style MoE expert tying is a real, non-default trick** for
    the MoE blocks: identical LoRA factor across all 128 experts at init, and
    gradient *sum* (not mean) across the expert dim before each AdamW step.
    huikang implements this in `_tie_grads()`. The saved adapter still
    contains 128 per-expert copies (which is what vLLM expects), but during
    training they remain identical. dgxchen doesn't tie — `embed_tokens` and
    `gate_proj` are simply not in his target list — so this trick is
    presumably worth a few tenths of a percent at most, not a make-or-break.

12. **Cut-Cross-Entropy is a real VRAM win at 8k seq len.** huikang patches
    `CausalLM.forward` to call `cut_cross_entropy.linear_cross_entropy(...)`
    so logits (B × T × 200k vocab) are never materialized. Without this, an
    8192-token sequence on a 30B-A3B MoE will not fit on a Kaggle GPU. It is
    *not* required if you're using `SFTTrainer` + Unsloth's own loss kernels
    (dgxchen doesn't use CCE and is fine), but it's necessary for the
    custom-loop / per-token-weight setup.

13. **At submission time, only ship `adapter_config.json` and
    `adapter_model.safetensors`.** Rewrite
    `base_model_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"`
    and set `inference_mode = True`. If `lm_head` is included in the adapter,
    rename keys `base_model.model.lm_head.*` →
    `base_model.model.backbone.lm_head.*` before zipping, otherwise vLLM
    silently ignores those LoRA weights at eval time.

14. **The teacher / dataset is Tong Hui Kang's `nemotron-master` corpus on
    GitHub (`tonghuikang/nemotron`).** Both genuine 0.85 notebooks pull from
    it; dgxchen via a CSV repackage (`problem_ids_matched.csv`), huikang via
    the pre-tokenized `training/sft/04-08-16-14/tokens` snapshot. If you want
    to beat 0.85 toward the 0.87 leader, distil your own CoT against a
    stronger teacher (e.g., Nemotron-Ultra or DeepSeek-R1-class) using the
    same `\\boxed{...}` reward extraction and rejection-sampling on
    correctness.

### TL;DR recipe to reproduce 0.85 (the converged formula)

- **Data**: huikang's `nemotron-master` CoT corpus (or your own distillation
  with the same `<think>...</think>\boxed{ans}` structure), per-token loss
  masked to the assistant turn only.
- **Model load**: `FastLanguageModel.from_pretrained(MAX_SEQ_LEN=8192,
  load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
  attn_implementation="eager", dtype=torch.bfloat16, trust_remote_code=True)`.
- **LoRA**: `r=32, alpha=32, dropout=0.0, bias="none",
  use_gradient_checkpointing="unsloth"`, target_modules = `["q_proj","k_proj",
  "v_proj","o_proj","in_proj","out_proj","up_proj","down_proj","lm_head"]`
  (manually re-add `lm_head` if Unsloth drops it on MoE).
- **Optimizer**: AdamW (full precision, NOT 8-bit), `lr=2e-4`,
  `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.0`, `max_grad_norm` off,
  **linear** decay, **zero warmup**.
- **Batch**: micro-batch=1 (or 4 if VRAM allows), grad-accum so effective
  batch = 32; 1 epoch or ≈1000 optimizer steps; stratify batches by puzzle
  type or replay teacher's epoch-0 order; `packing=False`.
- **At submission**: zip `adapter_config.json` + `adapter_model.safetensors`
  only; rewrite `base_model_name_or_path` and rename `lm_head` keys to
  `backbone.lm_head` so vLLM finds them.
