# 03 - Chain-of-Thought & Synthetic Reasoning Data Notebooks

This group focuses on the single biggest lever in the NVIDIA Nemotron Model Reasoning
Challenge: **how to make Nemotron-3-Nano-30B actually reason its way to `\boxed{...}`**.
The competition ships only 9,500 prompt + final-answer pairs (no native CoT), so the
top notebooks all converge on the same recipe: generate a synthetic chain-of-thought
for each puzzle using a stronger / external LLM (or a domain-specific solver),
**filter against the ground-truth answer**, and SFT a rank-32 LoRA on the surviving
(prompt, CoT, answer) tuples formatted with the Nemotron `<think>...</think>\boxed{...}`
chat template.

The four notebooks below show the full arc of the community discovery:

1. `kienngx/nvidia-nemotron-training-copy-run-instantly` (rank 11, 255 votes) -
   the **no-CoT baseline** - trains on raw `(prompt, answer)` and is the foil the
   later notebooks fix.
2. `kienngx/nvidia-nemotron-training-cot-labels` (rank 3, 647 votes) - the same
   recipe but pointed at a `generated_cot` column produced by a teacher model -
   the "drop-in CoT upgrade" that unlocks the public-LB pack.
3. `konbu17/nemotron-sft-lora-with-cot` (rank 6, 502 votes) - the most detailed
   description of a **type-specific, solver-augmented CoT generation pipeline**
   with rule-based correctness filtering and token-budget control.
4. `kienngx/nemotron-sft-reasoning-trajectories-dataset` (rank 19, 106 votes) - a
   community-curated "reasoning trajectories" dataset
   (`kishanvavdara/nemotron-reasoning-traj`) shoved through the same SFT loop.

The recurring pattern: **rank-32 LoRA on all-linear / `(in_proj|out_proj|up_proj|down_proj)`,
bf16, gradient checkpointing, cosine LR, 2 epochs, max_seq_len 2048-4096, packing off,
~2-6k filtered CoT samples**. The variable that moves the score is the *data*, not the
LoRA config.

---

## 1. `kienngx/nvidia-nemotron-training-copy-run-instantly` (rank 11, 255 votes)

The "no-CoT baseline" - identical training scaffolding to notebook #2 but pointed
at the raw competition CSV. Useful as the control to show why CoT is the win.

### What kind of CoT/reasoning data is used?

- **Raw final answer only** (column `answer` from `train.csv`).
- No teacher model, no synthetic CoT, no rejection sampling.
- The notebook itself flags this as a known weakness in a comment:
  > "CRITICAL FIX: You must ensure the reasoning steps exist BEFORE the boxed answer.
  >  If the 'answer' column only contains the number, you need to synthesize reasoning data."

So the author explicitly knows the data is wrong for a reasoning model, but ships
it as the "copy & run instantly" starter.

### Prompt / completion template

User message = puzzle prompt + literal suffix `"\nPut your final answer inside \\boxed{}."`.
Assistant message = the `answer` column verbatim (i.e. just the final value, no
reasoning, no `\boxed{}` wrapping, no `<think>` tags).

```python
def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"{answer}"
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}
```

- Uses `tokenizer.apply_chat_template`, with a manual `<|im_start|>user ...
  <|im_end|>\n<|im_start|>assistant ...<|im_end|>` fallback.
- No prompt-loss masking (default TRL SFT, so loss is computed over the *whole*
  templated string).
- No `<think>` block, no explicit `\boxed{}` wrapping - meaning the model is
  being told "produce just the bare answer" while inference will (per
  competition rules) extract from `\boxed{...}`. This is the bug the next
  notebook fixes.

### Data filtering / curation

- `train_df.sample(n=1200, seed=42)` - simple uniform subsample of 1,200 rows
  out of ~9,500. No type balancing, no length filter, no correctness check.

### Training framework + LoRA config + hyperparams

- Framework: **TRL `SFTTrainer` + PEFT LoRA**, base model
  `metric/nemotron-3-nano-30b-a3b-bf16` loaded with
  `AutoModelForCausalLM(..., dtype=torch.bfloat16, device_map={"": 0},
  trust_remote_code=True)`, `gradient_checkpointing_enable()`.
- Kaggle-specific kludges: pure-PyTorch `rmsnorm_fn` monkey-patch, a copied
  `ptxas-blackwell` binary, and `nv_compiler.get_ptxas_version = lambda arch: "12.0"`
  to make Triton happy.
- Mamba fast path disabled (`is_fast_path_available = False`) on every
  `modeling_nemotron_h` module.
- LoRA: `r=32, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05,
  bias="none", task_type=TaskType.CAUSAL_LM`.
- SFT hyperparams:
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`
  - `num_train_epochs=2`
  - `learning_rate=1e-4` (the cell comment says `2e-4`, the code says `1e-4`)
  - `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`
  - `max_length=2048`, `packing=False`
  - `bf16=True`, `max_grad_norm=1.0`, `optim="adamw_torch"`
  - `gradient_checkpointing=True`, `use_reentrant=False`
  - `save_strategy="no"` (Kaggle disk constraint)

### Reported LB score

- Not stated in the notebook. The CoT-labeled fork (notebook #2) is positioned
  as a strict upgrade.

### Why CoT helps here

Not discussed - this notebook is the *negative* example. The only acknowledgement
is the inline comment quoted above admitting that without reasoning before
`\boxed{...}` the training signal is wrong.

### Dataset references

- `dennisfong/nvidia-nemotron-offline-packages` - offline pip wheels (`datasets`,
  `trl`) so the kernel can run with internet disabled.
- `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1` - the base model.
- Reads `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv`
  (the raw competition CSV).
- Utility script: `ryanholbrook/nvidia-utility-script` (ships the
  `ptxas-blackwell` binary).

---

## 2. `kienngx/nvidia-nemotron-training-cot-labels` (rank 3, 647 votes)

The notebook is **structurally identical to notebook #1** - same Triton/RMSNorm
patches, same LoRA config, same SFTTrainer scaffold - but swaps the data source
for a CSV that already contains a `generated_cot` column. This is the "minimal
CoT upgrade" baseline that most of the leaderboard appears to fork.

### What kind of CoT/reasoning data is used?

- **Synthetic CoT pre-generated and shipped as a Kaggle dataset:**
  `kienngx/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv`.
- The notebook itself does not run the generation - it only consumes the
  `(prompt, answer, generated_cot)` triples. The dataset slug and the author's
  other notebooks strongly suggest the CoT was produced by a stronger external
  reasoner (community consensus on this competition is a DeepSeek-R1 / Qwen3
  family teacher, but the notebook does not state the teacher model
  explicitly).
- No on-the-fly rejection sampling here - filtering, if any, was done upstream
  when the labelled dataset was built.

### Prompt / completion template

```python
def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    cot = example["generated_cot"]                # the new piece

    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"   # CoT + explicit \boxed

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}
```

Key differences vs notebook #1:

- Assistant target = `<generated_cot>\n\n\boxed{<answer>}` - reasoning first,
  then the boxed final answer that the evaluator regex will catch.
- Still no `<think>` tags - this notebook leans on the chat template to add
  whatever the Nemotron tokenizer prepends and just plops the CoT plus the
  boxed answer into the assistant turn.
- No loss masking - full assistant string is supervised, but because the assistant
  string is now CoT + boxed answer, the model is trained to *produce* that
  format end-to-end.

### Data filtering / curation

- `train_df = train_df.sample(n=1200, seed=42)` - uniform 1,200-row subsample
  from the labelled CSV. No length, type or correctness filter applied inside
  the notebook.
- The dataset itself (`final_Nemotron_training_data.csv`) is presumably already
  filtered upstream - but that pipeline is not in this notebook.

### Training framework + LoRA config + hyperparams

Identical scaffolding to notebook #1; deltas:

- `learning_rate=5e-5` (notebook #1 used `1e-4`) - lower LR because the target
  string is now much longer (CoT + answer).
- `logging_steps=30` instead of 5.
- Everything else (LoRA rank/alpha/target_modules, batch size, grad accum,
  cosine LR, warmup_ratio=0.1, 2 epochs, max_len 2048, packing off, bf16,
  gradient checkpointing) is unchanged.

### Reported LB score

- Not stated explicitly inside the notebook, but its rank (3rd by votes) and
  the explicit "(CoT + Labels)" title imply this is the popular public-LB
  recipe in the mid-pack.

### Why CoT helps here

Implicit, not ablated:

- Notebook #1 trains on bare numeric answers - the model never sees how to
  reason and never produces `\boxed{...}` consistently.
- Notebook #2 swaps in CoT and an explicit `\boxed{<answer>}` suffix - the model
  learns both the reasoning chain *and* the boxed-answer extraction format the
  evaluator depends on. This is the entire point of the upgrade.

### Dataset references

- `kienngx/nemotron-30b-competition-trainingdata-cot-labels` - shipped CSV
  `final_Nemotron_training_data.csv` with columns `prompt`, `answer`,
  `generated_cot` (origin / teacher model not stated in this notebook).
- `dennisfong/nvidia-nemotron-offline-packages` - offline pip cache.
- `rubyducklove/nvidia-cutlass` - CUTLASS wheels (needed for Nemotron-H
  Mamba kernels in offline mode).
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`.
- Utility: `ryanholbrook/nvidia-utility-script`.

---

## 3. `konbu17/nemotron-sft-lora-with-cot` (rank 6, 502 votes)

The most carefully documented CoT data pipeline in the comp. Author explicitly
states the thesis: **"training data quality matters more than quantity"** and
walks through prompt engineering, rule-based correctness filtering, token
budget control and per-puzzle-type sampling.

### What kind of CoT/reasoning data is used?

- Custom-generated CoT for **all 9,000 train prompts**, then filtered down to
  6,558 verified-correct samples, then sampled per puzzle type to **2,907
  rows** for SFT.
- Teacher model: the notebook describes the generation as "LLM inference with
  `temperature=0.7`" but does not name the teacher in the notebook text. The
  pipeline lives in the Kaggle dataset
  `konbu17/nemotron-sft-lora-cot-selection` (file `train_split_with_cot.csv`),
  produced by the author's two earlier notebooks
  (`konbu17/doc-to-lora-knowledge-injection-nemotron-3-nano` and
  `konbu17/bit-manipulation-solver-cot-generator`).
- The Bit Manipulation traces in particular are *solver-augmented*: a Python
  per-bit boolean-function analyzer (`ID 40%, AND 15%, XOR 8%, ...`)
  enumerates candidate functions per output bit and embeds CERTAIN vs
  AMBIGUOUS structure in the prompt before the LLM reasons.
- Text Encryption traces use a *prompt-injected dictionary* (the 77-word Alice
  in Wonderland vocabulary) so the LLM gets the substitution table for free.

### Prompt / completion template

```python
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. ' \
                'For example: `\\boxed{your answer}`'

cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
user_content = prompt + PROMPT_SUFFIX
assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"

records.append({"messages": [
    {"role": "user",      "content": user_content},
    {"role": "assistant", "content": assistant_content},
]})
```

Important details:

- Uses the Nemotron *thinking* chat template - the author notes
  "chat template auto-adds `<think>\n`, so assistant starts with CoT directly"
  and the assistant string ends with `</think>\n\boxed{<answer>}`. So the
  on-wire training target is effectively:

  ```
  <think>
  <generated_cot>
  </think>
  \boxed{<ground_truth_answer>}
  ```

- The original teacher-generated `\boxed{...}` is **stripped out** of the CoT
  body (`re.sub(r'\\boxed\{[^}]*\}', '', cot)`) so the only boxed answer in
  the assistant string is the trusted ground-truth one - this prevents the
  model from learning to box wrong intermediate numbers.
- Loss is computed over the full assistant string (default TRL SFT).
- The prompt suffix matches the Kaggle official wording exactly:
  ``Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}` ``.

### Data filtering / curation (the headline contribution)

Pipeline, per the notebook's own description:

1. **Prompt construction** with the official `\boxed{}` suffix.
2. **Type-specific prompt engineering** (Text Encryption gets the 77-word
   dictionary; Bit Manipulation gets per-bit boolean-function candidate
   analysis; the four other types use generic step-by-step reasoning).
3. **LLM inference** at `temperature=0.7` to generate `generated_cot`.
4. **Rule-based correctness filtering** - extract `\boxed{...}` from the
   generation and compare to ground truth using *per-type* checkers:
   - Bit Manipulation: exact 8-bit binary match.
   - Gravitational Constant / Unit Conversion: numeric match within `+/- 0.05`
     absolute *or* `+/- 0.5%` relative.
   - Numeral Conversion: case-insensitive Roman numeral exact match.
   - Text Encryption: case-insensitive exact match.
   - Equation Transformation: exact string match.

   Only samples where the teacher's boxed answer was verified correct survive
   (rejection sampling).
5. **Token-length control** - the resulting CoT is tokenized with the
   Nemotron-3-Nano tokenizer; any `generated_cot` over **7,600 tokens** is
   sent back to the LLM with the instruction "shorten by `<overflow %>` while
   preserving the answer in `\boxed{}`". 7,600 (not 7,680) leaves headroom for
   the `</think>\n\boxed{<answer>}` suffix and stays within eval
   `max_tokens=7680`.

Pass-rate table reported (this is essentially the "synthetic data per puzzle
type" disclosure):

| Puzzle type            | Total avail. | Sampled | %     | Teacher pass rate |
|------------------------|-------------:|--------:|------:|------------------:|
| Gravitational Constant | 1,511        | 400     | 26.5% | 99.7% (easy)      |
| Numeral Conversion     | 1,491        | 300     | 20.1% | 99.6% (easy)      |
| Unit Conversion        | 1,342        | 700     | 52.2% | 88.7%             |
| Text Encryption        | 1,407        | 700     | 49.8% | 94.3% w/ dict     |
| Bit Manipulation       |   607        | 607     |  100% | 40.3% (hard)      |
| Equation Transformation|   200        | 200     |  100% | 13.6% (hardest)   |
| **Total**              | **6,558**    | 2,907   | 44.3% |                   |

Key implications for anyone copying the recipe:

- Bit Manipulation and Equation Transformation are *the bottleneck* - the
  teacher only solves them 40% / 14% of the time, so every verified-correct
  sample is precious and all are used.
- Easier types (Gravitational / Numeral) are deliberately downsampled
  (26% / 20%) to avoid drowning the harder types in the training mix.
- CoTs that are too long get *rewritten by the teacher* rather than truncated -
  preserves answer integrity.

Minor in-notebook filter on top of the upstream pipeline:

```python
if not cot or cot == "nan" or len(cot.strip()) < 5:
    continue   # drop empty / placeholder CoTs
```

### Training framework + LoRA config + hyperparams

- Framework: TRL `SFTTrainer` + PEFT LoRA, same Nemotron-3-Nano-30B base
  loaded with `device_map="auto"`, `torch_dtype=torch.bfloat16`.
- LoRA:
  - `r=32, lora_alpha=32, lora_dropout=0.05, bias="none"`
  - `target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$"` -
    the **explicit Nemotron-H module list** rather than the convenience
    `"all-linear"`. The author calls this "the standard competition target
    modules".
  - `task_type=TaskType.CAUSAL_LM`.
- SFTConfig:
  - `num_train_epochs=2` (comments show `# 1 # 2` - they tried both)
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
  - `learning_rate=1e-4` (comments: `# 5e-5 # 1e-4`)
  - `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`
  - `max_length=4096` (comments: `# 7680 # 4096`) - notably 2x the baseline's
    2048, to fit the longer CoTs
  - `bf16=True`, `gradient_checkpointing=True` (`use_reentrant=False`)
  - `packing=False`, `dataloader_num_workers=2`
  - `save_strategy="no"`, `report_to="none"`, `seed=123`
- Mamba kernels: explicitly stubs out `mamba_ssm.modules.mamba3` and
  `mamba_ssm.ops.cute.mamba3` etc. with empty `ModuleType`s, then sets
  `Mamba3 = None` - same "force the slow / non-fast-path" trick as the
  kienngx notebooks.
- Submission packaging mutates `adapter_config.json` after training to set
  `base_model_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"`,
  `inference_mode = True`, `lora_dropout = 0.0` before zipping - important
  because the trained config still references the kagglehub local path
  otherwise.

### Reported LB score

- No explicit numeric LB inside the notebook; the rank-6 / 502-vote slot puts
  it among the top public recipes, and the notebook's narrative frames it as
  "high quality > high quantity" rather than chasing a particular score.

### Why CoT helps here (explicit)

- Whole-notebook thesis statement, verbatim:
  > "Training data quality matters more than quantity: rather than using all
  >  9,000 training samples, I generated CoT reasoning via LLM inference and
  >  then kept only the correct answers verified by rule-based scoring."
- Implicit ablation in the type-table: types where the teacher passes >88% of
  the time (Gravitational, Numeral, Unit, Text Encryption) provide clean
  reasoning supervision; the hard types (Bit, Equation) need solver-injected
  prompts before the teacher can even produce usable CoT - i.e. CoT alone
  isn't enough on the hardest sub-tasks.
- Author also flags an untaken win:
  > "I didn't input solvers into Equation Transformation, so we may get
  >  better scores from inputting 'Equation Transformation' knowledge to solve."

### Dataset references

- `konbu17/nemotron-sft-lora-cot-selection` - the curated CSV
  `train_split_with_cot.csv` with `(type, prompt, answer, generated_cot)`,
  6,558 verified-correct rows; also ships the author's already-trained adapter
  for Mode B "skip training, just package".
- `mayukh18/nemotron-packages` - bundled wheels for `mamba_ssm`,
  `causal_conv1d`, `triton`, `trl`, `datasets`, `dill`, `xxhash` etc.
- `dennisfong/nvidia-nemotron-offline-packages` - fallback offline pip cache.
- `rubyducklove/nvidia-cutlass` - CUTLASS wheels for Nemotron-H kernels.
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`.
- Author's upstream pipelines referenced:
  - `konbu17/doc-to-lora-knowledge-injection-nemotron-3-nano` (Text Encryption
    dictionary / multi-step SFT).
  - `konbu17/bit-manipulation-solver-cot-generator` (per-bit boolean function
    analysis).

---

## 4. `kienngx/nemotron-sft-reasoning-trajectories-dataset` (rank 19, 106 votes)

A near clone of notebook #3 by a different author, but pointed at a *different
community-curated CoT dataset*: `kishanvavdara/nemotron-reasoning-traj`.
The interesting bit is that it shows how `konbu17`'s recipe generalizes when
fed an externally-produced trajectories dataset.

### What kind of CoT/reasoning data is used?

- Pre-baked "reasoning trajectories" dataset
  `kishanvavdara/nemotron-reasoning-traj/nemotron_traj.csv` with columns
  `problem type`, `correct answer`, `generated` and a `correctness` column
  ("true" / "false").
- Teacher model is not named in the notebook, but the dataset slug and the
  presence of a boolean correctness flag implies the same teacher-generation +
  rejection-sampling pipeline.
- First filter: keep only rows where the teacher got the answer right:
  ```python
  df = df[df['correctness'] == 'true']
  ```
- Second remapping: collapse the dataset's puzzle-type taxonomy to the
  competition's six canonical types:
  ```python
  mapping = {
      "numeral":            "Numeral Conversion",
      "gravity":            "Gravitational Constant",
      "unit_conversion":    "Unit Conversion",
      "cipher":             "Text Encryption",
      "bit_manipulation":   "Bit Manipulation",
      "equation_numeric":   "Equation Transformation",
      "equation_symbolic":  "Equation Transformation",
  }
  ```
  Note that `equation_numeric` and `equation_symbolic` are both folded into
  Equation Transformation - so this dataset distinguishes sub-flavours of
  equation problems that the competition prompt does not.
- Columns are renamed to match konbu17's pipeline (`type`, `answer`,
  `generated_cot`).

### Prompt / completion template

Identical to notebook #3:

```python
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. ' \
                'For example: `\\boxed{your answer}`'

cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
user_content = prompt + PROMPT_SUFFIX
assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
```

i.e. Nemotron thinking template, `</think>` close + `\boxed{<gt>}` suffix,
teacher-emitted `\boxed{...}` stripped from the CoT body.

### Data filtering / curation

- Hard filter on `correctness == "true"`.
- Per-type sampling table:

  | Puzzle type             | n samples sampled |
  |-------------------------|------------------:|
  | Numeral Conversion      | 700               |
  | Gravitational Constant  | 600               |
  | Unit Conversion         | 700               |
  | Text Encryption         | 483               |
  | Bit Manipulation        | 128               |
  | Equation Transformation | 178               |

  vs konbu17's 300/400/700/700/607/200 - this dataset is much *thinner* on
  Bit Manipulation (128) and Equation Transformation (178), reinforcing the
  earlier observation that hard puzzles are scarce after correctness filtering.
- In-notebook drop for empty / short CoTs (`len(cot.strip()) < 5`).

### Training framework + LoRA config + hyperparams

Same scaffolding as konbu17; deltas to flag:

- `per_device_train_batch_size=2` (vs 1) - half the grad-accum micro-steps.
- `gradient_accumulation_steps=8` (same effective batch ~16).
- `learning_rate=1e-5` (vs `1e-4`) - **10x smaller** than konbu17's
  hyperparams. Possibly tuned for the different dataset, or just a more
  conservative LR for a smaller / noisier corpus.
- `dataloader_num_workers=8` (vs 2).
- `seed=42` (vs 123).
- Everything else identical: 2 epochs, cosine, warmup 0.05, max_len 4096,
  packing off, bf16, gradient checkpointing, LoRA `r=32, alpha=32`,
  target_modules `(in_proj|out_proj|up_proj|down_proj)`.

### Reported LB score

Not stated. The rank-19 / 106-vote slot suggests this performs worse than
konbu17's own dataset - consistent with the much thinner Bit Manipulation /
Equation Transformation pools after correctness filtering.

### Why CoT helps here

Same implicit thesis - the only data the notebook is willing to train on is
trajectories that the teacher got *right*. The correctness flag is the entire
filter; the notebook does not run its own per-type checker (it trusts the
upstream label).

### Dataset references

- `kishanvavdara/nemotron-reasoning-traj` - the external "reasoning
  trajectories" CSV (file `nemotron_traj.csv`) with `correctness` flag.
- `konbu17/nemotron-sft-lora-cot-selection` - referenced as the "Mode B"
  pretrained-adapter fallback (Mode B uses konbu17's adapter, not this
  notebook's data).
- `mayukh18/nemotron-packages` - Mamba/Triton wheels.
- `dennisfong/nvidia-nemotron-offline-packages` - offline pip cache.
- `rubyducklove/nvidia-cutlass` - CUTLASS wheels.
- Model: `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`.

---

## Synthesis

Distilling across the four notebooks, the *data-pipeline* consensus is:

### What changes when you add CoT

Going from notebook #1 (no CoT) to notebook #2 (CoT + boxed-answer suffix) is
the cheapest, biggest win in the comp:

- Without CoT, the assistant target is just the raw answer string -> the model
  never learns to (a) reason or (b) emit `\boxed{...}` reliably, both of which
  the vLLM greedy / regex evaluator depends on.
- With CoT, the assistant target is `<reasoning>\boxed{<answer>}` (or, on the
  thinking template, `<cot>\n</think>\n\boxed{<answer>}`) - this matches the
  inference contract exactly.
- The same TRL + LoRA scaffolding works for both - the only delta that matters
  is the *target string*.

### Teacher model(s) - the candid picture

None of these four notebooks names a teacher model in code or markdown.
What they reveal:

- All synthetic CoT in this group comes from "LLM inference at temperature
  0.7" (konbu17), or from community-curated trajectories CSVs (kienngx,
  kishanvavdara). The dataset slugs - `*nemotron-reasoning-traj*`,
  `*nemotron-30b-competition-trainingdata-cot-labels*`,
  `*nemotron-sft-lora-cot-selection*` - all carry Nemotron in the name but
  none disclose whether the teacher was Nemotron-49B-Reasoning, DeepSeek-R1,
  Qwen3, or something else.
- The pattern across the dataset slugs strongly suggests "any sufficiently
  strong reasoning LLM + ground-truth rejection sampling" rather than a
  specific named teacher.
- konbu17 is explicit that for the *hardest* types (Bit Manipulation,
  Equation Transformation) the teacher alone is insufficient and must be
  *augmented with a Python solver in the prompt* (per-bit boolean function
  analysis, 77-word substitution dictionary) before it can produce
  reliably-correct CoT.

**Takeaway for a winning approach:** the teacher identity matters less than
(a) feeding the teacher domain-specific solver outputs in-context for the
hard puzzles, and (b) ruthlessly rejection-sampling against ground truth.

### Winning CoT format

Across the three CoT notebooks (#2, #3, #4) the convergent format is:

```
<user>   <puzzle prompt>
         Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
<asst>   <think>
         <generated CoT, with any stray \boxed{...} stripped out>
         </think>
         \boxed{<ground truth answer>}
```

Notes:

- Two notebooks (kienngx #2 and konbu17 #3 / #4) wrap the CoT in the
  Nemotron *thinking* template `<think>...</think>`; the earliest baseline
  (kienngx #2 plain) puts the CoT outside any think tags. The thinking-tag
  version is the one that the higher-vote konbu17 recipe uses.
- Always strip teacher-emitted `\boxed{...}` from inside the CoT body and
  re-append a single, trusted `\boxed{<ground truth>}` outside (or after
  `</think>`).
- The prompt-suffix wording matches the official Kaggle example exactly:
  ``Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}` ``.
- No prompt-loss masking is applied in any of these notebooks - TRL SFT
  default behaviour (loss on the whole templated string) is used.

### Size & quality of the most-used reasoning datasets

| Dataset slug                                                              | Rows used  | Filtered? | Notes |
|---------------------------------------------------------------------------|-----------:|-----------|-------|
| `kienngx/nemotron-30b-competition-trainingdata-cot-labels`                | 1,200 sample of full CSV | Upstream | No type balancing in notebook |
| `konbu17/nemotron-sft-lora-cot-selection` (`train_split_with_cot.csv`)    | 2,907 of 6,558 verified-correct | **Yes - rule-based, per-type** | 9,000 -> 6,558 (verified) -> 2,907 (per-type sampled) |
| `kishanvavdara/nemotron-reasoning-traj` (`nemotron_traj.csv`)             | 2,789 (sum of per-type caps) | **Yes - `correctness == "true"`** | Bit Manip and Equation Transformation are scarce (128 / 178) |

The *quality bottleneck* is Bit Manipulation and Equation Transformation -
they have low teacher pass rates (~40% and ~14% in konbu17's pipeline) and
scarce verified-correct samples (607 / 200 at best; 128 / 178 in
kishanvavdara). Multiple authors converge on "use all verified-correct
samples for the hard types, downsample the easy ones".

### Is rejection sampling standard? - Yes.

Three of the four notebooks (notebooks #2, #3, #4) either consume a
rejection-sampled dataset or run rejection sampling upstream:

- **Notebook #2** (`cot-labels`): consumes a pre-filtered CSV; filtering
  policy not in the notebook but implied by the dataset name.
- **Notebook #3** (`konbu17`): explicit per-type rule-based filter (exact
  match / numeric tolerance / Roman exact / etc.). This is the gold standard
  description in the comp.
- **Notebook #4** (`reasoning-trajectories`): filters by a boolean
  `correctness` column shipped with the dataset.

Rejection sampling against ground truth is effectively a precondition for a
competitive entry in this group.

### Training recipe consensus (so we can stop tuning it)

- **Base model:** `metric/nemotron-3-nano-30b-a3b-bf16` loaded with
  `trust_remote_code=True`, `dtype=torch.bfloat16`, `device_map="auto"` or
  `{"": 0}`.
- **LoRA:** `r=32, lora_alpha=32, lora_dropout=0.05, bias="none"`,
  `task_type=CAUSAL_LM`. Target modules either `"all-linear"` (kienngx) or
  the explicit `(in_proj|out_proj|up_proj|down_proj)` regex (konbu17). The
  explicit regex is the one used by the highest-voted konbu17 notebook.
- **SFT (TRL `SFTTrainer` / `SFTConfig`):**
  - `num_train_epochs=2`
  - `per_device_train_batch_size=1-2`, `gradient_accumulation_steps=4-8`
  - `learning_rate=1e-5` to `1e-4` (the higher CoT-target notebooks use
    `5e-5 - 1e-4`; the safer `1e-5` shows up in the trajectories notebook)
  - `lr_scheduler_type="cosine"`, `warmup_ratio=0.05-0.1`
  - `bf16=True`, `gradient_checkpointing=True (use_reentrant=False)`
  - `max_length`: 2048 for short-CoT, **4096 once you adopt CoT**, with
    capacity for 7680 if needed.
  - `packing=False`, `save_strategy="no"`, `report_to="none"`.
- **Kaggle environment kludges (recurring):**
  - Monkey-patch `rmsnorm_fn` with a pure-PyTorch implementation.
  - Copy `ptxas-blackwell` to `/tmp`, chmod, and set
    `TRITON_PTXAS_PATH` + `TRITON_PTXAS_BLACKWELL_PATH`.
  - `nv_compiler.get_ptxas_version = lambda arch: "12.0"`.
  - Stub `mamba_ssm.modules.mamba3` and `mamba_ssm.ops.cute.mamba3` modules
    with empty `ModuleType`s and set `Mamba3 = None`; also
    `is_fast_path_available = False` on every `modeling_nemotron_h` module.
  - Offline wheels from `dennisfong/nvidia-nemotron-offline-packages`,
    `mayukh18/nemotron-packages`, `rubyducklove/nvidia-cutlass`.
- **Submission packaging:**
  - Zip exactly `adapter_config.json` + `adapter_model.safetensors`.
  - Before zipping, rewrite `adapter_config.json` to set
    `base_model_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"`,
    `inference_mode = True`, `lora_dropout = 0.0`.

### Open levers (where the next gain probably lives)

- **Pre-prompt the teacher with solver output for *every* hard puzzle type**,
  not just Bit Manipulation and Text Encryption. konbu17 explicitly flags
  Equation Transformation as untouched (only 13.6% teacher pass rate, 200
  samples).
- **Multi-pass rejection sampling** with diverse teachers / temperatures to
  build a deeper pool for Bit Manipulation and Equation Transformation,
  where current verified-correct counts (607 / 200) are the binding
  constraint.
- **Explicit prompt-loss masking** - none of these notebooks mask the user
  turn, which wastes capacity on memorizing puzzle phrasing. Switching to
  `DataCollatorForCompletionOnlyLM` (or TRL's `assistant_only_loss`) would
  free signal for the CoT.
- **Per-type adapters or per-type sampling curricula** - the per-type pass
  rates differ by ~10x, and a single LoRA tries to learn all six.
