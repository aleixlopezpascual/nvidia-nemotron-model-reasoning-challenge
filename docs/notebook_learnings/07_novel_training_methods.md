# Novel training methods beyond the 0.85 recipe

This document is a round-two read of five Kaggle notebooks for the NVIDIA
Nemotron Model Reasoning Challenge that each *attempt* to introduce a training
ingredient not present in the convergent 0.85 recipe (see
`04_top_scoring_0_85_lb.md`). The baseline that recipe rests on is: Unsloth
LoRA r=32/α=32/dropout=0 on `q,k,v,o,in_proj,out_proj,up,down,(lm_head)`,
AdamW (β2=0.95), lr=2e-4, linear decay, zero warmup, no grad clip, effective
batch=32, 1 epoch / ~1000 steps over huikang's distilled CoT corpus, seq=8192.
The candidate "novel" ingredients reviewed here are: (1) Muon optimizer,
(2) loss-masking via `completion_only_loss`, (3) soft-balanced sampling over
puzzle types, (4) a "Think Twice" self-correcting CoT pattern,
(5) SFT-then-GRPO with a custom reward stack. For each notebook I extract the
novel idea, the exact code, what the author compared against, any reported LB
number, the hyperparameters that *had* to change, the risks, and a final
verdict on whether the idea is likely to materially lift accuracy.

The headline finding: only one of the five (mask-loss) is essentially free and
likely a real lift in cases where the baseline forgot it; the Muon notebook is
the most interesting research idea but lacks the LB-confirmed evidence its
title implies; soft-balanced sampling and "Think Twice" are mostly cosmetic;
and the SFT+GRPO notebook is explicitly self-reported as "GRPO not helping
yet" — and ships with GRPO disabled.

---

## 1. pkuszboi — `0-85-lb-training-with-muon` (Muon optimizer)

**Author/ref**: `pkuszboi/0-85-lb-training-with-muon` ("[0.85 LB] Training with
Muon"). Forked from dgxchen's 0.85 LB notebook.

### Novel idea (one sentence)
Swap AdamW for the **Muon optimizer** (Newton–Schulz orthogonalisation of
momentum, applied to LoRA matrix parameters) while keeping dgxchen's data,
8-module LoRA layout (now with `lm_head`), and stratified-by-type SFT loop.

### Implementation details (verbatim)
The optimizer is created inside a `StratifiedSFTTrainer.create_optimizer`
override that gathers only LoRA parameters:

```python
def create_optimizer(self):
    if self.optimizer is None:
        print("====== 正在创建自定义 Optimizer ======")
        matrix_params = [
            p
            for name, p in self.model.named_parameters()
            if "lora" in name
        ]
        self.optimizer = Muon(
            matrix_params,
            lr=2e-3,
            momentum=0.95,
            backend_steps=5,
        )
    return self.optimizer
```

The Muon class and its Newton–Schulz iterator are pasted verbatim from the
public Muon reference:

```python
@torch.compile()
def zeropower_via_newtonschulz5(G, steps: int = 10, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            curr = 0
            for i, p in enumerate(params):
                if p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    p.add_(g, alpha=-lr)

        return loss
```

Critically, Muon is applied to **every** LoRA parameter (i.e. both LoRA-A and
LoRA-B for every target module), not just the "matrix" weights it is designed
for. The author labels the list `matrix_params` but the filter is just
`"lora" in name`, which sweeps embeddings/heads-LoRA factors in too.
This is a mild mis-application of the Muon convention (which typically wants
AdamW for embeddings, head, scalars, biases, and Muon only on hidden-state
matmul weights), and worth flagging because it can produce silent
mis-orthogonalisation on highly rectangular factors.

### Baseline comparison
The notebook is forked from dgxchen and the only meaningful change is the
optimizer + microbatch swap (per_device_train_batch_size from 1 → 2,
grad_accum from 32 → 8, effective batch still 32). The author does not
report a head-to-head Muon-vs-AdamW LB delta. The title claims 0.85 LB but
no submission score is printed; markdown reads "I use Muon Optimizer to
train a model that can reach around 0.85". This is the same number
dgxchen's recipe was already getting with AdamW — so the **Muon swap is
NOT shown to lift LB**, only to "still hit 0.85".

### Reported LB lift
None claimed beyond parity with the dgxchen baseline. Wall-clock is reported
as ~5 hours (vs dgxchen's 6–7h), which would be Muon's main practical
benefit if it actually reproduces 0.85.

### Hyperparameters changed for Muon
- `lr=2e-3` (10× the AdamW lr=2e-4 used by dgxchen/huikang). This is the
  standard advice for Muon, which "wants" a higher LR than AdamW because the
  spectral-norm-bounded update is much smaller per step in operator-norm
  terms.
- `momentum=0.95` (Muon's β-equivalent). No betas tuple — Muon only has the
  one momentum scalar.
- `backend_steps=5` (NS iterations per step). Reference value.
- `nesterov=True` by default — applied unconditionally.
- `per_device_train_batch_size=2, gradient_accumulation_steps=8` — note this
  is exactly the regime dgxchen's own ablation says costs ~0.01 LB. The
  author seems unaware of dgxchen's micro-batch=1 finding. Effective batch
  size is preserved at 16, but in dgxchen's recipe effective batch was 32.
  So this notebook has **both** a different optimizer and a smaller effective
  batch than the recipe it claims to match.
- LoRA module list now includes `lm_head`, contra dgxchen's deliberate
  removal — bringing it back in line with huikang's setup.

### Risks / failure modes
- **No LR sweep**: lr=2e-3 is the Muon reference default but is wildly off
  for AdamW-trained baselines. If the comparison is to the dgxchen baseline,
  it's unclear whether the lift/parity is from Muon or from the
  reintroduction of `lm_head`.
- **`lm_head` LoRA isn't a matrix update Muon was designed for** — it's a
  very rectangular `hidden × vocab` factor, and the NS iteration's
  per-iteration norm assumes a roughly square or wide matrix. The
  `max(1, g.size(0)/g.size(1))**0.5` scale fix partially corrects for this
  but isn't a substitute for treating vocabulary/embedding rows with AdamW.
- **No grad clipping** (max_grad_norm not set in the new SFTConfig — it
  defaults to the trl default of 1.0, which contradicts the recipe's
  "max_grad_norm=1e9 / effectively off" choice). The original recipe
  deliberately turns clipping off; the Muon notebook silently keeps it on
  and may be clipping away exactly the kind of large-singular-value updates
  Muon is designed to surface.
- **Effective batch=16** (the author dropped from 32). dgxchen explicitly
  warns this loses ~0.01 LB. So even if Muon were neutral, this notebook
  starts in a hole vs. the recipe.
- **`Muon` is applied to LoRA-A and LoRA-B together**, but LoRA-A is
  typically Kaiming-init normal, LoRA-B is zero-init. Orthogonalising a
  zero-init matrix on the very first step is a no-op, but it still consumes
  an NS pass and a memory allocation per step.

### Verdict
**Curiosity bordering on plausibly useful, but not demonstrated.** Muon has
real momentum behind it on small-scale language pretraining and on
Modded-NanoGPT-style training, and the LoRA-only application is a natural
fit (LoRA factors are small matmul weights). But this notebook does not run
the ablation: there's no AdamW-vs-Muon side-by-side on the same data and
same effective batch, no LR sweep, the effective batch is in a regime
dgxchen says is sub-optimal, and the headline number is just "still ~0.85".
**Worth implementing as a controlled ablation against the recipe** (Muon
only on `q,k,v,o,in/out_proj,up,down` LoRA-A+B, AdamW on `lm_head` LoRA, lr
sweep 1e-3 to 4e-3, effective batch=32) — if it lifts to 0.86+ that's a
real result. If it ties at 0.85, it's a wall-clock win and nothing more.

---

## 2. waterjoe — `nvidia-nemotron-mask-loss` (loss masking)

**Author/ref**: `waterjoe/nvidia-nemotron-mask-loss` ("NVIDIA Nemotron - Mask
Loss").

### Novel idea (one sentence)
Pass **`completion_only_loss=True`** into `SFTConfig` so the SFTTrainer
masks loss to the assistant turn only — this is the same "loss only on
responses" gap that mayukh18 closes via `train_on_responses_only` and
huikang bakes into his pre-tokenized weight masks.

### Implementation details (verbatim)
The relevant line is one keyword in the `SFTConfig`:

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    completion_only_loss=True,
)
```

That last keyword — `completion_only_loss=True` — is the entire novel claim
of the notebook. It instructs trl's `SFTTrainer` to use the chat-template
boundaries (between the user turn and the assistant turn) to mask the loss
to the completion portion only.

For the input format, the assistant body is the raw boxed answer (no CoT):

```python
def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"\\boxed{{{answer}}}"
    ...
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}
```

### Baseline comparison
None. The notebook does not run an experiment, does not produce an LB number,
and does not even mention what it's comparing against — it's strictly a
"here's the SFT pipeline with `completion_only_loss=True` flipped on"
demonstration.

### Reported LB lift
**None**. No LB number is shown.

### Hyperparameters changed for the new method
None for the mask itself. The notebook does use a much smaller training run
(SUBSAMPLE_SIZE=600, MAX_SEQ_LEN=1024, NUM_EPOCHS=1, GRAD_ACCUM=4, LR=2e-4,
LoRA r=32 with `alpha=16, dropout=0.05, target_modules="all-linear"`) but
those are not consequences of masking — they're just a small-scale demo.
Also notable: `lora_alpha=16` (not 32 — scaling = 0.5), `lora_dropout=0.05`
(not 0.0 — incompatible with Unsloth's fast LoRA kernels), and
`target_modules="all-linear"` (not the curated 8/9-module list). These are
all *worse* than the recipe defaults.

### Risks / failure modes
- `completion_only_loss=True` is only well-defined for chat-formatted
  datasets. The notebook does pass through `apply_chat_template` so this is
  fine here, but it would silently break if the prompt format changed.
- For Nemotron-H specifically, the assistant tokens include the
  `<|im_start|>assistant\n` header and any `<think>` block opening — the
  exact boundary trl uses is implementation-defined. In this notebook the
  assistant body is just `\boxed{answer}` with no `<think>` block, so the
  masked region is tiny and the trainer is essentially memorising a single
  short token-sequence per example. With CoT data this would mask in the
  full reasoning trace — much more useful.
- The author uses `adamw_torch` (not 8-bit, not the recipe's full-precision
  AdamW with β2=0.95) and `max_grad_norm=1.0` (recipe wants ~off), so any
  delta vs baseline confounds optimizer/clipping changes with the
  masking change.

### Verdict
**Conceptually correct and essentially free, but the notebook itself does
not prove anything.** Loss-masking the assistant turn is unambiguously the
right default for SFT — it's what huikang bakes into his tokenized corpus
and what mayukh18 gets via `train_on_responses_only`. If your baseline
trains the loss across the full prompt+response stream, *flipping
`completion_only_loss=True` is a likely accuracy lift of multiple points*
(this is well-attested in the SFT literature). But that's a statement about
the SFT default in general, not about this notebook, which makes no
controlled measurement and ships with a sub-baseline LoRA configuration
that would underperform regardless. **Use the flag; don't take the
notebook's other choices.**

---

## 3. newduck — `nvidia-nemotron-soft-balanced-sampling-sft` (soft-balanced sampling)

**Author/ref**: `newduck/nvidia-nemotron-soft-balanced-sampling-sft` ("NVIDIA
Nemotron - Soft-Balanced Sampling + SFT").

### Novel idea (one sentence)
Heuristically classify each `train.csv` prompt into one of four puzzle
categories (`bit`, `string`, `number`, `math`) via regex, then sample a
fixed-size training subset by **interpolating between the original category
distribution and a fully-balanced distribution** with a single mixing
coefficient `alpha`.

### Implementation details (verbatim)
The interpolation is a 4-line formula:

```python
def soft_balance_sample(df, total=600, alpha=0.5):
    """
    alpha = 0 keeps the original distribution.
    alpha = 1 uses a fully balanced distribution.
    """
    counts = df['category'].value_counts()
    orig_ratio = counts / counts.sum()

    balanced = {c: 1/len(counts) for c in counts.index}
    target_ratio = (1-alpha)*orig_ratio + alpha*pd.Series(balanced)

    samples = []
    for c in counts.index:
        n = int(total * target_ratio[c])
        subset = df[df.category == c]
        samples.append(subset.sample(min(n, len(subset)), random_state=SEED))
    return pd.concat(samples)
```

The category classifier is a hand-tuned regex scoring function:

```python
def classify_prompt(text):
    text_lower = text.lower()
    scores = {"bit": 0, "string": 0, "number": 0, "math": 0}
    if re.search(r'[01]{6,}', text):              scores["bit"] += 3
    if "bit" in text_lower or "binary" in text_lower: scores["bit"] += 2
    if "->" in text:                              scores["string"] += 3
    if re.search(r'[a-z]{3,}\s*->\s*[a-z]{3,}', text_lower): scores["string"] += 2
    if re.search(r'[!@#$%^&*()\[\]{}]', text):    scores["string"] += 1
    if re.search(r'\b[ivxlcdm]+\b', text_lower):  scores["number"] += 3
    if re.search(r'\d+\.\d+', text):              scores["number"] += 2
    if re.search(r'\d+\s*(cm|mm|kg|m)', text_lower): scores["number"] += 2
    if re.search(r'[\+\-\*/=]', text):            scores["math"] += 2
    if re.search(r'\d+\s*[\+\-\*/]\s*\d+', text): scores["math"] += 2
    if any(op in text for op in ["@", "|", "!", "%"]): scores["math"] += 1
    return max(scores, key=scores.get)
```

Hyperparams: `SAMPLE_SIZE=600`, `BALANCE_ALPHA=0.5`. The pipeline then runs
a vanilla 2-epoch LoRA SFT (`r=32, alpha=16, dropout=0.05,
target_modules="all-linear"`, lr=2e-4 cosine, effective batch 4, max_len
2048, `adamw_torch`).

### Baseline comparison
None. The notebook explicitly states "does not: full hyperparameter search,
synthetic data, multi-run ensembling" — it's a pipeline demo, not an
experiment. There's no comparison to `alpha=0` (original distribution) or
`alpha=1.0` (fully balanced), nor to a no-sampling baseline.

### Reported LB lift
**None**. No LB number is shown.

### Hyperparameters changed for the new method
- `BALANCE_ALPHA=0.5` (the new method's only hyperparameter).
- `SAMPLE_SIZE=600` (very small — the recipe uses thousands of CoT rows).
- The 4-category grouping is a crude proxy for the actual 6 puzzle types
  the competition has; in particular `bit` collapses to a 6-binary-digit
  regex which would mis-classify many Equation Transformation prompts (most
  contain digits) as `math` rather than the correct puzzle type.

### Risks / failure modes
- **Tiny training set (600 rows) on raw `train.csv` (no CoT)**. With or
  without balancing, this is far below the recipe's data scale, and
  trains the model to emit `\boxed{answer}` directly with no reasoning.
- **Coarse category labels.** The competition has 6 well-known puzzle
  types; collapsing to 4 hand-rolled regex categories likely
  mis-classifies a substantial fraction (e.g. "bit" via the `[01]{6,}`
  pattern would catch encoded numbers in any puzzle type).
- **The novel idea is a knob with no measured optimum.** alpha=0.5 is a
  guess; the notebook doesn't even check whether it helps over alpha=0.
- **`target_modules="all-linear"`** with PEFT on a hybrid Mamba+MoE model
  is unpredictable — it depends on PEFT's heuristic for what counts as a
  "linear" module on this architecture. The recipe's explicit
  9-module list is safer.
- **`adamw_torch` + `max_grad_norm=1.0` + `lora_dropout=0.05`** all break
  the recipe's optimizer-side defaults and would underperform
  independent of sampling.

### Verdict
**Curiosity at best.** The recipe's stratification (dgxchen's
`StratifiedSFTTrainer` round-robins by `type`) already addresses
batch-composition fairness with the *correct* puzzle types and without
discarding examples. The soft-interpolation knob between empirical and
uniform is a legitimate idea in general (it's effectively temperature on the
class distribution), but here it's applied to a 4-class regex proxy on a
600-row subsample with a worse LoRA config, and never measured. **Skip:**
if you want class-balanced SFT, use dgxchen's stratified sampler with
explicit puzzle-type labels (e.g. from a known classifier or from the CoT
corpus' `type` column) rather than this approach.

---

## 4. hammadfarooq470 — `think-twice-self-correcting-reasoning` (NOT actually self-correction)

**Author/ref**: `hammadfarooq470/think-twice-self-correcting-reasoning`
("Think Twice: Self-Correcting Reasoning").

### Novel idea (one sentence)
**The title is misleading**: the notebook contains no self-correction logic
whatsoever. What it actually does is monkey-patch tinker_cookbook's
`_merge_fused_projections` to **truncate the merged-fused LoRA back to rank
32 via SVD**, so a >32-rank Tinker-trained adapter can be packaged for
Kaggle's rank-32 cap.

### Implementation details (verbatim)
The "Think Twice" claim has no associated code. The real novel artefact is
this monkey-patch:

```python
FORCED_FUSED_RANK = 32

def _compress_lora_pair_to_rank(B: torch.Tensor, A_mat: torch.Tensor, rank: int):
    # Delta = B @ A, shape [out_dim, in_dim]
    delta = B.float() @ A_mat.float()
    # Best rank-k approximation in Frobenius norm
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    sroot = torch.sqrt(S)
    B_new = U * sroot.unsqueeze(0)
    A_new = sroot.unsqueeze(1) * Vh
    return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous()

def patched_merge_fused_projections(...):
    ...
    merged_lora_A = torch.cat(lora_A_parts, dim=0)
    merged_lora_B = ...  # block-diagonal merge of per-component LoRA-Bs
    final_rank = merged_rank
    if merged_rank > FORCED_FUSED_RANK:
        merged_lora_B, merged_lora_A = _compress_lora_pair_to_rank(
            merged_lora_B, merged_lora_A, FORCED_FUSED_RANK
        )
        final_rank = FORCED_FUSED_RANK
    ...

A._merge_fused_projections = patched_merge_fused_projections
```

Then it builds the adapter via:

```python
from tinker_cookbook import weights
weights.build_lora_adapter(
    base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    adapter_path="/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
    output_path="/kaggle/working/nemotron-adapter-ready-to-submit",
)
```

So this is an **adapter-packaging utility** for fused QKV / gate-up LoRA
that would otherwise exceed rank 32 after merging the per-component
factors, not a training method and not a CoT pattern.

### Baseline comparison
None. The notebook reads in huikang's existing adapter (model_sources lists
`huikang/nemotron-adapter` v20) and re-exports it with the rank-cap fix.

### Reported LB lift
None. There is no evaluation, no training run, no LB number.

### Hyperparameters changed for the new method
`FORCED_FUSED_RANK = 32`. That's it.

### Risks / failure modes
- **SVD truncation discards information.** A merged rank-(r1+r2+r3)
  adapter compressed back to rank 32 loses the singular values beyond the
  cap. If the original adapter was trained at, say, r=16 per component
  with 3 components fused (r=48), truncating to 32 discards a third of
  the spectrum. For sufficiently dominant top-k singular values this is
  near-lossless, but the notebook never verifies.
- **The title is wrong.** Anyone searching for self-correcting reasoning
  patterns will not find them here. There is no "think twice" prompting
  template, no second-pass critic, no verifier loop.
- **Depends on `tinker-cookbook` internals.**
  `tinker_cookbook.weights._adapter._merge_fused_projections` is a private
  function — the monkey-patch will silently break with future
  tinker-cookbook releases.

### Verdict
**Skip — the headline is misleading and the underlying utility is narrow.**
The rank-32 SVD compression is a real and reasonable trick for packaging
Tinker-trained fused-projection LoRAs that exceed Kaggle's rank cap, but
that's a *submission-packaging* artefact and there are other public
notebooks (`asalhi`, `afr1ste`, `rauffauzanrambe`,
`mohamedamr992`) covering the same ground more carefully and without
the deceptive title. There is no self-correction method to evaluate.

---

## 5. amanatar — `nemotron-ultimate-sft-grpo-v3` (SFT + GRPO with multi-reward stack)

**Author/ref**: `amanatar/nemotron-ultimate-sft-grpo-v3`
("nemotron-ultimate-sft-grpo-v3"). The "v3" suggests two prior iterations.

### Novel idea (one sentence)
A two-stage training pipeline: standard SFT on konbu17's CoT-labeled data
followed by **GRPO with a three-reward stack — cosine-scaled accuracy
reward (with numeric tolerance), format reward, and a "reasoning quality"
reward** that scores step-marker keywords, math symbols, pre-boxed prose
length, and word uniqueness — but **GRPO is disabled by default** and the
author's own comment says "GRPO not helping yet — model needs more SFT
first".

### Implementation details (verbatim)

Mode flags:
```python
USE_GRPO = False    # GRPO not helping yet — model needs more SFT first
USE_COT_DATA = True # Use konbu17's verified CoT data
```

The accuracy-tolerant matching + cosine-scaled reward:
```python
def _answers_match(pred, gt, rel_tol=1e-2):
    pred_n, gt_n = _normalize_answer(pred), _normalize_answer(gt)
    if pred_n == gt_n: return True
    if pred_n.lower() == gt_n.lower(): return True
    try:
        p, g = float(pred), float(gt)
        if g == 0: return abs(p) < 1e-6
        return abs(p - g) / abs(g) <= rel_tol
    except (ValueError, OverflowError):
        return False

def cosine_reward(completions, ground_truth, **kwargs):
    """Cosine-scaled accuracy reward with numeric tolerance."""
    max_len = GRPO_MAX_COMPLETION
    rewards = []
    for i, (completion, gt) in enumerate(zip(completions, ground_truth)):
        content = _get_content(completion)
        extracted = _extract_boxed(content)
        clen = len(content)
        progress = min(clen / max(max_len, 1), 1.0)
        cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        if extracted is not None and _answers_match(extracted, gt):
            reward = 0.1 + 0.9 * cos_scale  # [1.0, 0.1]
        elif extracted is not None:
            reward = -0.1 - 0.9 * (1.0 - cos_scale)  # [-0.1, -1.0]
        else:
            reward = -0.5 * progress
        rewards.append(reward)
    return rewards
```

The format reward is trivial: 1.0 if a `\boxed{...}` is present, else 0.0.

The "reasoning quality" reward is the most novel piece — it scores a
completion for step-marker keywords, math symbols, pre-boxed prose length,
and word-uniqueness:
```python
def reasoning_quality_reward(completions, **kwargs):
    """Rewards structured reasoning (step markers, examples cited)."""
    rewards = []
    for c in completions:
        content = _get_content(c)
        score = 0.0
        if re.search(r'step\s*\d|example\s*\d|first|then|therefore|thus', content.lower()):
            score += 0.3
        if re.search(r'[=×÷+\-*/^]', content):
            score += 0.2
        boxed_pos = content.find('\\boxed')
        if boxed_pos > 100:
            score += 0.2
        words = content.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.3:
                score += 0.3
        rewards.append(score)
    return rewards
```

GRPO config:
```python
GRPO_LR = 5e-6
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_COMPLETION = 1024
GRPO_TEMPERATURE = 0.8
GRPO_BETA = 0.04
...
grpo_kwargs = dict(
    output_dir=OUTPUT_DIR + '_grpo',
    num_train_epochs=GRPO_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=GRPO_LR,
    lr_scheduler_type='cosine',
    warmup_steps=5,
    num_generations=GRPO_NUM_GENERATIONS,
    generation_batch_size=GRPO_NUM_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION,
    temperature=GRPO_TEMPERATURE,
    beta=GRPO_BETA,
    ...
    max_grad_norm=0.1,
)
```

SFT side uses LoRA r=32/α=32/dropout=0.0 on
`r'.*\.(in_proj|out_proj|up_proj|down_proj)$'` (a regex pattern that
catches the Mamba projections + MLP up/down but **omits** `q,k,v,o,lm_head`
— a substantial regression vs the recipe), lr=1e-4 cosine with
warmup_ratio=0.05, effective batch=8, `max_length=7680`,
`neftune_noise_alpha=5.0` (NEFTune noise — another deviation from the
recipe).

### Baseline comparison
The notebook explicitly bakes in evidence from prior iterations:
- `USE_GRPO = False` with the comment "GRPO not helping yet — model needs
  more SFT first" — this is the author's own self-report from v1/v2 runs.
- `SFT_EPOCHS = 1` and `SFT_LR = 1e-4` with comments "Proven: 2 epochs"
  and "Proven by konbu17" — i.e. these defaults are choices the author
  has run before and is leaning on.

So the **headline finding from the v3 iteration is: GRPO did not improve
LB after SFT, and was disabled.**

### Reported LB lift
**None claimed.** The notebook is "v3" and has not yet produced a positive
GRPO result. The only retained ingredient is SFT on konbu17's CoT data
with NEFTune noise — and neither of those is being highlighted as the
novel contribution.

### Hyperparameters changed
For GRPO (when enabled):
- `GRPO_LR=5e-6` (40× smaller than SFT lr).
- `GRPO_TEMPERATURE=0.8` (eval is greedy, so this is a *training-time*
  exploration knob only).
- `GRPO_BETA=0.04` (KL coefficient).
- `max_grad_norm=0.1` — extremely aggressive clipping (the recipe runs
  with clipping effectively off; GRPO conventionally clips around 1.0).
- `num_generations=4` and `generation_batch_size=4` — minimum viable
  group size; smaller groups give noisier advantages.

For SFT:
- `neftune_noise_alpha=5.0` — NEFTune noise injection on the input
  embeddings during training. Not present in the recipe.
- `max_length=7680` — explicit alignment with the eval `max_tokens=7680`.
- `target_modules` regex omits attention projections and `lm_head` — this
  is a clear regression vs. the recipe's 9-module list.

### Risks / failure modes
- **GRPO is disabled.** Even if the multi-reward design were excellent,
  no submission has used it.
- **`reasoning_quality_reward` is a Goodhart magnet.** A reward for "has
  the word 'step' or 'therefore' in the output" and "has at least one of
  `[=×÷+-*/^]`" can be trivially hacked by a model that's already
  reasoning to instead just sprinkle in those tokens. The "word-uniqueness
  > 0.3" check is similarly easy to satisfy by adding nonsense. The
  cosine length-scaling on accuracy also rewards *shorter* correct
  answers (good for the boxed-only-eval) but punishes longer correct
  ones — for a CoT model trained to think step by step, this works
  *against* the SFT objective.
- **GRPO on a 30B MoE with `generation_batch_size=4` and
  `num_generations=4`** is roughly the slowest plausible setting; the
  v3 ablation of `MAX_STEPS_GRPO=30` is way too few for stable
  policy-gradient learning even if the rewards were well-calibrated.
- **SFT target-modules regex drops attention QKV/O and lm_head**, which
  is a known regression from the 0.85 recipe. The author probably
  intended `(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj|lm_head)`
  but truncated mid-thought.
- **NEFTune at α=5.0** is on the high end and on a hybrid Mamba+MoE model
  with no prior NEFTune literature — its interaction with the recipe's
  no-grad-clip / lr=2e-4 is untested.
- **`max_grad_norm=0.1` in GRPO** with `beta=0.04` and a sparse correctness
  signal is likely to under-update.

### Verdict
**Skip GRPO; keep one ingredient.** The author's own v3 note ("GRPO not
helping yet") is the clearest signal here. The multi-reward design is
clever but the "reasoning quality" reward is a soft constraint that
rewards stylistic markers — exactly the kind of reward that drifts the
policy off the SFT distribution without lifting answer accuracy. The one
ingredient worth investigating in isolation is **NEFTune
(`neftune_noise_alpha=5.0`) on top of the recipe SFT** — it has independent
literature evidence on instruction-following SFT and could be a 0.1–0.3pt
lift; the rest of the notebook (broken target_modules, disabled GRPO,
sub-baseline LR/grad-clip) is noise. Even the NEFTune claim is unconfirmed
on this competition.

---

## Synthesis — ranked by expected accuracy lift on this competition

Ranking is by *expected* lift if implemented well on top of the 0.85 recipe,
*conditioned on the evidence in these notebooks*. I weight LB-confirmed
deltas heavily and discount "clever-looking" ideas without measurement.

| Rank | Notebook | Expected lift | Confidence | Reason |
|---|---|---|---|---|
| **1** | waterjoe — mask loss | **+1–3 pts** if your baseline forgot it, ~0 if you already had it | High (in general); zero in this notebook | The one-liner `completion_only_loss=True` is the right default for SFT. huikang and mayukh18 already do this via other paths; if any custom training loop omitted it, this is the single biggest free win. The notebook itself proves nothing, but the literature is clear. |
| **2** | pkuszboi — Muon optimizer | **±0–1 pts**, plausibly +0.5 with proper LR sweep | Low-medium | Muon has real momentum in the broader community and the LoRA-only application is natural. But the notebook doesn't measure it head-to-head and runs at a sub-recipe effective batch. Worth a 1-day ablation against AdamW with lr ∈ {1e-3, 2e-3, 4e-3} on the recipe data. If neutral, take the wall-clock win (~25%); if it lifts, take both. |
| **3** | amanatar — NEFTune piece only | **+0.0–0.3 pts** | Low | The author's own evidence says GRPO doesn't help. The only salvage is NEFTune (`alpha=5.0`) on the SFT stage, which has independent literature support on instruction-following. Worth a single A/B; everything else in the notebook (broken target_modules regex, multi-reward GRPO) is worse than baseline. |
| **4** | newduck — soft-balanced sampling | **~0 pts** | Medium-high | Class-balanced sampling is fine in principle, but dgxchen's stratified-by-type sampler already covers this with the *correct* puzzle labels. The 4-category regex is a worse proxy than the existing `type` column in the CoT corpus, and the notebook doesn't measure its own `alpha`. The 600-row no-CoT setup is a sub-baseline. |
| **5** | hammadfarooq470 — "Think Twice" | **N/A** | High | The notebook does not contain a self-correcting reasoning method. It's an SVD rank-cap utility for packaging Tinker adapters that exceed rank 32 after fused-projection merging — a narrow submission-time fix that other notebooks handle better. |

### Cross-cutting takeaways

1. **Three of five notebooks have zero LB confirmation of their own
   "novel" claim.** Mask-loss, soft-balance, and Think-Twice are each
   pipeline demos with no measured delta. Muon ties but doesn't beat the
   baseline. Only amanatar v3 has authored evidence about its centrepiece
   (GRPO) and that evidence is *negative*.

2. **The single most consequential observation across the five is the
   amanatar `USE_GRPO=False` comment.** If a multi-iteration experimenter
   with a well-designed reward stack can't get GRPO to lift LB on this
   competition with this base model, the prior on "GRPO will help" should
   shift sharply downward. The likely reason: greedy eval at temp=0 means
   policy-gradient training with sampling temperature 0.8 is optimising a
   distribution the eval never sees, while the SFT distribution is exactly
   what eval samples from.

3. **Optimizer swaps (Muon) and reward stacks (GRPO) both look more
   attractive in the abstract than in practice on this competition.** The
   data-side gap (CoT teacher quality, per-puzzle solver coverage) is much
   wider than the optimization-side gap. A 5-hour Muon run that ties the
   AdamW baseline is fine but won't push past 0.86; a 5-hour CoT
   redistillation against a stronger teacher very plausibly will.

4. **Mask-loss is the only "free" item.** Anyone whose custom SFT loop
   does not mask loss to the assistant turn should fix that today — either
   via `completion_only_loss=True`, `train_on_responses_only`, or a
   pre-tokenized weight mask. Everything else here requires a
   controlled experiment to be worth the wall-clock.

### Top 2 to try

- **Mask loss** (`completion_only_loss=True` or equivalent) — verify your
  pipeline does it; if not, add it. Expected lift: large if missing, zero
  if already present.
- **Muon optimizer** on LoRA matmul params only, with an LR sweep against
  the recipe — expected ~parity at minimum, possible small lift, real
  wall-clock win.

### Bottom 2 to skip

- **"Think Twice" notebook** — the title doesn't match the content; the
  SVD rank-cap utility is narrow and better covered elsewhere.
- **Soft-balanced sampling at α=0.5** — supersedes nothing the recipe's
  StratifiedSFTTrainer already does, uses a worse classifier, never
  measures its own knob.
