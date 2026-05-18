# Local CV Harness and Adapter Fusion / Rank Compression

These two notebooks address the two biggest leverage points in the next four weeks: (1) a **local validation harness that mirrors the host metric byte-for-byte**, so we can iterate without burning our 5-per-day submission budget, and (2) an **adapter fusion + SVD rank-compression pipeline** that takes N independently trained rank-32 LoRAs and produces a single rank-32 adapter — the obvious lever if/when we have multiple models that win on different problem categories.

The first one (`jiazhuang/nemotron-local-cv`) is, on close reading, an **exact** copy of the host's `score()` function from `metric.py` (right down to the docstring), pointed at a custom 950-row validation split. That is the strongest possible guarantee of faithfulness: every change to the host metric reaches this harness verbatim, because it *is* the host metric — only the test data and the path resolution differ. The harness is directly liftable into our pipeline with minimal modification.

The second one (`rauffauzanrambe/lora-adapter-fusion-and-rank-compression-pipeline`) is more speculative: it monkey-patches `tinker_cookbook.weights._adapter._merge_fused_projections` to do SVD rank-compression after fusing per-component (Mamba `in_proj`, MoE fused `gate_up_proj`) LoRA chunks. There is **no reported LB result**, no ablation, and the notebook description was clearly LLM-written. It only really makes sense if we end up training multiple specialist adapters and want to fuse them into a single rank-32 submission.

---

## Local CV Harness (`jiazhuang/nemotron-local-cv`)

**Source**: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/jiazhuang__nemotron-local-cv/nemotron-local-cv.ipynb`

### Validation split

- Loads a **950-row** held-out set from a private Kaggle dataset (`jiazhuang/nemotron-val-set-950/val_set_950.jsonl`).
- No stratification logic is shown in the notebook — the split was constructed offline and shipped as a dataset; we cannot directly inspect how it was sampled. The notebook does later do `merge_df.groupby('label').is_correct.mean()`, which confirms each row has a `label` column (problem category), so they probably stratified by category.
- Columns: `id`, `prompt`, `answer`, `label`. Saved as `test.csv` before being consumed (the metric reads `DATA_PATH / 'test.csv'`).
- Seed: not exposed. The split is fixed (it's a static jsonl).

**Action item for us**: we need to construct an analogous split from the train data (`train.csv` shipped with the competition). The right move is a stratified-by-`label` sample, probably 400–800 rows depending on how much wall-clock we can spend.

### Mirroring the host metric

This is the most important finding: **the `score()` function in this notebook is byte-identical to the host's `metric.py` in every load-bearing detail** (prompt assembly, chat template, sampling params, answer extraction, verification). The only differences are (a) it expects `test.csv` and `solution.csv` to live in `DATA_PATH` instead of `/kaggle/input/competition_evaluation`, and (b) it passes through `debug=True` to dump raw outputs.

#### Environment / page-cache warmup

```python
import subprocess, sys
commands = [
    'uv pip uninstall torch torchvision torchaudio',
    'tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp',
    'chmod +x /tmp/triton/backends/nvidia/bin/ptxas',
    'chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell',
]
for cmd in commands:
    print(f'Running: {cmd}')
    subprocess.run(cmd, shell=True, check=True)
sys.path.insert(0, '/tmp')
```

This is the same Triton/ptxas patch the host uses. It is irrelevant outside Kaggle's runner but we should mirror the equivalent setup in our own environment (mostly: make sure vLLM finds a compatible ptxas).

```python
def cache_model(path, exts=('.bin', '.pt', '.safetensors'),
                num_workers=None, chunk_mb=256):
    """Pre-read model weight files into the OS page cache to speed up later loads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def warmup_file(fpath):
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    path = Path(path)
    files = ([p for p in path.rglob('*') if p.is_file() and str(p).endswith(exts)]
             if path.is_dir() else [path] if path.exists() else [])
    if not files:
        return 0
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for fut in as_completed(futures):
            _, n = fut.result()
            total_bytes += n
    return total_bytes
```

Used as `cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)` right before vLLM init. Pure latency optimization, not load-bearing for correctness.

#### Prompt construction (CRITICAL — matches host exactly)

```python
tokenizer = llm.get_tokenizer()
prompts = []
for item in test_df.itertuples(index=False):
    user_content = (
        item.prompt
        + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
    )
    try:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        prompt = user_content
    prompts.append(prompt)
```

Note vs. the version documented in `01_baseline_and_submission.md`: the suffix here uses **backticks around `\boxed{}` and `\boxed{your answer}`**:

```
\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

`01_baseline_and_submission.md` notes the literal suffix is `\nPlease put your final answer inside \boxed{}. For example: \boxed{your answer}` (no backticks). **One of these is wrong.** Given that this notebook's `score()` is otherwise a verbatim copy of `metric.py` (the docstring is the host's docstring), the backtick version is the more likely truth — but we should verify against the host's `metric.py` directly before trusting either. If our submissions to date have been training with the no-backtick suffix and the host actually uses backticks, we have a real distribution-shift bug.

#### vLLM init and sampling

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=str(MODEL_PATH),
    tensor_parallel_size=1,
    max_num_seqs=max_num_seqs,
    gpu_memory_utilization=gpu_memory_utilization,
    dtype='auto',
    max_model_len=max_model_len,
    trust_remote_code=True,
    enable_lora=True,
    max_lora_rank=max_lora_rank,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
)
```

Driven by the `score()` call at the bottom of the notebook:

```python
val_score = score(
    solution=pd.read_csv('./test.csv'),
    submission=pd.DataFrame([{'prediction': './'}]),
    row_id_column_name='id',
    max_lora_rank=32,
    max_tokens=7680,
    top_p=1.0,
    temperature=0.0,
    max_num_seqs=64,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    debug=True,
)
```

These match the documented host config (`max_model_len=8192`, `max_tokens=7680`, `temperature=0.0`, `top_p=1.0`, `max_lora_rank=32`). **Caveat**: the `score()` function's *default* args are `max_tokens=3584`, `temperature=1.0`, `max_model_len=4096` — which are wrong if you ever call it without override. Always pass the correct kwargs.

Generation call:

```python
outputs = llm.generate(
    prompts,
    sampling_params=sampling_params,
    lora_request=LoRARequest('adapter', 1, lora_path),
)
```

#### Answer extraction (verbatim from host)

```python
def extract_final_answer(text):
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'
```

Important quirks:
- The boxed regex is `\\boxed\{([^}]*)(?:\}|$)` — it does **not** support nested braces. `\boxed{\frac{1}{2}}` will extract `\frac{1` (stops at the first `}`). This is a known host metric weakness; we should train the model to avoid emitting nested-brace answers.
- The regex picks the **last** non-empty match, not the first, so if the model double-emits, we use the final one.
- The numeric fallback uses `-?\d+(?:\.\d+)?` — no scientific notation, no fractions, no commas. A model output of `1,234` would extract as `234`. A `1e5` would extract as `1`.

#### Verification (verbatim from host)

```python
def verify(stored_answer, predicted):
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
```

Two non-obvious behaviors:
- **Binary strings** (regex `[01]+` matches the entire ground truth) bypass the numeric path entirely — so `"11011"` will not float-coerce to 11011.0 and compare-equal to `"011011"`. This is specifically for the bit-string subset of the dataset.
- **Numeric tolerance**: `rel_tol=1e-2, abs_tol=1e-5`. That's pretty loose — `42` matches anything in `[41.58, 42.42]`. Useful for problems where the model's arithmetic drifts in the last digit.

#### Full score() function

```python
def score(solution, submission, row_id_column_name,
          max_lora_rank=32, max_tokens=3584, top_p=1.0, temperature=1.0,
          max_num_seqs=128, gpu_memory_utilization=0.85, max_model_len=4096,
          debug=False):
    lora_path = submission['prediction'].iloc[0]
    test_df = pd.read_csv(DATA_PATH / 'test.csv', index_col=None)
    row_id_col = str(test_df.columns.to_list()[0])
    test_df = test_df[test_df[row_id_col].isin(solution[row_id_column_name])]

    submission = generate_predictions(
        test_df=test_df,
        lora_path=lora_path,
        row_id_col=row_id_column_name,
        max_lora_rank=max_lora_rank,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        debug=debug,
    )

    dataset = solution.merge(submission, on=row_id_column_name)
    num_correct = 0
    for item in dataset.itertuples(index=False):
        ground_truth = item.answer
        extracted_answer = item.prediction
        if verify(str(ground_truth), str(extracted_answer)):
            num_correct += 1
    return float(num_correct / len(solution))
```

#### Per-category breakdown

After scoring, the notebook does:

```python
merge_df['is_correct'] = merge_df.apply(
    lambda r: verify(r.answer, r.extracted_prediction),
    axis=1,
)
merge_df.groupby('label').is_correct.mean()
```

This is gold — per-label accuracy on a fixed 950-row set lets us see *which problem categories* a change helps or hurts, not just the aggregate.

### Reported correlation with public LB

**Not explicitly reported in the notebook.** No table, no scatter plot, no "this CV = X, this LB = Y" callout. The notebook is a tooling release, not an ablation study. We will have to validate the LB correlation ourselves on our first few submissions.

That said: because this *is* the host metric verbatim, the only sources of CV→LB gap are:
1. Distribution shift between the 950-row val set and the hidden test set.
2. Stochasticity (none — temperature is 0, but vLLM kernel non-determinism across machines can still introduce ~0.5pp jitter).
3. The val set possibly overlapping with the training data (depends on how `jiazhuang` built it).

The 950-row size gives a standard error of roughly `sqrt(0.5 * 0.5 / 950) ≈ 1.6pp` at 50% accuracy, ~1.1pp at 80%. That's tight enough to detect improvements of 2pp or more confidently — exactly the granularity we need.

### Compute requirement

- Single GPU (`tensor_parallel_size=1`, `CUDA_VISIBLE_DEVICES=0`).
- `gpu_memory_utilization=0.85`, `max_num_seqs=64`, `max_model_len=8192`.
- The notebook was clearly run on Kaggle's L4×4 / H100 / similar accelerator. Wall-clock per CV run is not printed in the cells, but with 950 prompts × up to 7680 generated tokens and `max_num_seqs=64`, a reasonable estimate is **30–90 minutes on an H100 / 1–2 hours on an L4 / longer on a T4 (which probably won't fit)**.
- Implication: we can afford **roughly 1 CV run per training experiment** if we have a dedicated H100, fewer if we share compute. We cannot afford CV per epoch during training — only as a post-training validation step.
- Prefix caching is enabled, which helps if multiple prompts share a system prefix. Chunked prefill is on. These are reasonable defaults; no further tuning needed.

### Reusable code blocks to lift directly

All from `cell-11` of `nemotron-local-cv.ipynb`:
- `cache_model(...)` — verbatim, optional but worth keeping.
- `extract_final_answer(...)` — verbatim, **mandatory** (must match host).
- `verify(...)` — verbatim, **mandatory** (must match host).
- `generate_predictions(...)` — keep the prompt-building loop and the chat-template call verbatim. The vLLM init args can be exposed as ours.
- `score(...)` — adapt freely, but keep the loop body intact.

Suggested package layout for our repo:

```
src/local_cv/
  __init__.py
  harness.py        # cache_model, extract_final_answer, verify, score
  prompts.py        # build_user_prompt(prompt_text) -> str; build_chat_prompt(tokenizer, user) -> str
  split.py          # build_stratified_val(train_df, n=600, seed=42) -> (train, val)
```

---

## Adapter Fusion & SVD Rank Compression (`rauffauzanrambe/...`)

**Source**: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/rauffauzanrambe__lora-adapter-fusion-and-rank-compression-pipeline/lora-adapter-fusion-and-rank-compression-pipeline.ipynb`

### What problem fusion solves (in this notebook specifically)

Despite the project description's claim about "merging multiple LoRAs," the notebook does **not actually fuse multiple independently trained adapters**. It does something subtler and Nemotron-specific:

A Tinker-trained adapter for Nemotron has separate LoRA components for each sub-projection of a *fused* base-model linear layer. For example, Nemotron's MoE expert MLP combines `gate_proj` and `up_proj` into a single `gate_up_proj` weight matrix in the on-disk checkpoint. Tinker tracks LoRA on `gate_proj` and `up_proj` *separately* (`rank=r_gate + r_up`). When the cookbook's `build_lora_adapter()` converts this to a PEFT-compatible adapter, it has to merge those two component-LoRAs into a single LoRA on the fused `gate_up_proj`. The default merge **stacks them** — concatenating B matrices along the row axis and A matrices along the rank axis — producing a fused LoRA of rank `r_gate + r_up` (i.e. **double** the original).

Same story for the Mamba `in_proj` layer, which fuses many components together. The default behavior can blow past `max_lora_rank=32`.

This notebook patches `tinker_cookbook.weights._adapter._merge_fused_projections` to **SVD-compress** the stacked LoRA back down to rank 32 before writing it. So it's not "fusing N adapters into one" — it's "fusing the multi-component LoRAs *within* one adapter back into rank-32 fused-projection LoRAs."

That said, the SVD trick is exactly the math you would use if you were combining N independently trained rank-32 LoRAs into a single rank-32 LoRA (sum their `BA` products, then SVD-truncate to rank 32) — so the code is reusable for that purpose.

### Fusion algorithm / math

For each component `(lora_A, lora_B)` belonging to a fused projection, the patch:

1. Concatenates all `lora_A` matrices along the rank axis: `merged_lora_A = cat([A_i], dim=0)` (shape `[sum(r_i), in_dim]`).
2. Builds a block-diagonal `merged_lora_B` of shape `[fused_out_dim, sum(r_i)]`, where each component's B occupies its own row-band and rank-band (i.e. component `i`'s B sits in rows `[row_start_i, row_end_i)` and cols `[rank_offset_i, rank_offset_i + r_i)`, zeros elsewhere).
3. If the total `merged_rank > 32`, calls `_compress_lora_pair_to_rank(B, A, 32)`:
   - Compute `delta = B @ A` (full `[out_dim, in_dim]` matrix).
   - `U, S, Vh = torch.linalg.svd(delta, full_matrices=False)`.
   - Truncate to rank 32: `U[:, :32]`, `S[:32]`, `Vh[:32, :]`.
   - Split singular values via `sqrt`: `B_new = U * sqrt(S)`, `A_new = sqrt(S) * Vh` (i.e. balance the spectral mass between A and B).

This is **plain SVD compression of the BA product** — no TIES, no DARE, no task arithmetic, no DARE-TIES, no magnitude pruning. Just "the best rank-32 approximation in Frobenius norm of the full delta matrix."

It's task-arithmetic-friendly: if you want to fuse N independently trained adapters with weights `w_i`, you would replace step 1–2 with `delta = sum(w_i * B_i @ A_i)` and feed that into SVD.

### Rank compression details

The key trick to "avoid losing capacity" is **honest acknowledgement that you do lose capacity** — SVD truncation is optimal in Frobenius norm, but the singular values past rank 32 are discarded. There is no clever trick here; the assumption is that the original component LoRAs were trained with low-effective-rank gradients, so the tail singular values are small and the truncation is near-lossless. For an actual N-adapter fusion, this assumption is much weaker — N adapters trained on different tasks will typically span more than 32 directions.

### Verbatim code

```python
import torch
import shutil
import tinker_cookbook.weights._adapter as A
from tinker_cookbook import weights

FORCED_FUSED_RANK = 32


def _compress_lora_pair_to_rank(B: torch.Tensor, A_mat: torch.Tensor, rank: int):
    # Compute delta = B @ A
    delta = B.float() @ A_mat.float()

    # Perform SVD decomposition
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    # Truncate to desired rank
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    # Reconstruct LoRA matrices (split singular values via sqrt for balance)
    sroot = torch.sqrt(S)
    B_new = U * sroot.unsqueeze(0)
    A_new = sroot.unsqueeze(1) * Vh

    return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous()


def patched_merge_fused_projections(
    fused_model_key: str,
    adapter_layer_prefix: str,
    components,
    model_state_shapes,
    peft_weights,
    target_modules,
    profile,
) -> int:
    fused_out_dim = model_state_shapes[fused_model_key][0]
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]

    # Find component order from the model profile (e.g. ["gate_proj", "up_proj"]
    # for a fused gate_up_proj)
    component_order = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            component_order = comps
            break
    assert component_order is not None, "Component order not found"

    comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

    lora_A_parts = []
    comp_slices = []
    merged_rank = 0
    row_offset = 0

    # Walk components in the declared order; record where each component's B
    # rows live in the fused output and how many ranks it contributes.
    for comp_name in component_order:
        if comp_name not in comp_by_name:
            raise RuntimeError(f"Missing component {comp_name}")
        lora_A, lora_B = comp_by_name[comp_name]
        r = lora_A.shape[0]
        out_dim = lora_B.shape[0]
        lora_A_parts.append(lora_A)
        comp_slices.append((row_offset, row_offset + out_dim, r))
        row_offset += out_dim
        merged_rank += r

    # Concatenate A matrices along the rank axis.
    merged_lora_A = torch.cat(lora_A_parts, dim=0)

    # Build block-diagonal B: each component sits in its own row-band and
    # rank-band, with zeros elsewhere.
    merged_lora_B = torch.zeros(
        fused_out_dim,
        merged_rank,
        dtype=merged_lora_A.dtype,
        device=merged_lora_A.device,
    )

    rank_offset = 0
    for i, (row_start, row_end, r) in enumerate(comp_slices):
        _, lora_B = comp_by_name[component_order[i]]
        merged_lora_B[row_start:row_end, rank_offset:rank_offset + r] = lora_B
        rank_offset += r

    # SVD-compress if we exceed the rank cap.
    final_rank = merged_rank
    if merged_rank > FORCED_FUSED_RANK:
        merged_lora_B, merged_lora_A = _compress_lora_pair_to_rank(
            merged_lora_B,
            merged_lora_A,
            FORCED_FUSED_RANK,
        )
        final_rank = FORCED_FUSED_RANK

    # Write the compressed pair back into the PEFT weights dict.
    peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"
    A._add_peft_weight(
        peft_target_key,
        merged_lora_A,
        merged_lora_B,
        peft_weights,
        target_modules,
    )
    return final_rank


# Install the patch.
A._merge_fused_projections = patched_merge_fused_projections

# Build the PEFT adapter from a Tinker checkpoint.
weights.build_lora_adapter(
    base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    adapter_path="/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
    output_path="/kaggle/working/nemotron-adapter-ready-to-submit",
)

shutil.make_archive(
    '/kaggle/working/submission',
    'zip',
    '/kaggle/working/nemotron-adapter-ready-to-submit',
)
```

### Nemotron-specific quirks handled

- **Fused projections** (Mamba `in_proj`, MoE `gate_up_proj`) are looked up via `profile.fused_projection_map` to recover the declared sub-component order. The order matters because the B rows have to line up with the on-disk weight layout.
- **`model → backbone` rename** is NOT handled here. The notebook only patches `_merge_fused_projections`; the rest of `build_lora_adapter()` (key renaming, target-module list filtering, MoE per-expert merging) runs unmodified. So this patch composes with the existing tinker-cookbook → PEFT translation, it does not replace it.
- **MoE per-expert LoRAs** are presumably handled in a sibling cookbook function (`_merge_moe_experts` or similar) that the patch does not touch.

### Reported LB results

**None.** No accuracy number is reported in the notebook at all. No "before patch: X, after patch: Y" comparison. No ablation between different `FORCED_FUSED_RANK` values. No comparison against the unpatched `build_lora_adapter`.

This is a serious red flag — if the patch worked, you'd expect the author to flex the number. The reasonable interpretation is: this patch lets the author **submit at all** (because the unpatched output exceeded rank 32 and the metric refuses it), but they have no claim about it improving accuracy.

### Constraint check

- `FORCED_FUSED_RANK = 32` — yes, the output fits the host's `max_lora_rank=32`.
- The patch only acts on fused projections. Non-fused LoRAs (e.g. attention `q_proj`, individual MoE layers) pass through unchanged. If those are already rank 32, the final adapter is uniformly rank 32. If any of them is rank > 32, the submission will still fail.

---

## Recommendation

### (a) Local CV harness — **copy this immediately**

Concrete plan:

1. **Lift `score()`, `generate_predictions()`, `extract_final_answer()`, `verify()`, `cache_model()` verbatim** from cell 11 of `nemotron-local-cv.ipynb` into `src/local_cv/harness.py`. Do not paraphrase, do not refactor, do not "improve" — these need to byte-match the host metric.
2. **Build our own 600-row stratified val split** from `train.csv` (stratify by `label`, fixed seed). 600 is the sweet spot between statistical power (~2pp SE at 80% accuracy) and per-run cost (~30 min on H100). Save as `data/val_600.jsonl` and check into the repo.
3. **Wire `score()` to point at our adapter directory** instead of a Kaggle path. Pass `max_lora_rank=32, max_tokens=7680, top_p=1.0, temperature=0.0, max_model_len=8192` explicitly every time (the function defaults are wrong).
4. **Validate suffix string against the actual host `metric.py`** before trusting either docs entry. Read `/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script/metric.py` (or whatever path the host ships) and confirm whether the boxed-instruction suffix uses backticks. If it does, our training-time prompts need to match (otherwise we have a train/eval prompt mismatch).
5. **Add per-`label` breakdown** as a standard output of every CV run — that's where the actionable signal lives.
6. **Treat one CV run as one "submission unit"** in our experiment budget: if CV says +X pp on the 600-row val, only then spend a real submission to confirm on the public LB.

Does the harness genuinely mirror the host metric? **Yes**, with two caveats:
- The boxed-instruction suffix string needs to be verified against `metric.py` (this notebook uses backticks; `01_baseline_and_submission.md` says no backticks).
- vLLM kernel non-determinism across hardware (L4 vs. H100 vs. T4) can introduce ~0.5pp jitter run-to-run even at temp=0. Not a correctness issue, just noise.

### (b) Adapter fusion — **defer; revisit only if we end up training multiple specialists**

Concrete plan:

1. **Do not adopt this right now.** Our current pipeline trains one adapter end-to-end; we do not have multiple LoRAs to fuse.
2. **Keep the `_compress_lora_pair_to_rank` function on file** (literally 10 lines of SVD). It is the reusable kernel for any future fusion experiment — multi-task fusion, multi-seed averaging, ensembling specialist adapters.
3. **If we adopt the tinker-cookbook training path** (we have not, per the existing docs — we use PEFT directly), then we likely **do need** the `patched_merge_fused_projections` patch just to produce a rank-≤32 submission. In that case, lift it verbatim.
4. **If we ever want to ensemble** (e.g. one math adapter + one logic adapter), the recipe would be: compute `delta_i = w_i * B_i @ A_i` for each adapter, sum them, SVD-truncate to rank 32 with `_compress_lora_pair_to_rank`. Expected to lose some specialist capability but possibly worth ~1–2pp on aggregate. Untested in this notebook.

Is adapter fusion recommended for our use case? **No, not now.** The notebook provides zero empirical evidence that fusion helps, and we don't have multiple adapters to fuse anyway. Revisit only if (a) we switch to tinker-cookbook training and need the patch to satisfy the rank cap, or (b) we have trained 2+ promising adapters and want to combine them into a single submission.
