# Baseline and Submission Packaging Notebooks

These six notebooks define and exercise the submission contract for the NVIDIA Nemotron-3-Nano-30B reasoning challenge. Together they show: (a) the canonical PEFT `LoraConfig` that produces a `submission.zip` Kaggle's metric will accept, (b) how to validate that zip locally against the train split with vLLM + LoRA, and (c) several recipes for taking a Tinker-trained adapter (which uses fused projections and per-expert merged weights) and converting it into a PEFT-compatible adapter that the host's `metric` package can load. The notebooks also reveal load-bearing implementation details: the chat-template prompt suffix, page-cache warmup, the `ptxas` Triton patch, the `nemotron-3-nano-30b-a3b-bf16` kagglehub path, and the exact target-module set expected by the base model's mixed Mamba/MoE architecture.

---

## 1. `ryanholbrook/nvidia-nemotron-submission-demo` (host's official demo, 2246 votes)

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/ryanholbrook__nvidia-nemotron-submission-demo/nvidia-nemotron-submission-demo.ipynb`

### Approach
Bare-minimum reference: load base model in bf16, attach a fresh untrained LoRA, save it, zip it. **No actual training, no inference** — defines the file layout the metric expects and the exact `LoraConfig` recipe.

### Submission format
- Save with `model.save_pretrained("/kaggle/working")` (writes `adapter_config.json`, `adapter_model.safetensors`, etc. directly into the working dir).
- Zip with the shell command `zip -m submission.zip *` — the `-m` flag moves files into the archive (deletes originals), so `submission.zip` is the only artifact left. The archive is **flat** (no nested folder).

### Adapter source / config
Trained inline from the bf16 base. Exact config:
```python
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
LORA_RANK = 32  # Can be set to a maximum of 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
```
Key signal: `target_modules` is a **regex** that matches only `in_proj`, `out_proj`, `up_proj`, `down_proj` — i.e. the Mamba SSM projections (`in_proj`/`out_proj`) and the MoE expert MLP projections (`up_proj`/`down_proj`). The attention `q/k/v` projections and `gate_proj` are **not** targeted by the demo. `lora_alpha=16` with `r=32` gives a scaling factor of 0.5.

### Special tricks
- Adds a CUTLASS package via `site.addsitedir(...)` pointing at `/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/` before importing torch/mamba_ssm. Required because the bf16 Mamba forward pass depends on a CUTLASS DSL build that isn't in the default Kaggle env.
- Imports `mamba_ssm` explicitly to confirm the kernel is present.

### Inference / validation setup
None. The notebook does not invoke vLLM.

### Gotchas
- The comment `# Can be set to a maximum of 32` next to `LORA_RANK` confirms the rank ceiling enforced by the metric (`max_lora_rank=32` in scorer).
- `device_map="auto"` is used directly with bf16 — there is no quantization (no bitsandbytes, no AWQ).

---

## 2. `huikang/adapter-validation-notebook`

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/huikang__adapter-validation-notebook/adapter-validation-notebook.ipynb`

### Approach
Standalone **validation harness**. Inputs an existing `submission.zip`, extracts it, runs vLLM + LoRA on the train split, prints per-category accuracy and minimum log-probability, and re-zips a clean `submission.zip` as output. Designed to be a hold-out probe before submission.

### Submission format
- Reads `submission.zip` from an arbitrary Kaggle input path:
  `/kaggle/input/notebooks/huikang/tinker-submission-notebook/submission.zip` (i.e. it consumes the output of notebook #3).
- After extracting and validating, re-zips files whose names contain `"adapter"` into a fresh flat zip (and deletes them from the working dir). This is more selective than the demo's `zip *`.

### Adapter source / config
The adapter comes from another notebook's `submission.zip`. Not modified here; only inspected (`adapter_config.json` is printed).

### Inference / validation setup
Exhaustive. Reproduces the metric's environment closely:

**Env prep** (host's `metric` utility script, taken from `/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script`):
```bash
uv pip uninstall torch torchvision torchaudio
tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp
chmod +x /tmp/triton/backends/nvidia/bin/ptxas
chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell
```
Then `sys.path.insert(0, "/tmp")` so the patched triton is picked up.

**Page-cache warmup** (`cache_model`): pre-reads all `.bin`/`.pt`/`.safetensors` files under the model dir using 16 threads × 1024 MB chunks. Significantly reduces vLLM startup time.

**Required env vars before importing vLLM**:
```python
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"
```

**vLLM init (matches metric exactly)**:
```python
max_model_len = 8192
max_lora_rank = 32
max_tokens = 7680
top_p = 1.0
temperature = 0.0
max_num_seqs = 64
gpu_memory_utilization = 0.85

llm = LLM(
    model=str(MODEL_PATH), tensor_parallel_size=1,
    max_num_seqs=max_num_seqs,
    gpu_memory_utilization=0.85, dtype="auto",
    max_model_len=8192, trust_remote_code=True,
    enable_lora=True, max_lora_rank=32,
    enable_prefix_caching=True, enable_chunked_prefill=True,
)
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=7680, logprobs=1,
)
```
Note: validation uses `logprobs=1` so it can record the min-logprob per generation (useful for sample-difficulty triage). The metric itself does **not** request logprobs.

**Prompt template** — uses the tokenizer's chat template with `enable_thinking=True`:
```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": problem_text}],
    tokenize=False, add_generation_prompt=True, enable_thinking=True,
)
```
**Important divergence vs the host's metric**: the validation notebook passes the raw `problem_text`; the host's metric appends the boxed-instruction suffix (verified verbatim against host kernel `metric/nvidia-nemotron-metric` pulled 2026-05-18 — **note the backticks** around `\boxed{}` and `\boxed{your answer}`): `"\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`. So the validation prompt in cell 16 is *strictly weaker* than the real metric prompt (omits the boxed instruction). To exactly mirror the metric, append that suffix verbatim with backticks.

**LoRA load**:
```python
outputs = llm.generate(prompts, sampling_params=sampling_params,
                       lora_request=LoRARequest("adapter", 1, lora_path))
```
`lora_path = "/kaggle/working"` (i.e. the directory where the extracted `adapter_config.json` lives).

### Special tricks
- `detect_category(prompt)`: classifies each train prompt into one of 7 puzzle families by signature substrings (`"secret bit manipulation rule"`, `"secret encryption rules"`, `"gravitational constant"`, `"converted into a different numeral system"`, `"secret unit conversion"`, plus equation/cryptarithm sub-classes determined by inspecting the few-shot examples block). Categories: `bit_manipulation`, `cipher`, `equation_numeric_deduce`, `equation_numeric_guess`, `cryptarithm_deduce`, `cryptarithm_guess`, `gravity`, `numeral`, `unit_conversion`. Drives the per-category accuracy table and the per-category `mistakes/{category}.csv` dump.
- Equation/cryptarithm sub-classification logic peeks at the examples to decide whether the operator in the query was *seen* in examples (`*_deduce`) or *unseen* (`*_guess`).
- Per-row `minlogprob` = `min` over all logprobs of all output tokens; cheap uncertainty proxy.
- Writes a `results.csv`, `validation.csv`, and `mistakes/<category>.csv`.

### Reported score
No LB number; only train-side accuracy. The notebook is configured for `EVALUATION_SAMPLE_SIZE = 950` (the first 950 train rows).

### Gotchas
- The prompt-template mismatch above means the metric and this validation harness can disagree on identical adapters. Worth fixing in any pipeline.

---

## 3. `huikang/tinker-submission-notebook`

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/huikang__tinker-submission-notebook/tinker-submission-notebook.ipynb`

### Approach
**Hand-rolled converter**: take a Tinker-trained adapter (which targets fused Mamba `gate_proj+x_proj` and per-MoE `experts.w1`/`experts.w2` shared LoRAs) and rewrite it into the PEFT layout the metric expects (`in_proj`, `experts.<i>.up_proj`, `experts.<i>.down_proj`). Also relabels parameters from `base_model.model.model.*` to `base_model.model.backbone.*`. Optionally validates with vLLM (`TEST_GENERATION` flag). Packages a flat `submission.zip`.

### Submission format
- Input: a Kaggle model dataset at `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/26`.
- Reference layout pulled from `/kaggle/input/notebooks/huikang/nvidia-nemotron-all-linear` (contains a known-good `submission.zip`) to diff against.
- Output `submission.zip` is flat; every file in `/kaggle/working` except the zip itself is added (originals deleted after archiving).

### Adapter source / config
Tinker adapter `r=32`. `adapter_config.json` is patched to set `target_modules` to:
```python
["k_proj", "o_proj", "in_proj", "q_proj", "up_proj",
 "v_proj", "down_proj", "out_proj", "lm_head"]
```
i.e. **all attention projections + Mamba `in_proj`/`out_proj` + MoE `up_proj`/`down_proj` + `lm_head`**. This is the "all-linear" shape minus `gate_proj`/`x_proj` (because those got merged into `in_proj`).

### Conversion logic (the load-bearing trick)

**Key renaming** — Tinker uses `base_model.model.model.*`, PEFT/HF for Nemotron uses `base_model.model.backbone.*`:
```python
def trained_adapter_key_rename(key_name):
    return key_name.replace("base_model.model.model", "base_model.model.backbone")
```

**Expert unfusing** — Tinker stores a single shared LoRA per expert family (`experts.w1.lora_{A,B}` and `experts.w2.lora_{A,B}`), shape `[num_experts, ...]` with one axis size 1 (broadcast). The notebook expands and writes per-expert PEFT keys (`experts.<i>.up_proj` for w1, `experts.<i>.down_proj` for w2):
```python
if lora_A.shape[0] == 1: lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
elif lora_B.shape[0] == 1: lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()
proj_name = "up_proj" if ".w1" in base else "down_proj"
for i in range(num_experts):
    exp_renamed = re.sub(r"\.experts\.w[12]", f".experts.{i}.{proj_name}", renamed)
    tensors[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
    tensors[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
```
`.contiguous()` is mandatory — `safetensors` rejects views that share storage.

**Mamba gate+x → in_proj fusion via SVD** — Tinker has separate `gate_proj` and `x_proj` LoRA pairs; PEFT expects a single fused `in_proj` LoRA. Each `gate` and `x` is rank-32, so naive concat would be rank-64. Notebook builds a rank-64 representation then **SVD-compresses to rank 32** preserving the most singular-value mass:
```python
A_cat = torch.cat([gate_A, x_A], dim=0)          # (64, in_dim)
B_block = torch.zeros(in_proj_dim, 64)
B_block[:gate_B.shape[0], :32] = gate_B
B_block[gate_B.shape[0]:gate_B.shape[0]+x_B.shape[0], 32:] = x_B
Q_B, R_B = torch.linalg.qr(B_block)
Q_A, R_A = torch.linalg.qr(A_cat.T)
core = R_B @ R_A.T
U, S, Vh = torch.linalg.svd(core, full_matrices=False)
new_B = (Q_B @ U[:, :32]) * S[:32].unsqueeze(0)
new_A = Vh[:32, :] @ Q_A.T
```
The QR+SVD-of-core trick is numerically equivalent to SVDing the full delta but cheaper. Prints the fraction of singular-value mass retained per layer.

**Skip empty `experts.w3`**: `if ".experts.w3" in base and lora_A.numel() == 0: continue` — Tinker materializes a placeholder for w3 that is zero-sized in some configurations.

**`lm_head` is kept** (not in original reference but tolerated).

### Inference / validation setup
Same vLLM init / sampling params / prompt template as notebook #2. Includes the host's `score()` function verbatim — useful as the canonical reference for what the metric does:
- Default scorer args differ from the eval contract: `max_tokens=3584`, `top_p=1.0`, `temperature=1.0`, `max_num_seqs=128`, `max_model_len=4096`. **But** these are merely the function-signature defaults; the notebook's actual cell overrides them to `max_model_len=8192, max_tokens=7680, temperature=0.0` (the real eval values). When mocking the metric locally, override these defaults.
- The metric's `generate_predictions` builds prompts as:
  ```python
  user_content = item.prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
  prompt = tokenizer.apply_chat_template(
      [{"role": "user", "content": user_content}],
      tokenize=False, add_generation_prompt=True, enable_thinking=True,
  )
  ```
  with a fallback to raw text if `apply_chat_template` raises.
- `extract_final_answer`: prefers the **last** non-empty `\boxed{...}` capture; falls back through `"The final answer is: X"`, `"Final answer: X"`, then last numeric literal `-?\d+(?:\.\d+)?`, then last non-empty line.
- `verify`: tries float comparison with `rel_tol=1e-2, abs_tol=1e-5`; otherwise case-insensitive exact string match. Whitespace stripped.

### Special tricks
- Keeps a curated `problem_set` (~600 train IDs covering 9 categories) and a smaller `backup_problem_set` (~125 IDs) — used as evaluation subsets when full validation is too slow.
- `score()` reads test.csv with `DATA_PATH / "test.csv"` (DATA_PATH is an empty string in the notebook because the metric injects it; for local validation you must set `DATA_PATH = Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge")`).

### Reported score / LB
The adapter being packaged (`.../default/26`) is the v26 of huikang's training run; LB scores referenced elsewhere in this author's notebooks reach `LB 0.85`. Not stated explicitly in this file.

### Gotchas
- The reference `adapter_config.json` and Tinker's diverge on multiple keys (config diff cells); copying the reference's `target_modules` value is essential, because vLLM uses it to know which modules to attach LoRA to. A mismatched list silently no-ops some weights.
- `safetensors` requires `.contiguous()` after `.expand()`.
- `model_keys` is collected from the **base model** safetensors at `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/*.safetensors` to look up `in_proj` output dimensions per layer.

---

## 4. `michaelkong537/tinker-adapter-solved-some-internet-issues`

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/michaelkong537__tinker-adapter-solved-some-internet-issues/tinker-adapter-solved-some-internet-issues.ipynb`

### Approach
Tinker → submission converter using `tinker_cookbook.weights.build_lora_adapter()` (the official Tinker conversion helper), but **monkey-patched** to force the merged fused-projection rank back down to 32 (Tinker's default merge yields rank > 32, which would violate `max_lora_rank=32`).

### Submission format
```python
shutil.make_archive('/kaggle/working/submission', 'zip', '/kaggle/working/nemotron-adapter-ready-to-submit')
```
**Important difference**: `make_archive` produces a zip whose top-level contains the files of `nemotron-adapter-ready-to-submit/` flattened (since it zips the contents of the directory, not the directory itself). Result is a flat layout — compatible with the metric.

### Adapter source / config
- Input adapter: `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`.
- Base ref: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF hub spec — but the function uses cached weights so internet is not required at call time once cached).

### Special tricks — offline install
This notebook's whole reason for existence: pip-installs Tinker offline from a Kaggle dataset of wheels:
```bash
WHEEL_DIR="/kaggle/input/datasets/michaelkong537/wheel-linux/wheelhouse"
pip install --no-index --find-links="$WHEEL_DIR" tinker-cookbook tinker
```
**Critical for the submission environment** which has no internet at scoring time. Other notebooks that install from `git+https://...` only work in editing kernels with internet, not in submitted runs.

### Monkey-patch detail
Replaces `tinker_cookbook.weights._adapter._merge_fused_projections` with a custom version that:
1. Concatenates the per-component LoRA A matrices along dim 0 and builds a block-diagonal B matrix (default behavior).
2. After merging, if the resulting rank (sum of per-component ranks) exceeds `FORCED_FUSED_RANK = 32`, calls `_compress_lora_pair_to_rank(B, A, 32)` which forms `delta = B @ A`, runs full SVD, and reconstructs `B_new, A_new` at rank 32 using the symmetric `sqrt(S)` split:
```python
B_new = U[:, :32] * sqrt(S[:32]).unsqueeze(0)
A_new = sqrt(S[:32]).unsqueeze(1) * Vh[:32, :]
```
This is functionally the same SVD compression as huikang's hand-rolled version (notebook #3), but applied generically inside Tinker's library rather than only for the Mamba `in_proj` case.

### Inference / validation setup
None — packaging only.

### Reported score / LB
Not stated.

### Gotchas
- The wheelhouse approach requires building wheels for the exact Python/CPU/CUDA target of the Kaggle scoring runtime. The dataset reference is `michaelkong537/wheel-linux/wheelhouse`.
- `tinker_cookbook.weights.build_lora_adapter` writes both `adapter_config.json` and `adapter_model.safetensors` to `output_path` directly — no further surgery needed once monkey-patched.

---

## 5. `asalhi/tinker-adapter-to-ready-to-submit-adapter`

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/asalhi__tinker-adapter-to-ready-to-submit-adapter/tinker-adapter-to-ready-to-submit-adapter.ipynb`

### Approach
**Near-identical to #4**. Same monkey-patch (`patched_merge_fused_projections` + `_compress_lora_pair_to_rank`), same call to `weights.build_lora_adapter`, same `make_archive` packaging. The only delta:

### Special trick — online install
```bash
uv pip install --no-cache-dir --force-reinstall \
    "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly"
```
Pulls the **nightly** branch of `tinker-cookbook` from GitHub at notebook-run time. Works in editing kernels with internet; **will fail in the submission environment** (no internet). For submission, michaelkong537's offline wheelhouse approach (#4) is required.

### Adapter source / config
Identical: `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`, base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, output `/kaggle/working/nemotron-adapter-ready-to-submit`.

### Inference / validation setup
None.

### Reported score / LB
Not stated; this notebook is the upstream of `kienngx` version 14 (see #6).

### Gotchas
- Pin to `@nightly` because the monkey-patch targets `_merge_fused_projections` (note the leading underscore) — a private API likely to move/rename between releases.

---

## 6. `kienngx/nvidia-nemotron-trained-models-submission`

Source: `/Users/aleix.lopez/nvidia-nemotron-model-reasoning-challenge/notebooks/kienngx__nvidia-nemotron-trained-models-submission/nvidia-nemotron-trained-models-submission.ipynb`

### Approach
Pure **submission packager**: chooses one of 14 previously-trained adapter versions (stored as Kaggle model artifacts) and zips it via a shell call. No training, no validation, no inference.

### Submission format
```python
model_path = version_14  # currently selected
# shell call equivalent to: zip -r submission.zip "<model_path>"
```
**Important caveat**: `zip -r submission.zip /kaggle/input/.../1` produces a zip whose internal paths include the full `/kaggle/input/...` directory prefix. This is **not flat** and would only work because the metric uses a recursive walker (`**/adapter_config.json`) to locate the adapter. `lora_path` resolves to whatever directory contains the discovered `adapter_config.json`, and vLLM loads from there. So it works incidentally.

### Adapter sources
14 versions, each stored as a Kaggle model under `kienngx/nemotron-nano-30b-trained/...`. Default config across all:
- LoRA rank 32, alpha 32, dropout 0.05, grad accum 4, lr 2e-4, 2 epochs, seq len 2048, batch size 1.

Per-version deltas (most informative):
| v | samples | target_modules | LR | notes |
|---|---|---|---|---|
| 1 | 600 | q,k,v,o,gate,up,down | 2e-4 | rank=16 |
| 2 | 1800 | same | 2e-4 | batch=2 |
| 3 | 2400 | all-linear | 1e-4 | batch=2 |
| 4 | 9500 | in_proj,x_proj,dt_proj,out_proj | 1e-4 | 1 epoch, Mamba-only |
| 5 | 3000 | all-linear | 1e-4 | 2 epochs |
| 6 | 1200 | all-linear | 5e-5 | |
| 7 | 1200 | all-linear | 1e-4 | |
| 8 | 1200 | all-linear | 1e-5 | |
| 9 | (konbu17 nb) | — | 1e-5 | |
| 10 | (konbu17 nb) | — | 5e-5 | |
| 11 | (konbu17 nb) | — | 1e-5, batch=2, workers=8 | |
| 12 | (kienngx CoT-traj nb) | — | 1e-5, batch=2, workers=8 | |
| 13 | (impooping nb) | — | 1e-5, batch=2, workers=8, grad_accum=16 | ~4h, 94/96GB VRAM |
| 14 | (asalhi tinker-converter nb #5) | — | — | currently selected for submission |

### Special tricks
- Treats submission as a *selector* problem: train many variants in upstream notebooks, point at a Kaggle model path, zip.
- Version 4 is the only one targeting **Mamba-specific modules** (`in_proj, x_proj, dt_proj, out_proj`) — useful baseline to test whether Mamba-only LoRA matches attention-only LoRA.
- Versions 1–3 omit Mamba projections entirely (`q,k,v,o,gate,up,down` is the attention+MLP set), which the host's `target_modules` regex would *not* cover.

### Inference / validation setup
None.

### Reported score / LB
Not stated per version.

### Gotchas
- The `zip -r` packaging preserves directory structure including the leading `/kaggle/input/...` path. Works only because the metric's `adapter_config.json` glob is recursive (`**/adapter_config.json`). A safer pattern is what `huikang` does: `cd` into the adapter dir first, then zip its *contents*.
- Mixing different `target_modules` configurations across versions means the produced `adapter_config.json` differs per version; vLLM honors that config when loading the LoRA, so different versions exercise different parts of the network.

---

## Synthesis: non-obvious learnings about the submission contract

1. **The metric's prompt has a hard-coded suffix that the validation notebooks omit.** The real scoring prompt is `item.prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"` then `apply_chat_template(..., enable_thinking=True, add_generation_prompt=True)`. Training data formatting and local validation **must** include this suffix or LB will not track local accuracy.

2. **`target_modules` must match the base model's actual layer names** — `q_proj, k_proj, v_proj, o_proj` for attention, `in_proj, out_proj` for Mamba SSM, `up_proj, down_proj` for MoE expert MLPs, optionally `lm_head`. The official demo's regex `.*\.(in_proj|out_proj|up_proj|down_proj)$` deliberately skips attention and `gate_proj` — Mamba `in_proj` here is the fused gate+x projection of the SSM, not the MLP `gate_proj`. Confusing both projections breaks loading.

3. **Rank ≤ 32 is enforced per LoRA pair**, not summed. Tinker's fused-projection merge can produce ranks > 32 silently; both #4 and #5 monkey-patch `_merge_fused_projections` to SVD-compress back to 32. Without that patch, the LoRA loads but the metric will reject ranks > `max_lora_rank=32`.

4. **Tinker → PEFT requires three rewrites**: (a) rename `base_model.model.model.*` → `base_model.model.backbone.*` (Nemotron uses `backbone` not `model` as the top-level submodule); (b) unfuse shared per-expert LoRAs (`experts.w1`/`experts.w2`) into per-expert `experts.<i>.up_proj`/`down_proj` weights via `.expand(...).contiguous()`; (c) merge separate Mamba `gate_proj` + `x_proj` LoRA pairs into a single rank-32 `in_proj` LoRA via SVD-of-the-delta. Plus skip `experts.w3` if empty.

5. **The submission zip must be flat** (or at least have `adapter_config.json` discoverable via `**/adapter_config.json`). The `huikang` style (`cd /kaggle/working; zip submission.zip *`) is safest. `make_archive(dir)` with the dir as the source root works too. `zip -r submission.zip /kaggle/input/...` preserves the leading path; this only works due to the metric's recursive glob.

6. **Submission runtime has no internet.** Tinker must be vendored as wheels (see `michaelkong537/wheel-linux/wheelhouse` + `pip install --no-index --find-links`). Installs via `git+https://...` (asalhi-style) only work in editing kernels.

7. **The `metric` package has to be reconstructed at validation time** by extracting `/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script` to `/tmp` and `chmod +x` the bundled `ptxas` and `ptxas-blackwell` binaries, plus `export TRITON_PTXAS_PATH=/tmp/triton/backends/nvidia/bin/ptxas`. Without this, vLLM falls over on Blackwell GPUs. Also `uv pip uninstall torch torchvision torchaudio` first to clear Kaggle's preinstalled torch.

8. **Page-cache warmup before vLLM init is a meaningful speedup**: 16 threads × 1024 MB chunks pre-reading every `.safetensors`/`.bin` of the 30B base model. Without it the first vLLM load is I/O-bound; with it the bytes are in OS cache and vLLM mmaps them instantly.

9. **vLLM init flags that the metric uses** and any local validator should mirror: `tensor_parallel_size=1`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, `dtype="auto"`, `max_model_len=8192`, `trust_remote_code=True`, `enable_lora=True`, `max_lora_rank=32`, `enable_prefix_caching=True`, `enable_chunked_prefill=True`. Sampling: `temperature=0.0, top_p=1.0, max_tokens=7680` (greedy, full output budget). The `score()` function's *signature defaults* in #3 (`max_tokens=3584`, `max_model_len=4096`, `temperature=1.0`) are a red herring — the metric overrides them.

10. **Answer extraction is forgiving and falls through several heuristics**: last non-empty `\boxed{...}` capture → `"The final answer is: X"` → `"Final answer: X"` → last numeric literal in the text → last non-empty line. Numeric verify uses `rel_tol=1e-2, abs_tol=1e-5`; string verify is case-insensitive. Therefore training the model to always emit `\boxed{...}` is strongly preferred but not strictly required; the fallback to "last number" is a safety net (but dangerous for cipher/cryptarithm where the answer is a string).

11. **There are 9 puzzle categories** identifiable by signature substrings in the prompt (see #2's `detect_category`): `bit_manipulation`, `cipher`, `equation_numeric_deduce`, `equation_numeric_guess`, `cryptarithm_deduce`, `cryptarithm_guess`, `gravity`, `numeral`, `unit_conversion`. The deduce/guess distinction is whether the operator in the query also appears in the in-prompt examples. Worth stratifying any train/eval split by category and tracking per-category accuracy — `huikang` reports several authors hit `LB 0.85` overall, but per-category contributions are very uneven.
