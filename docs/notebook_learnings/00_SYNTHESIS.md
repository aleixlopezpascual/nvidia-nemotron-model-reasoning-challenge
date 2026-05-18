# Master Synthesis — Public Notebook Landscape (as of 2026-05-18)

Cross-cutting insights from **36 top notebooks** on the **NVIDIA Nemotron Model Reasoning Challenge**, distilled from nine thematic write-ups (`01_baseline_and_submission.md` → `09_per_puzzle_solvers.md`).

Read this first. The per-theme files are the deep references.

> **Round-2 update**: After reading 16 additional notebooks (the 0.86 LB set, novel methods, per-puzzle solvers, local CV tooling), the strategic picture sharpened considerably. See "0.85 → 0.86 → 0.87" section below.

---

## The competition in one paragraph

Train a **rank-≤32 LoRA** for **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** (a hybrid Mamba+attention MoE model with 128 experts) that solves rule-induction puzzles. Eval runs in **vLLM** with greedy decoding, `max_model_len=8192`, `max_tokens=7680`; the scorer extracts the answer from `\boxed{...}` (with fallback heuristics) and grades exact-string or relative-numeric. Public LB top is 0.87; multiple public notebooks claim 0.85. Our last submission scored 0.48 — there is a clear, reproducible recipe between us and the 0.85 cluster.

## The 0.85 recipe (convergent across top notebooks)

This is what `huikang`, `dgxchen`, and (indirectly) the Tinker-based packagers all do. **Reproducing this should be our first concrete milestone.**

| Component | Setting |
|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `attn_implementation="eager"`, `trust_remote_code=True`, **bf16 (NOT 4-bit)** |
| Framework | Unsloth `FastLanguageModel` + TRL `SFTTrainer` *or* a hand-rolled loop with Cut-Cross-Entropy |
| LoRA r / α / dropout | **32 / 32 / 0.0**, `bias="none"` |
| LoRA target_modules | **9 layers**: `q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj, lm_head` (Mamba `in_proj`/`out_proj` are critical — this is a hybrid arch) |
| Optimizer | full-precision **AdamW** (NOT 8-bit), β=(0.9, 0.95), eps=1e-8, weight_decay=0, no grad clip |
| LR & schedule | **2e-4**, **linear** decay, **zero warmup** |
| Batch / steps | effective batch **32** (micro=1, grad_accum=32), `max_seq_len=8192`, **~1 epoch / ~1000 steps** |
| Loss | masked to assistant turn only (prompt-loss masking is *standard* in the 0.85 set — see "open levers" below for the basic-SFT group that misses this) |
| Data | **`tonghuikang/nemotron` corpus** (Tinker-distilled CoT, `<think>...</think>\boxed{ans}` format) — this is the upstream data secret |
| Submission | `submission.zip` containing just `adapter_config.json` + `adapter_model.safetensors`, with `base_model_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"` |

**Key packaging quirks** (silent foot-guns):
- Unsloth silently drops `lm_head` from MoE LoRA — manually re-add it.
- After saving, rename `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*` in the safetensors so vLLM finds it (Nemotron's top-level submodule is `backbone`, not `model`).
- Submission env has **no internet**: ship a `wheelhouse` Kaggle dataset for any non-stdlib deps.

## The data is the moat

Across the top group, **the LoRA hyperparameters are almost identical — the only meaningful axis of differentiation is the training corpus**. Conclusions:

1. **Rejection sampling against ground truth is mandatory.** `konbu17` documents per-puzzle-type checkers (exact 8-bit for Bit Manipulation, ±0.05 abs / ±0.5% rel for numerics, exact for Roman/Cipher/Equation). All correctly-solved teacher traces are kept; easy puzzle types are downsampled to focus capacity on the hard ones.
2. **The hard puzzle types are the bottleneck.** Bit Manipulation: 40% teacher pass rate; Equation Transformation: 13.6% pass rate. Public notebooks explicitly flag Equation Transformation as the *next* big win — no one has cracked it yet.
3. **Solver-augmented prompts beat pure LLM CoT** on the hard types. `konbu17` injects:
   - The 77-word Alice's-Wonderland substitution dictionary into Text Encryption prompts.
   - Per-bit boolean function candidate analysis into Bit Manipulation prompts.
   - Nothing into Equation Transformation prompts — and it shows.
4. **CoTs over 7,680 tokens are *rewritten* by the teacher**, not truncated. Truncation = silent loss of the boxed answer = wrong label.
5. **No one names their teacher model in public code.** All top corpora are pre-built Kaggle datasets with opaque generation. The pattern is "any strong reasoner + per-type rejection sampling + optional solver prompts" — the *specific* teacher matters less than the *process*.

## Novel methods, verdicts (from round 2)

| Method | Notebook | Verdict |
|---|---|---|
| **Loss masking (`completion_only_loss=True`)** | waterjoe | **Use** — free win if our loop doesn't already mask the prompt. Most basic-SFT notebooks miss this. |
| **Muon optimizer** | pkuszboi | **Try** — matches AdamW with ~25% less wall-clock at LR ~1e-3 → 4e-3. Verbatim drop-in code in `07_novel_training_methods.md`. |
| **Adapter fusion / rank compression** | rauffauzanrambe | **Defer** — no empirical lift demonstrated. Revisit only if we have 2+ specialist adapters to combine. |
| **Soft-balanced sampling** | newduck | **Skip** — 4-category regex classifier is a worse proxy than the CoT corpus' real `type` column; α=0.5 unmeasured. |
| **"Think Twice" self-correcting CoT** | hammadfarooq470 | **Skip** — title is misleading; no actual self-correction logic in the notebook. |
| **GRPO (RL polish)** | amanatar v3 | **Skip for now** — amanatar's own v3 has `USE_GRPO=False` with comment "GRPO not helping yet — model needs more SFT first". Direct multi-iteration negative evidence. |

## Local CV (essential tooling)

`jiazhuang/nemotron-local-cv` provides a **950-row stratified CV harness** that closely mirrors the host metric. Verbatim code in `08_local_cv_and_adapter_fusion.md`. Two caveats to validate:

1. ~~The notebook's `\boxed{}` instruction wraps the example in backticks — check against the actual host `metric.py`.~~ **Resolved 2026-05-18**: pulled host kernel `metric/nvidia-nemotron-metric` directly. Backticks are correct — the verbatim host suffix is `'\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'`. Our local CV harness (`nemotron_local_cv/host_metric.py`) mirrors this byte-for-byte.
2. The `score()` defaults are wrong (`max_tokens=3584, temperature=1.0, max_model_len=4096`) — always override to match the host (`7680, 0.0, 8192`).

Cost: ~30–90 min per CV run on an H100. Statistical SE ~1.1pp at 80% accuracy on 950 rows. Affordable as ~1 CV per training experiment, which is exactly the iteration speed we need given the 5/day submission cap.

## Where there is no consensus (= our biggest open levers)

These are the disagreements between public notebooks. Each is a controlled experiment worth running:

| Knob | Camp A | Camp B | Our move |
|---|---|---|---|
| `lora_alpha` | 16 (α/r=0.5, conservative) | 64 (α/r=2.0, "NVIDIA-recommended") | Try both; alpha=α/r is the most direct lever on adapter strength. The 0.85 set uses α=32 (α/r=1.0). |
| `target_modules` | `"all-linear"` (everything) | narrow 9-layer set (top recipe) | The 0.85 recipe wins with the narrow set, so prefer that. But `embed_tokens` is an open question (`llkh0a` adds it). |
| Loss masking | Most basic-SFT notebooks train on prompt+answer combined | 0.85 set masks loss to assistant turn | **Masking is almost certainly correct.** This is a clear gap in the basic-SFT group. |
| Training data size | 600–1000 subsampled | full 9500 + synthetic CoT | The 0.85 recipe is `tonghuikang/nemotron` (synthetic CoT), not raw subsample. Use the synthetic corpus. |
| Quantization | 4-bit (Unsloth default) | bf16 full precision | 0.85 set uses bf16 — **do not** load_in_4bit at training. |
| SFT only vs SFT+GRPO | SFT-only dominates public LB | One public GRPO notebook, no verified LB lift | **Treat GRPO as a polish step, not a primary lever.** |

## 0.85 → 0.86 → 0.87 (updated)

This section is rewritten after round 2. **The picture is much clearer than the round-1 speculation.**

### 0.85 → 0.86: it's the training environment, not the recipe

All three public "0.86 LB" notebooks (`mohamedamr992`, `afr1ste`, `safar1`) are **packaging-only shells around a single shared upstream adapter** trained on **Tinker** (off-Kaggle compute): `kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1`. No notebook trains a 0.86 adapter from scratch. Crucially:

- `huikang`'s on-Kaggle v20 adapter (the canonical 0.85 recipe artifact) scores **0.85**.
- `kienngx`'s Tinker-trained adapter (same `nemotron-master` lineage, same recipe) scores **0.86**.
- **Same recipe, better compute environment → +0.01.**

This means our access to Vertex AI is *exactly* the right unlock for the 0.85→0.86 step. Training off Kaggle (where the GPU envelope is generous, bf16 is real, no notebook timeout) is what crosses the line. We don't need novel tricks for the first +0.01 — we just need to train somewhere serious.

Counter-evidence against red herrings: every "broad LoRA target" experiment (Sluitel, Error1249x, Rick v41, BharatMohan v31) lands at **0.57–0.62**, including with hand-tuned `lm_head` fills. Structural changes to the adapter are necessary but not sufficient — the 0.86 ceiling on huikang's recipe is robust.

### 0.86 → 0.87: the public ceiling is real, and only Equation Transformation cracks it

Top public scores cluster at 0.86–0.87. The top-2 cluster (~0.87) hasn't published. From the per-puzzle-type solver analysis (see `09_per_puzzle_solvers.md`), here's the per-type accuracy ceiling implied by the available solvers and CoT data:

| Puzzle type | Strongest published solver | Accuracy |
|---|---|---|
| Numeral Conversion | mohankrishnathalla Roman ↔ Arabic | ~100% |
| Gravitational Constant | mohankrishnathalla `d = ½gt²` extractor | 99.4% |
| Unit Conversion | mohankrishnathalla ratio extractor | 99.4% |
| Text Encryption | konbu17 77-word dictionary injection | 94–97% |
| Bit Manipulation | konbu17 4-phase per-bit boolean analyzer | 75.5–78.7% |
| Cryptarithm sub-family of Equation Transformation | kimberleyduran cryptarithm verifier | ~12% covered |
| **`equation_numeric` (732 rows)** | **NONE — zero public solver** | **~29% (model-only)** |

konbu17 v2's LoRA hits 29.6% on the full Equation Transformation family vs 78.7% on Bit Manipulation with the same recipe. **The Equation Transformation accuracy hole — ~700 rows worth of misses, worth roughly +0.05 LB if we could fully solve them — is the single most concrete unattacked frontier in public material.** kimberleyduran's partial cryptarithm port added +0.01 LB (0.83→0.84) when wired into dgxchen's recipe; a complete `equation_numeric` solver would plausibly do better.

The build target for the 0.87 leap: **port the unpublished `equation_numeric_deduce.py` from Tong Hui Kang's repo** (referenced but never reproduced inline in public Kaggle notebooks) and/or build an equivalent with constraint propagation / Z3 + joint deduction. This is the one public route to the top.

### Other plausible 0.87+ levers (lower confidence)

- **Better CoT teacher** for hard puzzle types (DeepSeek-R1, Qwen2.5-Math-72B, GPT-4, Claude — all rules-compliant per our compliance doc).
- **Test-time refinement**: greedy at temp=0 *blocks self-consistency*, but prompt-time "verify the rule against all examples, then apply" patterns can be baked into the CoT corpus.
- **Mamba-specific tricks**: public set treats `in_proj`/`out_proj` as ordinary Linear; no published exploration of state-aware adapters.
- **Test set diversity**: public LB is computed on the sample test (3 rows); hidden test is "several hundred" — there will be shake-up, so leaderboard position alone undersells our calibration.

## RL / GRPO summary (lowest-leverage area today)

- One public GRPO notebook (`johnnyhyland`). Pure GRPO; no SFT warm-start in the public version. `beta=0.0` (no KL, no reference model — to save memory). Small scale: 220 prompts × 4 generations × 1 epoch.
- **Most reusable artifact:** the cosine-length-scaled reward from Light-R1, which solves the *zero-advantage problem* (when all rollouts agree, flat ±1 rewards give zero gradient). See the verbatim reward function in `05_unsloth_and_rl.md`.
- **No public LB lift verified** for GRPO. Treat as "SFT first → optional 50–200 GRPO step polish on a strong SFT base", not as a competing primary lever.

## Submission environment foot-guns

From `01_baseline_and_submission.md`:

- The host's eval prompt appends the literal string (with backticks!) `'\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`'` to every user message *before* the chat template is applied with `enable_thinking=True`. **Any local validator that omits this suffix will disagree with the LB.** Verified verbatim against host kernel `metric/nvidia-nemotron-metric` (2026-05-18 pull).
- Tinker→PEFT adapter conversion has three load-bearing rewrites: `model.*` → `backbone.*` (Nemotron's submodule rename), per-expert unfusing of shared `experts.w1/w2` LoRAs into per-expert projections, and SVD-compression of Mamba's separate `gate_proj`+`x_proj` LoRAs into a single rank-32 `in_proj` LoRA.
- The submission environment has **no internet**. Only the wheelhouse-Kaggle-dataset approach works for external deps.

---

## Recommended next steps (refined after round 2)

The strategic picture is now concrete enough to give a specific plan. The 0.85→0.86 step is just "train off Kaggle on bigger compute" — which we have via Vertex AI. The 0.86→0.87 step has *one clear public-route attack*: build the missing Equation Transformation solver and rebuild that puzzle type's CoT data.

**Suggested order of operations:**

1. **Stand up a local CV harness** based on `jiazhuang/nemotron-local-cv`. Validate the boxed-instruction-suffix detail against the actual host `metric.py`. Verbatim code is in `08_local_cv_and_adapter_fusion.md`. *Estimated effort: 1 day.*
2. **Reproduce `huikang/end-to-end-finetuning-for-lb-0-85` on Vertex AI** with `completion_only_loss=True` baked in. Confirm we hit ~0.85 locally and ~0.86 on the LB (the "train off-Kaggle bonus"). *Estimated effort: 1–2 days once compute is provisioned.*
3. **Build the Equation Transformation solver** (the missing `equation_numeric_deduce.py`-style verifier; pair it with constraint propagation / Z3). Use it to (a) generate solver-augmented CoT and (b) rejection-sample teacher CoT against ground truth. *Estimated effort: 3–7 days.*
4. **Re-train with the augmented Equation Transformation data + Bit Manipulation upweighting** as konbu17 does. Target LB: ~0.87. *Estimated effort: 2–3 days.*
5. (Optional speed-up) **Swap AdamW for Muon** to reduce wall-clock per training run (~25% faster). Worth it once we're doing many iterations.

This is plausibly ~2–3 weeks of work with our compute access, against ~4 weeks of competition runway.

**Alternative path**: pursue an Open Contribution Award (Best Data Method, RL, or Fine-tuning — 1 DGX Spark each, requires top-10% LB + form submission per category). The Equation Transformation solver work would naturally compete in the Data Method category even if the LB lift falls short of the top-3 cash prizes.

**Open question for the user**: do we want to (a) commit to the LB-chasing plan above, (b) swing harder for a novel Open Contribution Award angle, or (c) brainstorm a different attack before committing?
