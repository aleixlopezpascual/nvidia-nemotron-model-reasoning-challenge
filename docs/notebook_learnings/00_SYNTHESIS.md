# Master Synthesis — Public Notebook Landscape (as of 2026-05-19)

Cross-cutting insights from **36 top notebooks** on the **NVIDIA Nemotron Model Reasoning Challenge**, distilled from ten thematic write-ups (`01_baseline_and_submission.md` → `10_open_progress_prize_winner.md`).

Read this first. The per-theme files are the deep references.

> **Round-2 update**: After reading 16 additional notebooks (the 0.86 LB set, novel methods, per-puzzle solvers, local CV tooling), the strategic picture sharpened considerably. See "0.85 → 0.86 → 0.87" section below.

> **Round-3 update (2026-05-19)**: Tong Hui Kang (`tonghuikang`) — the author of the `nemotron-master` corpus AND the Open Progress Prize winner — published his full write-up. This is the ground truth for the 0.85 recipe. Key corrections to our earlier analysis are marked **[CORRECTED]** below. See `10_open_progress_prize_winner.md` for the full deep-dive.

---

## The competition in one paragraph

Train a **rank-≤32 LoRA** for **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** (a hybrid Mamba+attention MoE model with 128 experts) that solves rule-induction puzzles. Eval runs in **vLLM** with greedy decoding, `max_model_len=8192`, `max_tokens=7680`; the scorer extracts the answer from `\boxed{...}` (with fallback heuristics) and grades exact-string or relative-numeric. Public LB top is ~0.87; the Open Progress Prize winner (tonghuikang) achieved **0.85 LB** with a published approach. Multiple public notebooks reproduce 0.85. Our last submission scored 0.48 — there is a clear, reproducible recipe between us and the 0.85 cluster.

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

This section is rewritten after round 2 and corrected after round 3. **The picture is now very clear.**

### 0.85 → 0.86: it's the training environment, not the recipe

**[CORRECTED in round-3]** Our earlier analysis said "Tinker = 0.86"; the winner's writeup clarifies the picture:

- `huikang` (tonghuikang, Open Progress Prize winner) uses **Tinker** and achieves **0.85** (not 0.86).
- `kienngx` / KhoA (who scores **0.86**) uses the same general recipe but achieves perfect scores on numeral, gravity and unit_conversion where huikang's model still sometimes fails — suggesting KhoA resolved the **training-serving misalignment** caused by the SVD adapter conversion, or uses a different training/conversion path entirely.
- **The 0.85→0.86 gap is not simply "better compute" — it is resolution of the SVD adapter conversion loss** (75% singular mass retained → training-serving misalignment on easy puzzles). Fix the adapter conversion fidelity to close the gap.

All three public "0.86 LB" notebooks (`mohamedamr992`, `afr1ste`, `safar1`) are packaging-only shells around `kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1`. No notebook trains a 0.86 adapter from scratch publicly.

Counter-evidence against red herrings: every "broad LoRA target" experiment (Sluitel, Error1249x, Rick v41, BharatMohan v31) lands at **0.57–0.62**. Structural changes to the adapter are necessary but not sufficient.

### 0.86 → 0.87: the real bottlenecks (corrected)

**[CORRECTED in round-3]** Our earlier analysis flagged `equation_numeric` as "the unattacked frontier." The winner's writeup shows this is wrong:

- `equation_numeric_deduce`: **90.6% solved** in the winning approach (≥ 541/596). **This is NOT the bottleneck.**
- The real bottlenecks are:

| Puzzle type | Winning solve rate | Gap vs 100% | Notes |
|---|---|---|---|
| cryptarithm_deduce | 8.2% (54/659) | 605 misses | Nemotron is bad at symbol concatenation; model memorizes, not generalizes |
| cryptarithm_guess | 6.7% (11/164) | 153 misses | Needs guessing / RL, not deterministic CoT |
| equation_numeric_guess | 15.4% (21/136) | 115 misses | Inherently ambiguous (unseen operator) |
| bit_manipulation (complex) | ~85% overall | Complex ops (ROT+XOR+NOT chains) exceed token budget |

The **single biggest recoverable gap** is cryptarithm (~758 misses × ≥ some fraction solvable). But: the winner notes Nemotron is "quite bad at splitting and concatenating symbols" — this may be a fundamental model limitation, not a CoT-design problem.

Top solutions at 0.877 will need to:
1. Fix SVD training-serving misalignment (KhoA likely already did this).
2. Handle complex bit manipulation chains with a more compact token representation (e.g. base-64 bit encoding).
3. Make progress on cryptarithm — requires either: (a) token-level work to fix Nemotron's symbol handling, or (b) a guessing/RL approach for the non-deterministic sub-problems.

### Other plausible 0.87+ levers (lower confidence)

- **Better CoT teacher** for hard puzzle types (DeepSeek-R1, Qwen2.5-Math-72B, GPT-4, Claude — all rules-compliant per our compliance doc).
- **Test-time refinement**: greedy at temp=0 *blocks self-consistency*, but prompt-time "verify the rule against all examples, then apply" patterns can be baked into the CoT corpus.
- **Mamba-specific tricks**: public set treats `in_proj`/`out_proj` as ordinary Linear; no published exploration of state-aware adapters.
- **Test set diversity**: public LB is computed on the sample test (3 rows); hidden test is "several hundred" — there will be shake-up, so leaderboard position alone undersells our calibration.

## RL / GRPO summary (lowest-leverage area today)

- One public GRPO notebook (`johnnyhyland`). Pure GRPO; no SFT warm-start in the public version. `beta=0.0` (no KL, no reference model — to save memory). Small scale: 220 prompts × 4 generations × 1 epoch.
- **Most reusable artifact:** the cosine-length-scaled reward from Light-R1, which solves the *zero-advantage problem* (when all rollouts agree, flat ±1 rewards give zero gradient). See the verbatim reward function in `05_unsloth_and_rl.md`.
- **No public LB lift verified** for GRPO. Treat as "SFT first → optional 50–200 GRPO step polish on a strong SFT base", not as a competing primary lever.
- **[CONFIRMED by winner]**: "I already have the optimal policy for solvable problems… The only job of the LLM is to follow the policy." RL adds no signal when the target tokens are deterministic. The exception is cryptarithm (guessing steps) — but that is a hard, speculative problem.

## Submission environment foot-guns

From `01_baseline_and_submission.md` and confirmed by the Open Progress Prize winner:

- The host's eval prompt appends the literal string (with backticks!) `'\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`'` to every user message *before* the chat template is applied with `enable_thinking=True`. **Any local validator that omits this suffix will disagree with the LB.** Verified verbatim against host kernel `metric/nvidia-nemotron-metric` (2026-05-18 pull).
- **[WINNER-CONFIRMED]** Tinker→PEFT adapter conversion has four load-bearing rewrites: `model.*` → `backbone.*` (Nemotron's submodule rename), per-expert unfusing of shared `experts.w1/w2` LoRAs into per-expert projections, SVD-compression of Mamba's separate `gate_proj`+`x_proj` LoRAs into a single rank-32 `in_proj` LoRA (only **75% singular mass retained — lossy**), and keeping `lm_head` LoRA weights.
- **[NEW]** The SVD lossiness is the root cause of training-serving misalignment observed at 0.85 LB: model fails to write `\boxed`, generates degenerate repeated-token sequences, misses template words. This accounts for the ~2–3pp gap between 87.7% solver target and 0.85 LB.
- The submission environment has **no internet**. Only the wheelhouse-Kaggle-dataset approach works for external deps.

---

## Recommended next steps (refined after round 3)

**[CORRECTED]** The strategic picture is now based on the winner's first-person writeup. Key changes vs round-2:
- Step 2 target is still ~0.85 on LB (not 0.86 as previously estimated — the 0.86 gap is adapter fidelity, not compute alone).
- Step 3 is **not** Equation Transformation (already ~91% solved) — it is fixing adapter conversion fidelity and tackling complex bit manipulation.

**Suggested order of operations:**

1. **Stand up a local CV harness** based on `jiazhuang/nemotron-local-cv`. Validate the boxed-instruction-suffix detail against the actual host `metric.py`. Verbatim code is in `08_local_cv_and_adapter_fusion.md`. *Estimated effort: 1 day.* ✅ (Done — `nemotron_local_cv/` mirrors host metric byte-for-byte.)
2. **Reproduce the tonghuikang 0.85 recipe on Kaggle Notebooks (Blackwell GPU)** with `completion_only_loss=True` baked in, using the `tonghuikang/nemotron` corpus. Target: ~0.85 LB. *Estimated effort: 1–2 days.*
3. **Resolve SVD training-serving misalignment** — either: (a) train rank-16 on Tinker / end-to-end and losslessly upscale, (b) post-finetune after SVD conversion on high-loss tokens (winner's untried proposal — promising), or (c) implement end-to-end SFT without Tinker entirely. This is the path from 0.85 → 0.86. *Estimated effort: 2–4 days.*
4. **Improve bit manipulation CoT** for complex operation chains — compact token representation (base-64 bit encoding?) to stay within 7680-token budget on nested ops. *Estimated effort: 3–5 days.*
5. **Investigate cryptarithm** — the only category that is nearly unimproved (7–8%). Requires either token-level fixes to Nemotron's symbol handling or a guessing-capable approach. *High uncertainty; may be a hard ceiling.*
6. (Optional) **Swap AdamW for Muon** to reduce wall-clock per training run (~25% faster). Worth it once we're doing many iterations.

This is plausibly ~2–3 weeks of work with our compute access, against ~4 weeks of competition runway.

**Alternative path**: pursue an Open Contribution Award (Best Data Method, RL, or Fine-tuning — 1 DGX Spark each, requires top-10% LB + form submission per category). The Open Progress Prize winner (tonghuikang) has published detailed methodology. We can build on it explicitly and contribute our own improvements.

**[NEW] Key training insight from winner**: The training objective is **maximize minimum logprob**, not minimize average cross-entropy. Monitor per-trace `min_logprob`; if any token has `logprob < ln(2) ≈ 0.693`, that token is more likely than not to be generated differently at inference. Redesign the CoT to address those specific tokens. This is a much richer feedback signal than the aggregate loss curve.
