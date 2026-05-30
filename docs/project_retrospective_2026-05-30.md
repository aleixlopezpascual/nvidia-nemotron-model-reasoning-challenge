# NVIDIA Nemotron Challenge — Project Retrospective (2026-05-30)

## Final status

- **Target:** reach **0.87** public LB.
- **Best reproducible result achieved:** **0.86**.
- **Final decision:** stop active experimentation due to diminishing returns and no available no-train path that reproducibly exceeds 0.86 in our environment.

---

## 1) Competition context (what mattered most)

This competition requires submitting a **LoRA adapter (rank <= 32)** for:
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

Key scoring constraints we aligned to:

- vLLM inference with greedy decoding (`temperature=0.0`, `top_p=1.0`)
- `max_model_len=8192`, `max_tokens=7680`
- answer extracted from `\boxed{...}` with fallback heuristics
- strict submission format: `submission.zip` containing `adapter_config.json` + `adapter_model.safetensors`

Operational realities that shaped strategy:

- hidden test is much larger than sample `test.csv` (avoid overfitting to public sample)
- submission quota is limited per day
- reproducibility and packaging correctness are as important as model quality

Source references in this repo:

- `docs/notebook_learnings/00_SYNTHESIS.md`
- `docs/kaggle_workflow.md`
- `docs/rules_compliance.md`

---

## 2) What we built

### A. Host-metric-aligned local CV stack

Implemented `nemotron_local_cv/` to mirror host behavior as closely as possible:

- host metric mirror, including `\boxed{}` extraction behavior
- stratified validation support (`data/val_split_600.csv`)
- per-label analysis and adapter-manifest evaluation flow

Why this mattered:

- prevented blind submissions
- improved comparability between candidates
- reduced false confidence from non-parity eval settings

### B. SFT training pipeline

Implemented `nemotron_train/` with a reproducible LoRA recipe:

- rank-32 LoRA config and training entrypoint
- data formatting pipeline aligned to competition prompt/answer behavior
- Kaggle kernel generation path for train + CV runs

### C. Kaggle-first engineering workflow

Built/maintained:

- code sync pipeline to Kaggle datasets
- notebook builders and kernel metadata flow
- Blackwell-specific stability patches (`kaggle/blackwell_setup.py`)
- helper scripts for packaging and state tracking

### D. No-train adapter triage + merge experiments

Operationalized:

- candidate manifest evaluation (`data/adapter_candidates.csv`)
- merge experiment manifest (`data/merge_experiments.csv`)
- phase automation scripts for merge polling/submission/materialization

---

## 3) What worked

1. **Strong evaluation discipline:** local CV parity and metric mirroring reduced guesswork and avoided many low-signal submissions.
2. **Infrastructure quality:** Kaggle automation, packaging, and kernel workflows matured significantly and became repeatable.
3. **No-train baseline discovery:** we reached and repeatedly validated a **0.86-tier** baseline with public-provenance adapters/variants.
4. **Research quality:** notebook synthesis and discussion tracking correctly identified key bottlenecks and non-solutions (e.g., naive GRPO or random hyperparameter churn).

---

## 4) What did not work (or did not pay off)

1. **No-train merge search beyond 0.86:** multiple merge variants tied or regressed (roughly 0.84-0.86 band), with no durable +0.01 lift.
2. **Speculative adapter chasing:** several promising public artifacts were not accessible/reproducible in our execution path (missing/mount-path issues).
3. **Packaging/layout tweaks as a performance lever:** useful for reliability, but insufficient alone to cross 0.86 -> 0.87.
4. **Broad experimentation without new signal:** additional sweeps had poor ROI once the 0.86 plateau was established.

---

## 5) Why 0.87 was not reached

Most evidence (internal and public) indicates the last +0.01 requires at least one of:

- higher-fidelity training/conversion alignment (especially around Tinker/SVD-style serving mismatch)
- higher-quality targeted traces for hardest categories (not just more data volume)
- additional compute budget for focused retraining and repeated variance-aware validation

Given constraints at the end:

- no practical training budget path
- no reproducible no-train artifact in our environment that beat 0.86

the project hit a rational stopping point.

---

## 6) Highest-potential improvements not fully tried

If work resumes with compute, these are the most promising unexhausted directions:

1. **Conversion-fidelity A/Bs (high priority):**
   - isolate training-serving mismatch sources
   - compare conversion paths on the same checkpoint/data
   - optimize for token-level reliability (not only aggregate loss)

2. **Targeted hard-category data generation:**
   - cryptarithm and complex bit-manipulation focused traces
   - rejection-sampled, quality-filtered CoTs
   - low-min-logprob replay buckets rather than broad synthetic expansion

3. **Focused continuation SFT (most actionable next step):**

   The idea: take an existing trained adapter and continue fine-tuning on cryptarithm-only traces — **no retraining from scratch**.

   How it works:
   - Load the frozen base model + existing adapter as the starting point (initialised from `adapter_model.safetensors`, not random weights)
   - Run a short additional training pass (~100–300 steps) on cryptarithm-specific data
   - Save the updated adapter

   Practical rules to avoid regressions:
   - Use a **low learning rate and few steps** — you are polishing, not relearning; too many steps causes catastrophic forgetting of the other puzzle categories
   - **Mix in ~10–20% anchor examples** from other puzzle types to keep those categories stable
   - **Gate with per-label CV** before submitting: confirm cryptarithm improved without other labels regressing

   The hard constraint: this only works if you have **correct cryptarithm CoT traces to train on**. Cryptarithm has a ~8% teacher pass rate even for strong models — generating usable traces requires a very capable teacher model plus aggressive rejection sampling. Training on bad or empty traces will hurt more than help.

4. **Variance-aware leaderboard protocol:**
   - repeated submissions for near-tie candidates
   - treat +/-0.01 as potentially noisy unless replicated

5. **Optional optimizer/runtime upgrades once baseline is stable:**
   - speed-focused experiments (e.g., Muon-style throughput gains) only after data/conversion fidelity is controlled

---

## 7) Practical handoff

Current repo is in a strong state for future continuation:

- robust local CV harness
- reproducible Kaggle workflow
- mature automation around triage/merge/submission
- extensive research synthesis documenting known good/bad paths

If resumed later, the fastest path is to restart from the proven **0.86 baseline**, then spend compute only on **conversion fidelity + hard-category data quality**, not broad no-train sweeps.

