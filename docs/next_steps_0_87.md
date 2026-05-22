# Next Steps to Reach 0.87 (No-Train First)

Current state:
- Best confirmed public score: **0.86**
- Best no-train merge so far: **0.86** (tie, not improvement)
- Constraint: prioritize **no retraining** due compute limits

If training resources are available, follow `docs/vertex_training_workflow.md` for the Vertex-first path.

## Goal
Get a reliable +0.01 lift (0.86 -> 0.87) with the smallest-risk sequence.

## Training fallback recipe (if no-train plateaus)
If no-train exploration stalls, use a two-stage training fallback:
1. **Continuation SFT first** (target weak category behavior and answer stability).
2. **Optional tiny GRPO second** (quality polish only after SFT is stable).

Notes:
- Do not replace SFT with GRPO as the first stage.
- GRPO is higher variance under sparse-reward scoring; keep it short and compare strictly against the SFT checkpoint.
- On smaller GPUs (like L4), prioritize SFT and treat GRPO as experimental.

## Phase 1 — Finish expanded candidate sweep (individual adapters)
1. Run latest `aleixlopez/nemotron-local-cv` (already pushed with expanded candidates).
2. Pull outputs and rank by:
   - overall accuracy (`adapter_eval_summary.csv`)
   - per-label winners (`adapter_eval_summary_per_label.csv`)
3. Keep shortlist of only adapters that:
   - tie or beat top overall score, and
   - improve weak labels (especially `equation_transformation`).

Decision gate:
- If any single adapter beats 0.86 in live LB, promote it as new base.

## Phase 2 — Minimal-risk soups on top candidates
For top 2 candidate bases from Phase 1:
1. Create tiny-weight merges (`w = 0.03, 0.05, 0.08`) with 1 donor at a time.
2. Start with full-tensor interpolation only.
3. Evaluate all variants in one CV batch.

Keep only variants that improve both:
- overall CV score, and
- weak-label slice without large regressions.

## Phase 3 — Two-donor conservative soups
If Phase 2 gives signal:
1. Try 2-donor soups with small total donor mass.
2. Example weight patterns:
   - base 0.92 + donorA 0.05 + donorB 0.03
   - base 0.90 + donorA 0.05 + donorB 0.05
3. Re-run CV and compare against best Phase 2.

Stop rule:
- If no measurable CV gain over best single-donor soup, stop this branch.

## Phase 4 — Submit discipline
Use submission slots only for strong candidates:
1. Candidate must beat/tie best CV overall.
2. Candidate must improve at least one weak label.
3. Candidate must pass packaging checks (`adapter_config.json` + `adapter_model.safetensors` at zip root).

Submit at most 1–2 candidates/day to preserve quota.

## Operational checklist (each iteration)
1. Update candidate/experiment CSV.
2. Regenerate Kaggle notebooks:
   - `uv run python scripts/build_kaggle_notebooks.py`
3. Push CV kernel:
   - `uv run kaggle kernels push -p kaggle/cv_kernel`
4. Wait for run completion and download outputs.
5. Promote only winning candidates to submit-packager notebook.

## When to stop no-train exploration
Stop no-train search if all are true:
- 2+ full sweep cycles produce no lift above 0.86,
- tiny soups and two-donor soups also plateau,
- per-label gains fail to transfer to overall improvement.

At that point, either:
- accept 0.86 as ceiling under no-train constraints, or
- run one targeted small retrain cycle only on weak categories.
