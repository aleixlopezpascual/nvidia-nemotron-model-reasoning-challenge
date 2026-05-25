# Next Steps to Reach 0.87

## Current State (2026-05-25)

| Item | Value |
|---|---|
| Best confirmed public score | **0.86** |
| LB teams at 0.87 | **7** (including Kh0a/kienngx, huikang, fate, JK-Piece, Lora-is-all-you-need, Researcher-7919, Y\|M\|F) |
| Phase 2 merge search | **Closed** — all 8 variants scored 0.84–0.86, none exceeded baseline |
| Untried public adapters | **10 new adapters** (kienngx new variants + huikang v21–v27) |

> **Key finding (2026-05-25):** Kh0a (kienngx) is now at **0.87** on the LB and has published many new adapter variants since our last scan. huikang also has v21–v27 (we only used v20). These are the highest-priority candidates to evaluate **before starting any retraining**.

---

## Phase 3 — Untried community adapter evaluation (IMMEDIATE priority)

### New candidates to evaluate (added to `data/adapter_candidates.csv`)

| Name | Path | Rationale |
|---|---|---|
| `kienngx_9500s_batch1_lr1e-4` | `.../9500s-batch1-lr1e-4/1` | Full 9500-sample corpus — likely Kh0a's 0.87 adapter |
| `kienngx_cot_labels_3000samples` | `.../cot-labels-3000samples/1` | CoT-labels corpus; significant sample jump |
| `kienngx_1200samples_cot_1e-5` | `.../1200samples-cot-1e-5/1` | CoT training variant |
| `kienngx_1200samples_cot_5e-5` | `.../1200samples-cot-5e-5/1` | Higher LR CoT variant |
| `kienngx_2400_lr1e-4_all_linear` | `.../2400-1e-4_lr-all_linear-packingfalse/1` | More data, all-linear target modules |
| `kienngx_1800s_rank32` | `.../1800s-32/1` | Rank-32 1800-sample run |
| `kienngx_600_packing_false` | `.../600-samples-packing-false/1` | Packing-off ablation |
| `huikang_v27` | `huikang/nemotron-adapter/.../27` | Latest huikang (7 versions newer than what we used) |
| `huikang_v25` | `huikang/nemotron-adapter/.../25` | Mid-point version |
| `huikang_v22` | `huikang/nemotron-adapter/.../22` | Earlier new version |

### Workflow

1. Update `data/adapter_candidates.csv` ✅ (done)
2. Push new code snapshot: `uv run python scripts/sync_code_to_kaggle.py -m "add new adapter candidates"`
3. Rebuild + push CV kernel: `uv run python scripts/build_kaggle_notebooks.py && uv run kaggle kernels push -p kaggle/cv_kernel`
4. Run CV on all candidates. Focus on: `kienngx_9500s_batch1_lr1e-4` and `huikang_v27` first.
5. Any single adapter scoring > 0.86 CV → submit directly for LB confirmation.

---

## Phase 4 — Merge with new bases (if Phase 3 adapters don't solo-hit 0.87)

If `kienngx_9500s_batch1_lr1e-4` or another new adapter scores 0.86 CV but not 0.87 LB:
- Attempt mini-soups with the best new adapter as base + tinker-adapter as donor (or vice versa)
- Keep donor weight ≤ 0.08 to stay conservative
- Evaluate 3 soup variants max; submit only if CV ≥ 0.87

---

## Phase 5 — Targeted continuation SFT (last resort, requires GPU)

If Phase 3 + 4 both plateau at 0.86:

The winner (huikang) confirmed **0.87 requires bit_manipulation improvement** — it goes from 85.1% → 88-90%. Strategy:
1. Train continuation SFT on `bit_manipulation` traces only (~1 epoch)
2. CoT redesign to reduce low-confidence tokens (check `min_logprob` per trace)
3. All other categories should remain unchanged (only fine-tune on bit_manipulation corpus)

This requires a Kaggle Blackwell GPU session (~1-2hr training). See `docs/kaggle_workflow.md`.

---

## Submit discipline
- Never submit without local-CV first
- Submit at most 1–2 candidates/day to preserve quota
- Submission must improve both: overall CV AND at least one weak label
- Packaging: `adapter_config.json` + `adapter_model.safetensors` at zip root

---

## When to stop
- If Phase 3 finds a ≥0.87 standalone adapter → submit, done
- If Phase 3+4 both plateau at 0.86 → accept 0.86 or run Phase 5
- Competition deadline: **2026-06-15** (~3 weeks remaining

## Historical context
- Phase 1 (individual adapter sweep, May 17-20): evaluated 7 adapters; best = 0.86 (kienngx tinker-adapter)
- Phase 2 (merge search, May 20-25): 8 merge variants; all scored 0.84–0.86; none exceeded 0.86
- See `docs/archive/vertex_ai_setup.md` for the dead-end Vertex AI path
