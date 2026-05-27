# Next Steps to Reach 0.87

## Final no-train status (2026-05-27)

| Item | Value |
|---|---|
| Best confirmed public score | **0.86** |
| Latest submission | `53064353` (`tinker_plus_1e5_full_005`) |
| Latest submission result | **0.85** |
| Training availability | **None** (no additional GPU budget) |

---

## What was completed

1. **Phase 1 (individual adapters):** completed; best = 0.86 (`kienngx_tinker_adapter`).
2. **Phase 2 (merge search):** completed; all variants 0.84–0.86.
3. **Phase 3 (new public adapter sweep):** completed with CV kernel v21 and preflight guard.
   - Path resolution + validation fixes worked (no EngineDead crash cascade).
   - New candidates from kienngx/huikang were evaluated where mountable.
   - Best CV from recovered merge artifacts was ~0.8517; did not beat 0.86 on LB.

---

## Blocking finding

The promising 0.87-provenance adapters (new kienngx variants and huikang versions) were not practically accessible in our Kaggle execution path:

- multiple candidates resolved to `status=missing` in batch evaluation,
- expected mount paths did not expose usable adapter files for evaluation/submission,
- no reproducible no-train path in our environment yielded >0.86.

Given no training resources, this is now a hard blocker.

---

## Final strategy (no training)

1. Keep **0.86** as the project baseline and safest known target.
2. Avoid additional GPU-heavy experimentation unless it directly reproduces a known >0.86 artifact.
3. Use remaining submission slots conservatively (only if there is clear evidence of improvement).
4. Treat training-dependent paths as blocked until compute is available again.

---

## Historical context

- Phase 1 (individual adapter sweep, May 17-20): evaluated 7 adapters; best = 0.86.
- Phase 2 (merge search, May 20-25): 8 merge variants; none exceeded 0.86.
- Phase 3 (new adapter sweep, May 25-27): infra fixes landed; inaccessible adapter artifacts prevented a reproducible >0.86 no-train result.
- See `docs/archive/vertex_ai_setup.md` for the dead-end Vertex AI path.
