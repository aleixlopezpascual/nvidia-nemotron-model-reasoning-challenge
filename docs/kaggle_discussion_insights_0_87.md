# Kaggle Discussion Insights for 0.87 (No-Train Track)

**Last updated: 2026-05-27**

---

## Current Leaderboard Context

| Rank | Team | Score |
|---|---|---|
| 1-7 | Y\|M\|F, Researcher 7919, Lora is all you need, Kh0a, fate, JK-Piece, Surya milenial | **0.87** |
| 8+ | Multiple teams including us | **0.86** |

**Current project reality**: our best confirmed result remains **0.86**. A later no-train direct submission (`53064353`) scored **0.85**.

---

## High-confidence takeaways

1. **0.86 plateau is real for many teams**
- Multiple strong participants report being stuck at 0.86 and struggling to find +0.01 without a new signal.
- This matches our current state: baseline 0.86, best merge tie at 0.86.

2. **Most 0.87+ narratives are training-heavy**
- The highest-scoring public writeups emphasize substantial SFT work, token budgets, and category-specific data/trace quality.
- There is no widely shared “packaging-only” trick that reliably jumps from 0.86 to 0.87.

3. **Metric behavior changed mid-competition**
- Discussions confirm metric/verify changes and rescoring effects.
- We should continue using the updated host-metric mirror as source of truth for local CV comparisons.

4. **Reproducibility quality in public notebooks is mixed**
- Discussions flag copy/fake notebooks and misleading claims.
- We should trust only reproducible model artifacts + verifiable leaderboard evidence.

5. **Prompt hacking is limited by design**
- Hosts confirmed eval prompt is fixed/hidden to reduce reverse engineering.
- This reinforces our current strategy: adapter/model selection and careful no-train merging, not prompt tricks.

## What this means for our strategy (updated)

Given no additional training resources:
1. **Preserve 0.86 baseline** as the best reproducible target.
2. **Stop speculative no-train sweeps** that do not expose real submission artifacts.
3. **Submit only evidence-backed candidates** (clear CV/LB upside, valid packaged adapter).
4. **Document blocked adapters clearly** (mount path inaccessible / artifact not exposed).

## Practical do / don’t

### Do
- Prioritize adapters with strong provenance (known public LB history).
- Compare per-label performance, not only aggregate score.
- Keep merges small and reversible.

### Don’t
- Chase unverified “0.87 claims” without reproducible artifacts.
- Spend submission quota on marginal CV ties.
- Assume packaging/layout tweaks alone can produce +0.01.

## Concrete conclusion from our run history

- Theoretical 0.87 signal exists in public leaderboard narratives.
- In our reproducible environment, no-train exploration plateaued at **0.86**.
- Without additional compute, pushing beyond 0.86 is currently blocked by artifact accessibility and lack of trainable budget.

## Linked artifacts
- Raw scan notes:  
  `/Users/aleix.lopez/.copilot/workspaces/38940bbc-5af8-40a8-83a7-be5ef59127d1/artifacts/discussions/kaggle_discussion_scan.md`
- Action plan:  
  `docs/next_steps_0_87.md`
