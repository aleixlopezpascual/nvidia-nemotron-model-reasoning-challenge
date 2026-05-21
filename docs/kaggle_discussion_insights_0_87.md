# Kaggle Discussion Insights for 0.87 (No-Train Track)

Date: 2026-05-20  
Source scan: competition discussions (124 topics scanned, 30 relevant by keyword)

## Why this note exists
Capture the useful signal from Kaggle discussions so future sessions do not repeat noisy exploration.

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

## What this means for our strategy

Given no-training constraints, the best remaining path is:
1. **Broaden candidate adapter mining** (individual public adapters first).
2. **Apply strict CV gating** (overall + weak-label improvement).
3. **Run conservative soups only on top bases** (tiny donor weights).
4. **Use submission slots sparingly** (1–2 high-confidence candidates).

## Practical do / don’t

### Do
- Prioritize adapters with strong provenance (known public LB history).
- Compare per-label performance, not only aggregate score.
- Keep merges small and reversible.

### Don’t
- Chase unverified “0.87 claims” without reproducible artifacts.
- Spend submission quota on marginal CV ties.
- Assume packaging/layout tweaks alone can produce +0.01.

## Linked artifacts
- Raw scan notes:  
  `/Users/aleix.lopez/.copilot/workspaces/38940bbc-5af8-40a8-83a7-be5ef59127d1/artifacts/discussions/kaggle_discussion_scan.md`
- Action plan:  
  `docs/next_steps_0_87.md`
