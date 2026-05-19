# 10 — Open Progress Prize Winner Write-up (Tong Hui Kang / tonghuikang)

Published 2026-05-19. This is the first authoritative public disclosure of the
winning approach for the **Open Progress Prize** (top leaderboard share + prize
for the most informative public write-up). The author is **Tong Hui Kang**
(`tonghuikang` on Kaggle / GitHub), who is also the creator of the
`tonghuikang/nemotron` corpus that the entire 0.85 cluster uses as training data.

**LB score**: 0.85 (achieved three times), 0.84 (five times).
**Cost**: ~$212.48 Tinker credits + ~$60 Modal + ~$10 Kaggle/Colab ≈ **$280 total**.
**Training tokens**: 27,850,703 (winning v20 submission).

---

## Core bet

> "Nemotron can act as a simple computer after LoRA training."

The author bets *against* RL and model distillation. His framing:

- **Optimal policy is known**: For solvable problems, the best next token is
  already derivable from a deterministic procedure. SFT to execute that policy is
  sufficient; RL adds complexity without signal.
- **Deterministic CoT design beats diverse CoT**: Because eval temperature is 0.0,
  you just need the model to follow a deterministic trace reliably. Diversity is
  a non-goal.

---

## Training objective: maximize minimum logprob

The objective is **not** average cross-entropy loss. The author monitors the
**minimum logprob** across all tokens in a trace. If `min_logprob < ln(2) ≈ 0.693`,
that token is the "weakest link" — there is a >50% chance the model will
generate a different token at inference, causing failure.

Workflow per iteration:

1. Write/update the Python code that generates completion traces (`reasoning.py`).
2. Run it; diff output before/after.
3. Send traces to Tinker for fine-tuning (`train_sft.py`).
4. Inspect `min_logprob` of each trace; flag any trace with `min_logprob < 0.693`.
5. Find the specific high-loss tokens; redesign the CoT to avoid them.
6. Upload adapter to Kaggle (`upload_adapter.py`).
7. Convert Tinker → submission-format adapter + eval on Kaggle.
8. Submit for public LB score.

---

## CoT design principles

| Principle | Detail |
|---|---|
| **Deterministic** | Temp=0 → maximize min logprob of training traces |
| **Simple** | Each token derived from as few sources as possible; decompose division into subtraction/addition |
| **Coverage** | Iterate through all 77 Wonderland words even if the first match is found — forces training on rare tokens |
| **Length budget** | Full CoT must fit in 7680 tokens |
| **Tokenization-aware** | Avoid constructs where the model is weak on tokenization (symbol groups, letter splits) |
| **Generalizable** | Don't memorize; be aware when the loss going to zero means memorization, not learning |
| **Non-goals** | Diversity, conciseness, self-verification (all unnecessary at temp=0) |

---

## Per-puzzle performance (winning submission v20)

Target solve rates computed by the solver; actual v20 training-set performance
(note: eval on training set — so slightly inflated):

| Category | Training-set | Target (solver) | LB behaviour |
|---|---|---|---|
| numeral | 100.0% (1576/1576) | 100% | ~100% |
| unit_conversion | 99.7% (1590/1594) | 100% | occasionally fails |
| gravity | 99.6% (1591/1597) | 100% | occasionally fails |
| cipher | 99.3% (1565/1576) | 100% | occasionally fails |
| bit_manipulation | 87.2% (1397/1602) | 85.1% | main uncertainty |
| equation_numeric_deduce | 90.8% (541/596) | 90.6% | well-solved |
| equation_numeric_guess | 16.2% (22/136) | 15.4% | hard by design |
| cryptarithm_deduce | 7.1% (47/659) | 8.2% | effectively memorized |
| cryptarithm_guess | 6.7% (11/164) | 6.7% | effectively memorized |
| **TOTAL** | **87.8% (8340/9500)** | **87.7% (8333/9500)** | **0.85 LB** |

Key insight: the gap between 87.7% target and 0.85 LB is the
**training-serving misalignment** caused by the lossy Tinker → PEFT adapter
conversion (see below).

---

## Puzzle-type approaches (in order of difficulty)

### Numeral (100% target)
- Enumerate all 1–100 Roman numerals; generate step-by-step trace for each.

### Unit Conversion / Gravity (100% target)
- Identify the scalar; decompose multiplication/division into addition/subtraction
  steps so each arithmetic token is easy for the model to predict.

### Cipher (100% target)
- Three phases: (1) compute current char mapping from examples, (2) collate
  unmapped characters, (3) assign unmapped characters.
- Key: iterate through **all 77 Wonderland words** even after a match — forces
  training coverage on all words.

### Bit Manipulation (85.1% target)
- The key differentiator vs other competitors. Competitive programming
  background helped reduce "time and space complexity" of the problem.
- Complex operation chains (e.g. `ROT(2) XOR NOT SHL(3)) OR SHR(1)`) are
  out of scope — would exceed 7680-token limit. Simpler combinations are solved.

### Equation Numeric (90.6% deduce / 15.4% guess target)
- 32 operators split into two groups: 8 popular + 24 rare. Match popular first.
- 4 reversals of operands/results: (BA,DC)→YX, (BA,DC)→XY, (AB,CD)→YX, (AB,CD)→XY.
- Unseen operators: assumed absolute difference (+0.2% on 21/9500).

### Cryptarithm (8.2% deduce / 6.7% guess target)
- Reduced to: identify whether question operator is concatenation or reverse
  concatenation. If unseen, assume concatenation (+0.6% on 65/9500).
- **Nemotron is poor at splitting/concatenating symbols.** Perfect performance
  here is described as "a baseline expectation to medal."
- The cryptarithm memorization issue: minlogprob → 0 but the model is merely
  memorizing the training set, not learning a generalizable procedure.

---

## Tinker → PEFT adapter conversion: known issues

The winning adapter was trained on **Tinker** (off-Kaggle GPU service). The
Tinker adapter format is incompatible with the competition submission format:

| Issue | Fix |
|---|---|
| Key prefix `model.*` | Rename to `backbone.*` |
| Fused `experts.w1/w2` LoRAs | Unfuse into per-expert `up_proj` / `down_proj` by broadcasting across 128 experts |
| Separate `gate_proj` + `x_proj` for Mamba | SVD-merge into single `in_proj` (rank 64 → rank 32) |
| `lm_head` LoRA present | Keep it |

**Critical problem**: The SVD merge is **lossy** — only 75% of singular mass is
captured. This causes **training-serving misalignment**:
- Model fails to write `\boxed` at inference
- Degenerate repeated-token responses
- Missing words from the template

Proposed fixes (not yet applied to winning submission):
1. Train with rank 16 on Tinker → losslessly upscale to rank 32. (Got 0.82.)
2. End-to-end SFT without Tinker. (Got 0.81 but 50s/step vs 8s/step on Tinker.)
3. **Post-finetune to align weights**: after converting, SFT on tokens where
   SVD increased the loss. *(Not yet tried — promising.)*

---

## Predicted frontiers for 0.877+

From the author's own analysis of what would be needed to reach his 0.877 target:

1. **Fix training-serving misalignment** — post-finetune after SVD conversion.
2. **Work with tokens, not text** — generate token IDs directly; renders
   tokenization ambiguity a non-issue for ~half the token-related failures.
3. **Fix Nemotron's char ↔ text conversion** — even with tokens in training,
   inference tokenizer is fixed; base model is poor at text→char splits.
4. **Synthetic problem generation + clean train/val split** — currently training
   on all 9500 (contaminated val), no held-out set.
5. **Complex bit manipulation** — ops like `(ROT(2) XOR NOT SHL(3)) OR SHR(1)`
   require a shorter bit representation (e.g. base-64) to fit in the token budget.
6. **Cryptarithm with guessing** — deterministic CoT is insufficient; needs the
   model to make probabilistic guesses, which may require RL or a different
   objective at guessing steps.

---

## Things the author thinks he did well

- **LeetCode / competitive programming background** → better bit manipulation CoT;
  contributed more than half the LB gap vs KhoA's notebook.
- **Looking at the data, not just the loss curve** → per-category loss, per-token
  min-logprob, not just aggregate scalars.
- **Knowing what not to try** → no RL, no distillation.
- **Using Tinker** → 20 min/training run vs 4 hours on Kaggle.

## KhoA note

"KhoA managed to get perfect scores on numeral, gravity and unit_conversion
whereas my winning model still sometimes fails in these categories." — KhoA
(likely the `llkh0a` / `kienngx` 0.86 LB entrant) has better reliability on
the "easy" categories, suggesting their adapter conversion or fine-tuning avoids
the SVD misalignment issue.
