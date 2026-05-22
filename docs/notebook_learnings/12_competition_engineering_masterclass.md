# Competition Engineering Masterclass — Key Breakthroughs

Source: community synthesis of top-tier writeups (huikang Open Progress Prize + broader gold-medal tier analysis).  
Status: authoritative reference as of 2026-05-22.

> **TL;DR:** The leaderboard was unlocked not by more compute or complex RL, but by three precise engineering choices — deterministic synthetic data, greedy-safe loss design, and post-training SVD compression.

---

## 1. Deterministic SFT Beats Reinforcement Learning

**The trap:** competitors burned thousands of dollars running GRPO/PPO reward loops to "teach" Nemotron logic transformations.

**The insight:** when a puzzle is *programmatically solvable*, you already possess the optimal policy. A deterministic Python script generates 100% correct, step-by-step reasoning traces for every puzzle instance. SFT then becomes pure imitation of a perfect solver — no reward noise, no exploration overhead, no instability.

Huikang's own words: *"I already have the optimal policy for solvable problems. I already know the best next token for the LLM to make. The only job of the LLM is to follow the policy."*

**Practical implication for our setup:**
- GRPO is explicitly NOT needed for the categories we care about.
- Every hour spent on RL reward engineering is an hour not spent improving CoT traces.
- `amanatar v3` confirms: `USE_GRPO=False` with comment "GRPO not helping yet — model needs more SFT first."

---

## 2. Minimum Log-Probability as the Real Training Signal

Standard training optimises **average cross-entropy loss**. But the eval uses **greedy decoding (temperature=0.0)** — which means a single low-confidence token derails the entire output.

**The learning:** track `min(log P)` per trace, not average loss. If any token drops below `log(0.5) ≈ −0.69`, the model may not generate it reliably under greedy decoding.

**Workflow:**
1. Train.
2. Run inference on training set; compute per-token log-probabilities.
3. Find tokens where `log P < −0.69`.
4. Inspect the **context** around those tokens. Usually the issue is a hard arithmetic step or ambiguous tokenisation.
5. Rewrite the CoT trace to simplify that step (e.g. break division into repeated subtraction).
6. Retrain. Repeat.

**Practical implication:** after training completes, don't just look at the loss curve. Check per-token log-prob distribution on the validation set. Weak links in the chain matter more than average loss.

---

## 3. Completion-Only Loss Masking (Binary Token Mask)

Loading raw `(prompt, completion)` pairs into a standard `SFTTrainer` wastes gradient budget on memorising randomly generated prompts.

**The solution:** apply a strict binary mask — `0` on every prompt token, `1` on every CoT + answer token. Gradients are multiplied by this mask, so 100% of parameter updates target *what the model should output*, not *what it is given*.

**Impact:** training loss drops to the 0.0X range much faster; easy-category scores stabilise without catastrophic forgetting.

**In our stack:** `completion_only_loss=True` in `SFTConfig` (TRL's built-in mask). This is already set in `nemotron_train/recipe.py`. **Do not remove it.**

---

## 4. Fused / Chunked Cross-Entropy for VRAM

Nemotron-3-Nano-30B has vocabulary size **V = 131,072**. At full context (S = 8192) and microbatch µ = 4, the unchunked logits tensor is:

```
4 × 8192 × 131072 × 2 bytes (bf16) ≈ 8.6 GB
```

This alone causes OOM on a 96 GB Blackwell, before activations are even considered.

**The fix:** chunked / fused cross-entropy (Unsloth's `linear_cross_entropy`, or `cut_cross_entropy`). Computes loss in sequence chunks — never materialises the full logits matrix. VRAM for the logits step drops from **8.6 GB → < 512 KB**.

**In our stack:** we currently use standard TRL `SFTTrainer` without fused CE. If we see OOM at µ=4 after QLoRA succeeds at µ=1, enabling fused CE is the next lever.

References:
- Unsloth `FastLanguageModel` includes this by default.
- `cut_cross_entropy` library is a standalone alternative.

---

## 5. Post-Training SVD for Training-Serving Alignment (Tinker path)

**Relevant when using Tinker or other training platforms that produce non-standard adapter keys.**

Tinker (huikang's training platform) produces adapters with:
- Key prefix `model.*` instead of `backbone.*`
- Fused expert weights (`experts.w1`, `experts.w2`) instead of per-expert (`experts.{i}.up_proj`)
- Separate `gate_proj` + `x_proj` for Mamba layers instead of the merged `in_proj`

vLLM requires the submission-format key schema. The conversion from Tinker → submission uses **Truncated SVD**:

```
ΔW = B × A  →  SVD  →  keep top-32 singular vectors  →  new (B', A') with rank=32
```

**This step is lossy** (huikang captures ~75% of singular mass). Observed failure modes after lossy SVD:
- Model fails to write `\boxed{}`
- Degenerate token repetition
- Template violations (missing required words)

**Mitigations:**
1. Train at rank 16 on Tinker → lossless upscale to rank 32 (huikang scored 0.82 this way).
2. Post-finetune after SVD to recover alignment on affected tokens.
3. Ali (15th LB) published a successful conversion notebook — reference if needed.

**In our stack (Kaggle-only):** we train with TRL directly onto the competition-compatible key schema. No SVD needed. This is an advantage over the Tinker path.

---

## 6. Per-Category Strategy (huikang's final table)

| Category | huikang target | Notes |
|---|---|---|
| numeral | 100% | Enumerate all 1–100 Roman numerals + step-by-step trace |
| cipher | 100% | Compute mapping → collate unmapped → assign → decrypt |
| gravity | 100% | Scalar lookup + chunked multiplication |
| unit_conversion | 100% | Same as gravity |
| bit_manipulation | 85.1% | Hardest; 32 operators; 2-group matching; most headroom for improvement |
| equation_numeric_deduce | 90.6% | 4 transformation types × 32 operators; well-solved |
| equation_numeric_guess | 15.4% | Hard; assume absolute difference for unknown operators (+0.2%) |
| cryptarithm_deduce | 8.2% | Reduced to concat/reverse-concat detection only |
| cryptarithm_guess | 6.7% | Same limited approach |

**Key for 0.87+:** `bit_manipulation` has the most headroom. huikang says 0.87 is achievable through better bit manipulation CoT without any cryptarithm breakthrough.

**Bit manipulation headroom:** currently 85.1% (1364/1602). Target: 88–90%. The bottleneck is the model's difficulty deciding whitespace (position match) vs `y` (value match, wrong position) tokens in the huikang trace format.

---

## 7. Chain-of-Thought Design Principles (huikang's five rules)

1. **Deterministic** — temperature=0.0 eval means you only need the most-likely token correct at each step. Design traces that have exactly one natural next token.
2. **Simple** — break every step into the smallest atomic operation. Division = repeated subtraction + addition. Matching = iterate all candidates, don't short-circuit.
3. **Coverage** — train on all cases, including rare operators. "I do not stop on the first word, I continue on with the rest" — ensures rare tokens get gradient signal.
4. **Within-length** — full CoT must be under 7,680 tokens. Budget accordingly per category.
5. **Tokenization-aware** — Nemotron is bad at character-level operations. Avoid reasoning steps that require the model to split/concatenate individual characters.

---

## 8. What NOT to do

| Approach | Why it fails |
|---|---|
| GRPO / PPO without prior SFT | Model needs format stability before RL polish; `amanatar v3` confirmed negative |
| Longer CoT traces = better | Inference budget is fixed (9h). Long traces → fewer samples → lower throughput |
| Distilling from larger models | Optimal policy is already known programmatically; distillation adds noise |
| Watching only average loss | Greedy decoding is fragile to single low-confidence tokens; track min log-prob |
| Packing with eager attention | TRL warns: packing requires flash_attn_2 for correctness; Nemotron hybrid needs eager |
| Microbatch > 1 without fused CE | Logits tensor alone can be 8.6 GB; causes OOM before activations |

---

## References

- `10_open_progress_prize_winner.md` — full huikang writeup deep-dive
- `11_memory_and_throughput_bounds.md` — Blackwell 96 GB operating envelope
- `04_top_scoring_0_85_lb.md` — convergent 0.85 recipe details
- `07_novel_training_methods.md` — Muon optimizer, fused CE code snippets
