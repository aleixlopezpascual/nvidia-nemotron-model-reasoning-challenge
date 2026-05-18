# Crossing 0.85 -> 0.86 (and the published 0.87 theory)

This file looks at four public Kaggle notebooks whose titles claim **LB 0.86**
(three) or speculate about **LB 0.87** (one), and asks: **what actually
changes between the convergent 0.85 recipe documented in
`04_top_scoring_0_85_lb.md` and the publicly-shared 0.86 submissions?**

The short, honest answer up front: **none of the three 0.86 notebooks
in this set is a novel 0.86 training recipe**. All three are
*packaging-only* notebooks. Two of them (`mohamedamr992`, `safar1`) are
literally one-cell zip scripts that select between several copies of the
**same upstream public adapter** — `kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1`
— and ship the one that scores 0.86. The third (`afr1ste`) is a careful
audit/decision guide built around exactly the same `kienngx`
"tinker-adapter" artifact and includes a public-score table that confirms
**0.86 is what that one specific upstream package scores**, not what a
novel local training run achieves.

So the 0.85 -> 0.86 delta in the public material is not a training trick;
it is the **discovery and clean packaging of a single upstream adapter**
(uploaded by Kaggle user `kienngx` under the name `tinker-adapter`)
**whose lineage is a Tinker-trained run of Tong Hui Kang's
`nemotron-master` recipe**. Everything else — the LoRA config, the data
distillation, the training loop — is happening *upstream* on Tinker, off
Kaggle, and is not reproduced in any of these notebooks. The 0.86 set's
real contribution is therefore: (a) identifying which of the many public
adapters scores 0.86, (b) packaging it correctly so the eval pipeline can
load it, and (c) cataloguing what does *not* work.

The 0.87 notebook (`johnjanson`) is speculative: it proposes a
**failure-mining + bucket-targeted synthetic curriculum + warmstart
PEFT** pipeline that starts from huikang's adapter v20 (0.85-class) and
fine-tunes further on a hand-coded synthetic puzzle corpus. It is not
labelled with a verified 0.87 score; the title ends with a question mark,
and the kernel does not print a leaderboard number. Its value is the
**explicit theory** of what should push past 0.86.

---

## 1. `mohamedamr992` — `0.86 Adapter Packaging Workflow`

**Claimed LB**: 0.86 (title only — no on-notebook score print).
**Actual work**: pure packaging. One code cell, ~14 lines of Python.

### What the notebook is

A `versions` dict with three keys (`tinker-adapter`, `5e-5`, `1e-5`) that
each point to a Kaggle model path under
`/kaggle/input/models/kienngx/nemotron-nano-30b-trained/triton/.../1`. The
author sets `model_name = "tinker-adapter"` and `shutil.copy`s
`adapter_model.safetensors` + `adapter_config.json` into
`/kaggle/working/submission/`, then `shutil.make_archive` zips it.

### Source of the adapter weights

The selected adapter is:

```
/kaggle/input/models/kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1
```

That is: an upstream model dataset published by Kaggle user **`kienngx`**,
slug `nemotron-nano-30b-trained`, framework `triton`, variation
`tinker-adapter`, version `1`. The notebook's `model_sources`
in `kernel-metadata.json` enumerate the full set of variations the author
trialled — seven `kienngx` variations covering `5e-5`, `1e-5`,
`1e-5-batch-2-workers-8`, `batch2-lr1e-workers8`,
`grad-accum1-workers8-maxlen-4096-lr1e-5`, `tinker-adapter`,
plus `huikang/nemotron-adapter/Transformers/default/20`. Of those, the
one chosen for the 0.86 zip is `tinker-adapter`.

### Deltas vs. the 0.85 recipe

- **No training is performed here.** There is no LoRA config, no
  `target_modules`, no optimizer, no learning rate, no scheduler, no
  data preprocessing, no chat-template construction, no `train.csv` or
  CoT corpus access. Every hyperparameter discussion in `04_*.md` is
  inapplicable.
- **No packaging trick** beyond the canonical one: copy two files,
  `shutil.make_archive`. There is no key rename
  (`base_model.model.lm_head.*` -> `base_model.model.backbone.lm_head.*`),
  no rewrite of `adapter_config.json`'s `base_model_name_or_path`, no
  `inference_mode` flip. The author trusts that whatever `kienngx`
  uploaded is already correctly named and configured for vLLM.
- **The only "delta"** vs. the 0.85 recipe is **the choice of source
  adapter**: instead of training their own from huikang's CoT corpus on
  Kaggle, the author downloads `kienngx`'s pre-trained `tinker-adapter`
  artifact and ships it.

### What this notebook tells us

That `kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1` is the
single artifact under that namespace that scores 0.86 — and that several
other `kienngx` variations (the various `1e-5` / `5e-5` / `batch2` /
`grad-accum1` / `maxlen-4096` learning-rate sweeps) **do not** score 0.86
in the same wrapper, otherwise the author would have shipped those.
Combined with `afr1ste`'s score table (below), this strongly implies the
`tinker-adapter` is itself the result of a separate, off-Kaggle **Tinker
training run**, distinct from the on-Kaggle hyperparameter sweeps in the
other `kienngx` variations.

---

## 2. `afr1ste` — `Nemotron 0.86 Tinker Adapter Guide`

**Claimed LB**: **0.86** (stated in markdown, backed by a published
score-anchor table).
**Actual work**: still packaging-only, but with a detailed audit
pipeline, header-only safetensors inspection, layout verification, and
— uniquely — a public score table from the author's own submission
history. CPU-only kernel (`enable_gpu: false`).

### What the notebook is

A six-step workflow:
1. Recursively scan `/kaggle/input` for any directory that contains both
   `adapter_config.json` and `adapter_model.safetensors`, and **prefer
   paths whose name contains `tinker`** (the author's sort key:
   `('tinker' not in str(item[0]).lower(), -item[1])`).
2. Pretty-print the selected adapter's PEFT config (`r`, `alpha`,
   `target_modules`, `task_type`).
3. Read **only the safetensors header** (`int.from_bytes(f.read(8),
   'little')` + JSON parse) to confirm tensor names and counts per
   target module — without loading the 30B model.
4. Build `submission.zip` with the two required files **at the zip root**
   (the markdown calls this a "hard contract") using
   `zipfile.ZipFile(..., compression=ZIP_DEFLATED, compresslevel=6,
   allowZip64=True)`.
5. Re-open the zip and `raise RuntimeError` if any required file is
   missing or if any path contains a `/` (i.e. if the adapter ended up
   nested inside a folder rather than at the root).
6. Markdown "decision checklist" for vetting future candidate adapters.

### Source of the adapter weights

`kernel-metadata.json` lists exactly **one** model source:

```
kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1
```

The same artifact `mohamedamr992` ships. The `find_adapter_dirs` sort
key is explicitly written to bubble "tinker" paths to the top of the
list, which means the author treats `tinker-adapter` as the canonical
choice when present in `/kaggle/input`.

### The score-anchor table (verbatim quotes from the notebook)

This is the most useful artifact in the notebook — a column of public LB
scores from the author's audit trail:

| Candidate family | Public score | Lesson |
|---|---:|---|
| Kien Tinker adapter package | **0.86** | "Current public baseline to protect" |
| Huikang v20 converted adapter | **0.85** | "Valid package, but below the best" |
| Sluitel broad adapter with best `lm_head` fill | **0.62** | "Structural compatibility alone is not enough" |
| Error1249x broad adapter with best `lm_head` fill | **0.59** | "Another valid but task-misaligned broad adapter" |
| Rick v41 full adapter replacement | **0.57** | "Full public replacements can collapse" |
| BharatMohan v31 full adapter replacement | **0.62** | "Same warning as above" |
| Tiny positive public-donor grafts | **0.86** | "Some perturbations preserve score, but did not improve it" |
| Takeshi verified-focus positive grafts | **0.86** | "Sparse inproj/attention direction tied but did not beat 0.86" |

Two things to note:
- **Huikang v20 scores 0.85, Kien Tinker adapter scores 0.86.**
  That's a +0.01 delta between two adapters that both come out of the
  `nemotron-master` lineage — one packaged from huikang's own Kaggle
  uploads (v20), the other packaged from kienngx's Tinker run.
- **"Tiny positive public-donor grafts" and "Takeshi verified-focus
  positive grafts" both score 0.86** — i.e. **no perturbation of the
  base 0.86 adapter has yet beaten it**, including hand-targeted graft
  experiments on `in_proj`/attention directions. The 0.86 ceiling
  appears to be *robust* to small donor edits, but also *unmoved* by
  them.

### Deltas vs. the 0.85 recipe

- **Still no training.** Same as `mohamedamr992`.
- **The new operational lesson**: zip layout matters as a hard contract.
  Files must be at the **root** of the archive (`arcname=path.name`,
  no enclosing directory). The author's checklist explicitly warns that
  "valid LoRA packages can still be very poor public submissions" if
  they're nested under a folder, contain extra files, or have wrong
  namespace.
- **`allowZip64=True`** is called out as required for multi-GiB adapter
  packages (the safetensors file is large enough to trip the 4 GiB Zip
  limit).
- **Structural compatibility is necessary but not sufficient.** The
  table directly anchors this: Sluitel/Error1249x/Rick/BharatMohan all
  produce *valid* PEFT packages but score 0.57–0.62 because the
  underlying LoRA weights are task-misaligned. So the 0.85 -> 0.86
  delta is **not about widening `target_modules`**; the failed
  candidates also have broad target sets.
- **`lm_head` "fill" experiments don't recover score** when the donor
  LoRA is task-misaligned. The author quotes the recipe "best `lm_head`
  fill" applied to Sluitel/Error1249x's broad adapter and the result is
  still 0.59–0.62. This is direct counter-evidence to the naive read of
  the 0.85 recipe that "adding `lm_head` to target_modules is what
  buys you 0.85".

### What this notebook tells us

The 0.86 set's contribution is **provenance discovery, not training**.
The author candidly says so:

> "This notebook adds a careful packaging, audit, and decision guide
> around the public artifact so that others can reproduce the submission
> shape and understand why several tempting alternatives underperformed."

And the closing markdown is explicit that the path forward is **not**
more packaging:

> "The best next step is not to keep resubmitting this same package; it
> is to use this package as a stable baseline while searching for a
> genuinely new data or adapter signal that can move beyond 0.86."

---

## 3. `safar1` — `[LB] [SCORE] [0.86]`

**Claimed LB**: 0.86 (title).
**Actual work**: one cell, ~30 lines, fourteen `version_N =` paths.

### What the notebook is

The notebook defines `version_1` through `version_14`, each a path to a
different `kienngx/nemotron-nano-30b-trained` artifact (mixing the
`transformers/` and `triton/` framework subdirectories). It then sets
`model_path = version_14` (the `triton/tinker-adapter/1` path) and
invokes the shell `zip -r submission.zip "<model_path>"` via Python's
`os.system`.

That is: a recursive zip of the entire adapter folder, so the resulting
archive contains the full Kaggle input path tree
(`kaggle/input/models/kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1/...`)
nested inside the archive. Note that this **violates the "root layout"
rule** that `afr1ste` makes a hard contract — yet the title still claims
0.86, so either (a) the eval pipeline tolerates nested layouts, or (b)
this submission actually scored lower and the title is aspirational.
(`afr1ste`'s table doesn't list `safar1`, so this can't be cross-checked.)

### Source of the adapter weights

`version_14` is the same `kienngx`/.../`triton/tinker-adapter/1` that
the other two 0.86 notebooks ship. `kernel-metadata.json` enumerates
the full menu of `kienngx` variations the author tested — same as
`mohamedamr992`'s seven Triton variations, plus an additional set of
Transformers-format variations (`600-samples-packing-false`,
`1800s-lora-rank32-false`, `2400-1e-4_lr-all_linear-packingfalse`,
`9500s-batch1-lr1e-4`, `cot-labels-3000samples`, `1200samples-cot-`,
`1200samples-cot-5e-5`, `1200samples-cot-1e-5`, `1800s-32`).

The variation names imply that **`kienngx` ran a substantial
hyperparameter sweep**: dataset sizes 600 / 1200 / 1800 / 2400 / 3000 /
9500, learning rates 1e-5 / 5e-5 / 1e-4, with-and-without packing,
with-and-without CoT labels, micro-batch 1 vs 2, dataloader workers 4
vs 8, max_len 4096 — and the one that ended up at 0.86 is none of those,
it is `tinker-adapter`. **The Tinker run beat the in-Kaggle sweep.**

### Deltas vs. the 0.85 recipe

- **No training.** Same packaging-only story.
- **Layout difference**: ships the adapter inside the full Kaggle input
  path tree (recursive shell zip of the directory). If 0.86 really is
  the score, the eval pipeline accepts nested layouts.
- **Has nothing of its own to offer on the recipe side.** The notebook
  is a one-line `model_path` switcher and a recursive zip.

### What this notebook tells us

It corroborates `mohamedamr992` and `afr1ste`: **`kienngx`'s
`tinker-adapter` is the 0.86 artifact**, and a wide sweep of
on-Kaggle hyperparameter variations (`kienngx`'s own
`transformers/...` and `triton/...` variations) **does not match it**.
The fourteen-version menu reads like a slow process of triaging which
upstream `kienngx` artifact is best, and concluding that the
Tinker-trained one wins.

---

## 4. `johnjanson` — `AGI FOR MEDAL, 0.87 IS POSSIBLE?`

**Claimed LB**: **not claimed.** The title ends with a question mark
("0.87 IS POSSIBLE?"), and the notebook does not print an LB score.
This is a **proposal pipeline**, not a verified 0.87 submission.

### What the notebook is

A seven-stage pipeline (all toggled by `RUN_*` flags, defaults `True`):

1. **Failure mining** on `train.csv`: bucket every prompt by a
   hand-coded classifier (`prompt_bucket`) into one of six puzzle
   types (`bit_like`, `unit_like`, `numeral_like`, `equation_like`,
   `cipher_like`, `other`), fingerprint it (length / digit count /
   alpha count / feature flags), and — optionally, if a predictions
   CSV is provided — compute per-bucket accuracy. Otherwise estimate
   a `failure_weight` heuristically as
   `examples * 0.4 + unique_fingerprints * 3.0`.
2. **Bucket-targeted synthetic data generation**: hand-coded
   generators (`gen_bit_like`, `gen_cipher_like`, `gen_unit_like`,
   `gen_numeral_like`, `gen_equation_like`) that emit synthetic
   puzzles with an answer and a **short, deterministic chain-of-thought
   trace** like:
   `"The rule reverses the bit string. Applying it to {target} gives \boxed{answer}."`
   Default budget: **12,000** synthetic examples,
   allocated across buckets in proportion to the failure weights.
3. **Curriculum export**: write three JSONLs (`real_curriculum.jsonl`,
   `synthetic_curriculum.jsonl`, `final_curriculum.jsonl`). The "real"
   curriculum **oversamples hard rows 4x** (`REAL_HARD_OVERSAMPLE = 4`
   when `hard_score >= 80th percentile`) and produces both a
   bare-answer record (`\boxed{answer}`) and a one-line "short trace"
   record per problem. The "short trace" template, when
   `TRACELESS_SHORT_REASONING=True`, is:
   `"I identify one rule that matches all examples, verify that the
   same rule is consistent across the prompt, then apply it to the
   target. Final answer: \boxed{...}."` — i.e. a stub trace, not a
   real CoT.
4. **Warmstart PEFT training**: `prepare_model_for_kbit_training` +
   `PeftModel.from_pretrained(model, WARMSTART_ADAPTER_DIR,
   is_trainable=True)` from `huikang/nemotron-adapter/transformers/default/20`,
   followed by `Trainer` with the configs below.
5. **Local eval** via vLLM with `LoRARequest`, greedy, max_tokens 7680.
6. **Submission packaging**: copies the trained adapter, opens the
   reference `submission.zip` from
   `huikang/nvidia-nemotron-all-linear/reference/submission.zip`, and
   **lifts a specific set of keys from the reference's
   `adapter_config.json`** into its own:
   `['alora_invocation_tokens', 'arrow_config', 'ensure_weight_tying',
   'peft_version', 'qalora_group_size', 'target_parameters',
   'use_qalora']`. Then zips the two adapter files at root.

### Deltas vs. the 0.85 recipe (proposed, not verified)

- **Warmstart from huikang v20**, not from scratch. This is the only
  notebook in the 0.86/0.87 set that *continues* training a published
  adapter rather than packaging it.
- **4-bit NF4 quantized base model** during training
  (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`).
  This is a hard divergence from the 0.85 recipe, which is unanimous
  on `load_in_4bit=False, load_in_8bit=False, dtype=torch.bfloat16`
  (pure bf16). The 0.85 notebooks explicitly avoid quantization
  because the recipe is sensitive to bf16 numerics on the MoE router.
  Using QLoRA-style 4-bit may be necessary to fit huikang's v20
  adapter + full base + Trainer state on a single Kaggle GPU after
  warmstart, but it's a recipe regression.
- **`paged_adamw_8bit`** optimizer — the 0.85 recipe is unanimous on
  full-precision AdamW for the genuine 0.85 runs; only mayukh18's
  non-CoT baseline uses 8-bit Adam.
- **Cosine LR schedule with 5% warmup**
  (`lr_scheduler_type='cosine', warmup_ratio=0.05`) vs. the 0.85
  recipe's **linear decay, zero warmup**.
- **`MAX_STEPS=240`** with effective batch
  `PER_DEVICE_BATCH_SIZE=1 * GRAD_ACCUM=16 = 16`. The 0.85 recipe
  trains 1000 steps at effective batch 32. This run is roughly
  4-5x less optimization compute than the canonical 0.85 run, on top
  of a warmstart adapter — i.e. it is a *fine-tune of a fine-tune*.
- **`MAX_LENGTH=6144`** vs the 0.85 recipe's 8192.
- **Training data is mostly synthetic** (12,000 generated puzzles +
  9.5k real with 4x oversampling on the hard tail). The 0.85 recipe
  trains on huikang's CoT-distilled corpus from a stronger teacher;
  this notebook trains on **hand-coded puzzle generators** that emit
  a one-line rule explanation as the "CoT". This is qualitatively a
  much weaker teacher signal.
- **No `target_modules` is set** — `PeftModel.from_pretrained` inherits
  the config from huikang v20.
- **`build_user_prompt` adds a slightly shorter suffix**
  (`"\nPlease put your final answer inside \\boxed{}."`) vs. the
  0.85 recipe's full
  `"\nPlease put your final answer inside ` + "`\\boxed{}`" +
  `. For example: ` + "`\\boxed{your answer}`" + `"`.
  Whether this matters at inference is untested.
- **Packaging trick**: lifts seven specific keys from a reference
  `adapter_config.json` (`alora_invocation_tokens`, `arrow_config`,
  `ensure_weight_tying`, `peft_version`, `qalora_group_size`,
  `target_parameters`, `use_qalora`). This implies the author has
  observed that the eval pipeline's PEFT loader cares about these
  fields and that getting them wrong silently degrades the loaded
  adapter. Not documented elsewhere in the 0.85 set.

### The 0.87 theory — key claims (verbatim where possible)

The notebook's central theory is **failure-mining + bucket-targeted
synthetic curriculum + warmstart from huikang v20**. The notebook does
not write out a long thesis; the theory lives in the code's structure.
The claims it makes by construction are:

1. **The remaining errors at 0.85/0.86 are concentrated in
   identifiable puzzle buckets.** The `prompt_bucket` function commits
   to the hypothesis that there are six exhaustive categories
   (`bit_like`, `cipher_like`, `unit_like`, `numeral_like`,
   `equation_like`, `other`) and that **failures are non-uniform
   across buckets**. The `bucket_report` is sorted by
   `failure_weight` so the per-bucket fix budget is proportional to
   where the model fails.
2. **The fix is more in-distribution training examples for the
   failing buckets, not a smarter teacher.** Synthetic generators
   emit *programmatically correct* puzzles with *programmatic*
   answers — there is no stronger LLM in the loop. The implicit
   bet: at 0.85/0.86, the model knows how to box answers and how to
   reason, but it has under-seen the *specific puzzle templates*
   that are failing, and exposure (with deterministic short rule
   explanations) is enough to lift accuracy on them.
3. **Hard real examples must be oversampled.**
   `REAL_HARD_OVERSAMPLE = 4` on the 80th-percentile-hardest real
   rows; each hard real row also appears in a "short trace" form
   in addition to the bare-answer form.
4. **Warmstart from the strongest available public adapter
   (huikang v20) and fine-tune at low compute** — `MAX_STEPS=240`,
   not a fresh 1000-step training. The bet is that the public 0.85
   adapter is already most of the way there and that a small,
   bucket-targeted nudge can move the needle without catastrophic
   forgetting.
5. **`TRACELESS_SHORT_REASONING=True` works better than empty
   `\boxed{...}` targets.** The flag controls whether the real-row
   "trace" record is the one-line "I identify one rule…" stub or
   just `\boxed{answer}`. Default is the stub. The implicit claim:
   even a stub trace is a better training signal than a naked boxed
   answer, but the stub does not need to actually solve the
   puzzle — it just has to look like a CoT-ish answer.

### Direct quotes from the notebook

The cleartext markdown at the top of the notebook is mostly garbled
encoding (`"???? notebook ??? ????? ?????:"`), so there is little prose
to quote. The two clearest theory-bearing strings in the code are:

- The traceless-short-reasoning stub itself:

  > `"I identify one rule that matches all examples, verify that the
  > same rule is consistent across the prompt, then apply it to the
  > target. Final answer: \\boxed{...}."`

- The synthetic trace pattern (e.g. for `bit_like`):

  > `f'{expl} Applying it to {target} gives \\boxed{{{answer}}}.'`

  where `expl` is one of `"The rule flips every bit independently."`,
  `"The rule reverses the bit string."`, `"The rule rotates the bit
  string left by N positions."`

The single titular claim is the kernel title itself:

> "AGI FOR MEDAL, 0.87 IS POSSIBLE?"

### Experimental evidence backing the 0.87 theory

**None published in the notebook.** There is no LB number printed, no
local accuracy comparison, no before/after table, no ablation. The
pipeline runs end-to-end but the kernel is uploaded as a *recipe*, not
a result. The `RUN_LOCAL_EVAL` cell would print `'local accuracy ='`
if executed, but the published kernel does not include that output.
This is the largest weakness of the notebook: the theory is plausible
but completely unvalidated.

### What this notebook tells us

It is the **only** notebook in this batch that proposes a concrete
mechanism for crossing 0.86. The mechanism is: **partition errors by
puzzle bucket, generate synthetic in-distribution data for the failing
buckets, warmstart from the best public adapter, train briefly with a
weaker (4-bit, 8-bit-Adam, cosine, 240-step) recipe that respects the
warmstart**. Whether it works is unknown.

---

## Synthesis — what actually changes from 0.85 to 0.86, and what doesn't

1. **The 0.85 -> 0.86 jump in the public material is not a new training
   recipe.** All three "0.86" notebooks are *packagers* around the same
   upstream adapter (`kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1`).
   None of them performs any training. The recipe that produced the
   0.86 weights lives somewhere upstream (Tinker, off Kaggle, by the
   `kienngx` account), and the recipe is not published in any of these
   notebooks.

2. **The single named secret is the artifact `tinker-adapter` under
   `kienngx/nemotron-nano-30b-trained/Triton/...`.** Three independent
   notebooks (`mohamedamr992`, `afr1ste`, `safar1`) converge on
   selecting exactly this variation out of a dozen+ available
   variations under the same `kienngx` namespace. If you want to *use*
   the 0.86 score, this is the adapter to fetch. The other
   `kienngx` variations from the on-Kaggle hyperparameter sweep
   (`5e-5`, `1e-5`, `batch2-lr1e-workers8`,
   `grad-accum1-workers8-maxlen-4096-lr1e-5`, the `*samples-cot-*`
   family, the `*samples-batch1-lr1e-4` family, etc.) **do not** score
   0.86 — otherwise the authors would have shipped those instead.

3. **The 0.85 -> 0.86 delta is plausibly +0.01 from a Tinker training
   run, not from any Kaggle-side change.** `afr1ste`'s public-score
   table shows huikang's own v20 adapter (the published artifact of the
   on-Kaggle 0.85 recipe) scoring exactly **0.85**, while the
   `kienngx` Tinker adapter scores **0.86**. Both are downstream of
   the same `nemotron-master` lineage. The most natural reading is that
   the Tinker pipeline (which `huikang/nemotron-master` is built for)
   produces a slightly better adapter than the on-Kaggle replica of
   the same recipe — possibly because Tinker is what the corpus was
   originally distilled with, possibly because Tinker can run longer /
   on better hardware. Either way, the +0.01 is a *training-environment*
   delta, not a *recipe* delta.

4. **Wider `target_modules` is NOT what buys 0.86.** `afr1ste`
   explicitly documents that "broad adapters" (Sluitel, Error1249x,
   Rick v41, BharatMohan v31) all have the wide LoRA target set but
   score **0.57-0.62** because the underlying weights are
   task-misaligned. Structural compatibility is necessary but not
   sufficient.

5. **`lm_head` fill / graft tricks don't beat 0.86.** Same table:
   "tiny positive public-donor grafts" and "Takeshi verified-focus
   positive grafts" both land at **0.86 exactly**, not above. So the
   public material has spent some effort trying to perturb the
   `tinker-adapter` toward 0.87 with directed `lm_head` and
   `in_proj`/attention grafts and has not succeeded. **The 0.86
   ceiling appears genuinely robust to small public-donor edits.**

6. **The eval pipeline accepts at least two zip layouts.**
   `mohamedamr992` and `afr1ste` ship the two adapter files at the
   zip root; `safar1` ships the full Kaggle input path tree (recursive
   shell zip of the directory). If all three really score 0.86, then
   root-vs-nested is not a contract violation — but `afr1ste`'s
   checklist treats root layout as a hard requirement, so the safer
   answer is to ship at the root.

7. **`allowZip64=True` matters** for adapter zips that exceed 4 GiB
   (relevant once `lm_head` LoRA is included). `afr1ste` calls this
   out explicitly; the other 0.86 packagers don't, but their
   `shutil.make_archive` / shell `zip` paths handle it transparently.

8. **The 0.87 theory (johnjanson) is the only constructive proposal
   for crossing 0.86 in this batch.** Its claim is:
   *partition train.csv errors by puzzle bucket, generate
   bucket-targeted synthetic puzzles (12k examples by default),
   oversample the hard real tail 4x, warmstart from huikang v20, and
   fine-tune briefly (240 steps, lr 2e-4, cosine, 5% warmup) with
   4-bit quantization + paged_adamw_8bit.* The notebook gives
   **no leaderboard evidence** that this works — the title is a
   question. Its strongest single artifact is the
   `prompt_bucket` classifier and the per-bucket
   `failure_weight`-allocated synthetic budget; the rest of the
   pipeline is plumbing.

9. **The 0.87 theory directly contradicts the 0.85 recipe on three
   axes** — quantization (4-bit NF4 vs bf16), optimizer
   (paged_adamw_8bit vs full-precision AdamW), and scheduler (cosine
   + 5% warmup vs linear + 0 warmup). These regressions are most
   likely VRAM concessions for fitting warmstart + 4-bit base +
   Trainer state on a single Kaggle GPU, not deliberate recipe
   choices. They are also exactly the choices `04_*.md` flags as
   correlated with *failing* to reproduce 0.85.

10. **The honest summary of crossing 0.85 -> 0.86 in public**: it
    requires *not* a better Kaggle training run, but **finding and
    correctly packaging the `kienngx` Tinker-trained adapter**. The
    real recipe for that adapter is upstream (`kienngx` on Tinker,
    presumably running a version of `tonghuikang/nemotron`'s recipe
    end-to-end on better hardware than Kaggle). Until someone
    publishes that upstream Tinker run, the on-Kaggle ceiling for an
    independently-trained adapter remains ~0.85. The 0.87 leader has
    presumably either (a) discovered an even stronger upstream
    adapter, (b) trained their own off-Kaggle, or (c) implemented
    something like johnjanson's bucket-targeted curriculum and
    measured it works — but none of that is in the public material
    reviewed here.
