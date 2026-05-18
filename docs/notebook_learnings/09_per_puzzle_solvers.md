# 09 - Per-Puzzle-Type Solvers & Rule-Augmented CoT Pipelines

This batch of five notebooks zeroes in on what is, by community consensus, the
single highest-leverage data trick in the competition: **a domain-specific,
deterministic Python solver per puzzle type, used to (a) verify teacher-generated
CoT against ground truth and/or (b) inject structural hints (candidate boolean
functions, deduced symbol-to-digit maps, dominant shift patterns, etc.) directly
into the training prompt so the teacher LLM has a fighting chance on the hard
types**.

The 03_cot_and_synthetic_data.md doc already showed *why* CoT + rejection
sampling matters in the abstract. This doc drills into the concrete per-type
solver code that the top public notebooks are actually running. The motivating
fact across all five notebooks is the same teacher-pass-rate table from
konbu17's earlier pipeline:

| Puzzle type             | Teacher pass rate (no solver) | Notes |
|-------------------------|------------------------------:|-------|
| Numeral Conversion (Roman) | 99.6% | Trivial; deterministic Python is 100% |
| Gravitational Constant     | 99.7% | Trivial; rule-based 99.4% |
| Unit Conversion            | 88.7% | Rule-based 99.4% |
| Text Encryption (cipher)   | 94.3% w/ dict | Char-mapping solver 38-90% standalone |
| Bit Manipulation           | 40.3% | konbu17 solver: 75.5%, with synth boost |
| **Equation Transformation**| **13.6%** | **the published gap** |

Three of the six types are essentially solved by a few dozen lines of Python.
The remaining three (Bit Manipulation, Text Encryption, Equation
Transformation) are where the leaderboard is decided, and Equation
Transformation in particular is the type that *no* publicly released notebook
fully cracks. The deepest published attack on the Equation family is the
kimberleyduran "solver-verified cryptarithm CoT v2" notebook, which only handles
the cryptarithm sub-flavour of Equation Transformation and only verifies ~95 of
~823 (~12%) puzzles - leaving ~88% of Equation Transformation untouched by any
solver in this batch.

---

## Equation Transformation (the headline gap)

### What the puzzle looks like

Every Equation Transformation puzzle has the same prefix:

```
In Alice's Wonderland, a secret set of transformation rules is applied to
equations. Below are a few examples:
<XX op XX> = <output>
<XX op XX> = <output>
<XX op XX> = <output>
Now, determine the result for: <XX op XX>
```

The kimberleyduran writeup decomposes the family into two distinct kinds of
puzzle, both sharing the prefix:

- **`equation_numeric_*`** - the 5 characters in each `<XX op XX>` are *real
  digits 0-9* and the `op` slot is one of a small set of arithmetic operators
  written as a *symbol* whose meaning is shuffled. The numbers are real; the
  operator-to-arithmetic mapping is the secret. There are 596 deduce + 136 guess
  rows; the published winner solves 90.6% / 15.4% respectively.
- **`cryptarithm_*`** - the 5 characters are *symbols* (any non-digit), and you
  must deduce both a symbol-to-digit mapping AND an operator-symbol-to-operation
  mapping from the examples. There are 659 deduce + 164 guess rows; the
  published winner solves 8.2% / 6.7% respectively.

The "deduce" vs "guess" suffix is whether the query operator was seen in the
in-context examples (`deduce`, can be uniquely pinned) or unseen
(`guess`, requires assuming one of the 5 base operators - inherently more
ambiguous).

### Operator alphabet

Across both sub-flavours, the operators are drawn from exactly 5 base
operations (see `OPS` in kimberleyduran cell 5):

```python
OPS = [
    lambda a, b: a + b,             # 0: add
    lambda a, b: abs(a - b),        # 1: abs_diff
    lambda a, b: a * b,             # 2: mul
    lambda a, b: a * 100 + b,       # 3: concat       (4-digit string)
    lambda a, b: b * 100 + a,       # 4: rev_concat   (4-digit string)
]
```

Critically, the result length tells you which operator class is feasible: 1-3
digits => add / abs_diff / mul; exactly 4 digits => add / mul / concat /
rev_concat. This is the feasibility prune that makes brute-force solving
tractable.

### Published solvers - what actually exists

Only **one** of the five notebooks attacks Equation Transformation directly,
and only on the cryptarithm sub-flavour:

#### kimberleyduran's brute-force cryptarithm deducer (verbatim port)

A faithful port of Tong Hui Kang's `cryptarithm_deduce.py`
(github.com/tonghuikang/nemotron, Apache 2.0). Architecture:

```python
class Solver:
    def __init__(self, examples, query, unique=True, strict_guess=True):
        # mapping: symbol -> digit, used: set of taken digits,
        # op_assign: operator-symbol -> op_id (0..4),
        # answers: Counter of {answer_string -> times produced},
        # answer_info: {answer_string -> (mapping_snapshot, op_info_snapshot)}.

    def _process(self, idx):
        # Recursive depth-first enumeration over all examples.
        # At each example:
        #   feasibility-prune op_id by result-length (1-3 digits vs 4 digits),
        #   for each candidate (d0,d1,d3,d4) symbol->digit assignment,
        #     for each consistent op_id,
        #       compute result_val = OPS[op_id](d0d1, d3d4),
        #       check len(result_digits) matches output length,
        #       try to consistently assign result symbols to those digits.
        #   recurse. Undo assignments on backtrack.
        # Stops at max_solutions=200.

    def _compute_query(self):
        # Once all examples assigned, plug query symbols through the locked
        # mapping + op_assign, count answer strings produced.
        # Strict guess mode: refuse to answer if the unseen op produces
        # multiple distinct candidate answers.

def solve_problem(data):
    # Front-door entry point. Two fast paths first:
    #   1. If query_op appears in examples and ALL example uses are pure
    #      concat / rev_concat (i.e. output is a literal permutation of input
    #      symbols), short-circuit: emit concat or rev_concat answer
    #      directly without invoking the recursive search.
    #   2. Otherwise filter out the concat/rev_concat examples (those don't
    #      constrain digits) and run the Solver on the arithmetic examples.
    # If unique=True fails, retry with unique=False (some puzzles do NOT
    # require distinct digits per symbol).
```

Two operating points are documented in the notebook:

- `strict_guess=False, unique=True->False fallback` — what actually shipped in
  the v2 dataset. Yields **95** of 823 cryptarithm puzzles verified (~11.5%).
- `strict_guess=True` — refuses ambiguous guess-mode answers. Tighter, drops
  to **91** verified.

Either way, the published cryptarithm coverage is ~11% of the cryptarithm
rows, which is ~11% × 823 / 9500 ≈ 1% of the whole corpus. The remaining 88%
of cryptarithms - and *100% of equation_numeric* - have no published
deterministic solver in this batch.

### Pass rate per Equation sub-type

From the kimberleyduran cell 7 run output (and the winner's writeup
reproduced in cell 1):

| Sub-type | Rows | Winner solve rate | Solver alone (this notebook) |
|----------|-----:|------------------:|-----------------------------:|
| `equation_numeric_deduce`  | 596 | 90.6% | not run (different solver) |
| `equation_numeric_guess`   | 136 | 15.4% | not run (different solver) |
| `cryptarithm_deduce`       | 659 | 8.2%  | ~12% (95 verified / 823 total) |
| `cryptarithm_guess`        | 164 | 6.7%  | combined with above |

The `equation_numeric` family has *no public solver* released in any of the
five notebooks we read. Tong Hui Kang's repo
(github.com/tonghuikang/nemotron) does include `equation_numeric_*` solvers
(see konbu17 v2's "32 operators × 4 pairings + neg format detection" remark),
but those are referenced rather than reproduced inline in the public Kaggle
notebooks examined here. The konbu17 v2 notebook trains on
`equation_numeric.csv` and `cryptarithm.csv` files that were prepared *outside
the published notebook* in `exp024-tong-style-cot-data/type_tong/`, and the
notebook does not show the generation code.

### How the verified Equation CoT is integrated into the prompt

kimberleyduran's pipeline assembles a deterministic 5-step CoT
(cell 6, `generate_cot`) when the brute-force solver verifies its own answer
matches the ground-truth `answer`:

```
We need to decode the transformation rule from the examples.
Each symbol represents a digit 0-9.
The operator in position 3 represents one of: add, abs_diff, mul, concat, rev_concat.
I will put my final answer inside \boxed{}.

After searching, the following assignment is consistent with all examples:
Symbol-to-digit mapping:
  '<sym>' = <digit>
  ...

Operator-to-operation mapping:
  '<op_sym>' = <op_name>
  ...

Verification on the examples:
  <XX op XX> = <out>: <a> + <b> = <result>
  ...

Apply to the query: <XX op XX>
  '<q0>''<q1>' = <a>, '<q3>''<q4>' = <b>
  Operator '<qop>' is <op_name>.
  <calc>
  Translate digits back to symbols: <d> -> '<sym>', ...
  Resulting string: <answer>
```

The teacher LLM is *not* asked to generate the deduction; the deduction *is*
the CoT, and the LoRA learns to imitate the exact narration. Whenever the
solver can't verify, the puzzle is **dropped** rather than fed in with a
broken CoT (the explicit fix vs. the dgxchen 0.83 baseline, where the
unsolvable rows used a `"operator unknown, default to concatenation"`
hallucination as their CoT, training the model to ignore reasoning).

### Reported leaderboard impact

kimberleyduran's "fix only the 65 broken cryptarithm rows" patch on top of
the dgxchen 0.83 recipe yields **+0.01 LB** (0.83 → 0.84) when everything
else is held byte-identical. The verified 95 rows are 12× upsampled to 1,140
(matching dgxchen's hard-category upsample ratio). The author notes that
porting the same pattern to `bit_manipulation` *regressed* by 0.12 because of
CoT structure mismatch with the rest of the corpus, and that adding
synthetic cryptarithms regressed by 0.01 (over-specialization at this rank).

### konbu17 v2 reports on Equation Transformation specifically

In the v2 local eval table (eval_split = 500 samples), Equation Transformation
("ET" row) tops out at **29.6%** even in the best run (`exp026/s012`).
For reference:

| Run | GC | NC | UC | TE | ET | BM | Total |
|-----|---:|---:|---:|---:|---:|---:|------:|
| s005 | 93.8 | 100 | 98.8 | 95.2 | 27.2 | 50.0 | 0.768 |
| s006 | 95.1 | 100 | 97.5 | 97.6 | 29.6 | 70.2 | 0.814 |
| s011 | 95.1 | 100 | 98.8 | 96.4 | 27.2 | 70.2 | 0.810 |
| s012 | 96.3 | 100 | 100 | 96.4 | **29.6** | **78.7** | **0.834** |

Bit Manipulation moved from 50% (s005) → 78.7% (s012) with the solver-driven
synth + priority-upweight recipe, but Equation Transformation barely budged
(27.2% → 29.6%). This is consistent with the published-solver picture:
nobody has cracked the Equation family yet.

---

## Bit Manipulation

The deepest-published solver in the entire batch. Three of the five notebooks
contribute:

- **mohankrishnathalla** - acknowledges 0% from rule-based, deferred to LLM.
- **konbu17 (bit-manipulation-solver-cot-generator)** - the heavyweight: a
  multi-phase per-bit boolean-function analyzer reaching ~75.5% standalone.
- **konbu17 v2 SFT** - consumes the above + synthetic data, reports
  78.7% BM accuracy at the LoRA stage.

### Rule (what the puzzle actually is)

Each puzzle presents 7-10 examples of 8-bit input → 8-bit output, then asks
for the output of a query input. The hidden transformation is structured as
follows (verbatim from konbu17 cell 4):

> Each output bit is independently a boolean function of 1-3 input bits.
> Output bit `i` typically reads from input bit `(i + shift) % 8` with a fixed
> shift value. **shift=0 almost never occurs** (0.5%) - the diagonal mapping
> is rare. Shifts 1-7 are roughly uniform at 14-15% each.

Operation distribution from the ground-truth analysis (used as a prior):

| Op | Freq | # inputs | Op | Freq | # inputs |
|----|------|----------|----|------|----------|
| ID | 40.2% | 1 | NOT | 5.0% | 1 |
| AND | 14.7% | 2 | NAND | 3.4% | 2 |
| XOR | 8.1% | 2 | MAJ | 0.8% | 3 |
| XNOR | 7.3% | 2 | CH | 0.5% | 3 |
| OR | 7.3% | 2 | XOR3 | 0.5% | 3 |
| NOR | 5.8% | 2 | unsolvable | 6.1% | 4+ |

### konbu17's solver (the headline contribution)

The solver runs in **four phases**, escalating from cheap global patterns
down to per-bit brute-force scoring. The key novel ingredient is the
inclusion of the four **asymmetric** 2-input boolean functions (INHIB,
reverse-INHIB, IMPL, reverse-IMPL) that standard symmetric-only solvers miss.
These affect 5.26% of bits across 23% of puzzles.

#### Phase 0: Whole-byte pattern detection

Fast global checks that fit the entire 8-bit output as one of:

```python
try_xor(ex, qb)       # output = input XOR constant
try_rot(ex, qb)       # output = rotate(input, k), k in 1..7
try_perm(ex, qb)      # output[i] = input[perm[i]], with optional NOT
try_rotxor(ex, qb)    # rotation followed by XOR with constant
try_revxor(ex, qb)    # bit-reverse followed by XOR with constant
try_uni2(ex, qb)      # all 8 bits use same 2-input function with shifts (s1,s2)
try_mix(ex, qb)       # some bits are ID/NOT shifts, rest uniform 2-input
```

If any of these matches all examples, return immediately - method tag
`"w_<name>"`.

#### Phase 1: Per-bit candidate enumeration

For each output bit position `op in 0..7`, enumerate every boolean function
of input bits that fits all examples. Three function families are searched:

```python
FUNC_1 = [("ID", lambda a: a), ("NOT", lambda a: 1 - a)]

FUNC_2_ORIG = [   # 6 symmetric 2-input functions
    ("AND", a&b), ("OR", a|b), ("XOR", a^b),
    ("NAND", 1-(a&b)), ("NOR", 1-(a|b)), ("XNOR", 1-(a^b)),
]

FUNC_2_ASYM = [   # 2 asymmetric 2-input functions - the novel addition
    ("INHIB", lambda a, b: a & (1 - b)),    # a AND NOT(b)
    ("IMPL",  lambda a, b: (1 - a) | b),    # NOT(a) OR b  = a -> b
]
# Reverse variants are covered by using ORDERED pairs (i1 != i2).

FUNC_3 = [        # only as fallback when no 1-2 input candidates exist
    ("MAJ", maj of 3), ("CH", choice), ("XOR3", a^b^c),
]
# Final fallback: composite f(g(a,b), c).
```

Per bit, each candidate is `(op_name, input_positions, predicted_value,
complexity)`. The 1-input space contributes 16 candidates, 2-input symmetric
168 (28 pairs × 6 ops), 2-input asymmetric 112 (56 ordered pairs × 2 ops),
plus 3-input/composite fallbacks if needed.

#### Phase 2: Unambiguous resolution + context propagation

```python
# Phase 2a: bits where all candidates agree on the same value -> confirmed.
# Phase 2b: build evidence histograms across all 8 bits:
#   shift_ev = Counter of (input_pos - output_pos) % 8
#   pair_ev  = Counter of ((s1,s2,op_name)) for 2-input candidates
#   perm     = {output_pos -> input_pos} from confirmed 1-input bits
# Phase 2c: for each ambiguous bit, try three disambiguation strategies:
#   - Shift propagation: if dominant shift weight >= 5, expect input from
#     (op + dom_shift) % 8.
#   - Pair propagation: if dominant 2-input pattern has weight >= 8, apply it.
#   - Missing-source fill: if 4+ bits already mapped, ambiguous bits should
#     fill the unused input positions.
```

After Phase 2, label remaining `None` bits as "brute-force candidates"
(method tag `"bf"`).

#### Phase 3: Brute-force scoring

Enumerate all 2^k value combinations for the k remaining ambiguous bits
(typically k ≤ 5). Score each combination with `score_v14`, which rewards:

- High operation prior (ID >> AND >> XOR >> ... >> MAJ).
- Low complexity (1-input > 2-input symmetric > 2-input asymmetric > 3-input).
- Template consistency: bonus for ≤2 distinct ops used across the 8 bits.
- Permutation structure: +35 if the implied input→output map is a perfect
  bijection across 8 positions.
- Shift alignment: bonus if a candidate respects the dominant shift.
- Source coverage: small bonus per distinct input position used.

A top-k consensus tiebreak: if the top combo is within 2 points of the second,
take a per-bit majority vote across the top-5 combos.

#### Reported pass rates

From cell 8 of the konbu17 BM notebook:

| Phase | Coverage | Accuracy |
|-------|---------:|---------:|
| Phase 0 (whole-byte) | small fraction | very high |
| Phase 1-2 (ctx resolved) | majority | ~85-90% (high-confidence rows) |
| Phase 3 (brute-force) | remainder | substantially lower |
| **Overall** | 100% of BM puzzles | **~75.5%** |

The `confidence` field separates "high" (zero brute-force bits) vs "low"
(any bf bits) - high-confidence rows are the cleanest SFT signal.

#### Synthetic puzzle generation

The same notebook generates 3,000 synthetic Bit Manipulation puzzles with
guaranteed-correct answers by reversing the discovery process: sample a
shift (1-7), sample 8 operations per the published prior, generate random
8-bit inputs, compute outputs by applying the rule. The synthetic CoT is
produced by running the *same solver* on the synthetic puzzle (so the LoRA
sees an identical CoT format on real vs synthetic data).

### How the BM solver output is integrated into the training prompt

konbu17 generates a per-puzzle CoT (`generate_cot` in cell 7) that
**embeds the solver internals as the reasoning trace**:

```
I need to find the secret bit manipulation rule from the examples.

Given 8 examples:
  00100110 -> 11011001
  ...
Query: 10110010

Step 1: Analyze each output bit as an independent boolean function of input bits.
  Output bit 0: All 24 candidates agree -> value = 1
    Best match: ID(in[3]) = 1
  Output bit 1: Ambiguous (12 candidates predict 0, 8 predict 1)
    For 0: NOT(in[5]), XOR(in[2],in[6]), AND(in[1],in[4])
    For 1: ID(in[4]), OR(in[0],in[7])
  ...
Step 2: Identify structural patterns.
  Confirmed bit mappings: bit0<-in[3](shift=3), bit2<-in[5](shift=3), ...
  Dominant shift = 3 (weight 18)
Step 3: Resolve ambiguous bits using structural patterns.
  Bit 1: Using shift=3, source = in[4] -> value = 1
  ...
Step 4: Resolve remaining 2 ambiguous bit(s) by scoring all combinations.
  Ambiguous positions: [5, 7]
  Scoring considers: operation priors, complexity, template consistency, shift alignment.
  Bit 5 -> 1
  Bit 7 -> 0
Step 5: Combine all bits to form the output.
  bit0=1(unambig) | bit1=1(ctx) | bit2=0(unambig) | ... | bit5=1(scored) | ...

The answer is 11011001.
```

The training datasets produced are:

| File | Rows | Purpose |
|------|-----:|---------|
| `bit_manipulation_cot_success.csv` | ~75% of 1602 = ~1200 | primary SFT data (solver verified) |
| `bit_manipulation_cot_failed.csv` | ~25% = ~400 | error analysis / external LLM CoT |
| `synthetic_bit_manipulation.csv` | 3000 | augmentation, all known ground truth |

### konbu17 v2 BM result

In the v2 SFT, BM accuracy climbed from 50.0% (s005, no synth) to **78.7%**
(s012, downsampled + 500 synth-including-wrong + priority upweighting of
low-logprob examples). The lift from synthetic + priority upweighting is
+25-29 percentage points on BM alone, the single biggest per-type movement in
the v2 sweep.

---

## Text Encryption (cipher)

### Rule

Each puzzle gives several `<encrypted word> -> <plaintext word>` example
pairs plus a query encrypted string to decrypt. konbu17 calls this puzzle
type "Text Decryption" in the v2 source files. mohankrishnathalla notes that
the cipher is character-level and the puzzle "uses a fixed substitution per
puzzle, with the plaintext vocabulary drawn from a 77-word Alice in
Wonderland list".

### Published solvers

#### mohankrishnathalla's `solve_cipher` (38% standalone)

```python
def solve_cipher(prompt):
    """Character-level substitution from word-aligned example pairs."""
    lines = [l for l in prompt.split('\n') if '->' in l]
    letter_map = {}
    for line in lines:
        parts = line.split('->')
        for cw, pw in zip(parts[0].split(), parts[1].split()):
            if len(cw) == len(pw):
                for cc, pc in zip(cw, pw):
                    letter_map[cc] = pc
    m = re.search(r'decrypt the following text: (.+)', prompt)
    query = m.group(1).strip()
    decoded = ''
    unknown = 0
    for ch in query:
        if ch == ' ': decoded += ' '
        elif ch in letter_map: decoded += letter_map[ch]
        else: decoded += '?'; unknown += 1
    return decoded.replace('?', '') if unknown <= 1 else None
```

Naive character mapping - works only when *every* query character appears in
at least one of the example word pairs. Reported accuracy: **38%**. The
notebook explicitly says "fails when the query contains letters not seen in
the provided examples."

#### konbu17's dictionary-injection trick (94.3% with teacher)

konbu17 doesn't release a solver per se. Instead, the upstream
`doc-to-lora-knowledge-injection-nemotron-3-nano` pipeline injects the
**77-word Alice in Wonderland vocabulary** directly into the prompt before
the teacher LLM reasons. With the vocabulary in-context, the teacher LLM
becomes a 94.3% solver (per the 03_cot_and_synthetic_data.md teacher-pass-rate
table). The injected dictionary plus the example word pairs is enough to
uniquely identify the cipher substitution table in almost every case.

#### konbu17 v2 reproduction

The v2 SFT notebook trains on `text_decryption.csv` from the
`exp024-tong-style-cot-data/type_tong/` dataset, which contains 1,492 verified
puzzles (per the SAMPLES dict in DATA_VARIANT 1). The notebook describes
the reproduction recipe as "77-word dictionary + char-by-char mapping" - i.e.
both the dictionary injection and the per-character substitution narration
are baked into the CoT.

### Pass rates summary

| Method | Pass rate | Source |
|--------|-----------|--------|
| Naive char-mapping (no dict) | 38% | mohankrishnathalla |
| Teacher LLM with dict injection | 94.3% | konbu17 03_cot |
| LoRA trained on dict-CoT (Tong-style) | 96-97% local eval | konbu17 v2 (s006-s012) |

The TE per-type accuracy is essentially "solved" at the 0.83-0.84 LB tier; it
is not a leverage point for further improvement.

---

## Numeral Conversion (Roman numerals)

### Rule

Despite the prompt wording ("In Wonderland, a secret numeral system..."), the
"secret" turns out to be **standard Roman numerals with no modification**
(mohankrishnathalla cell 12, verbatim: "The Wonderland secret is just the
Roman numeral system with no modification at all.").

### Published solver - mohankrishnathalla (100% accuracy)

```python
def solve_roman(prompt):
    m = re.search(r'Now, write the number (\d+)', prompt)
    n = int(m.group(1))
    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    sym = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    result = ''
    for v, s in zip(val, sym):
        while n >= v:
            result += s; n -= v
    return result
```

13 lines. 100% accuracy across all 1,576 Numeral Conversion rows in the
training set.

### konbu17 v2 reproduction

v2 trains on `numeral_system.csv` with a "1-100 Roman numeral table" embedded
in the CoT (the notebook description in cell 0). All four v2 runs (s005, s006,
s011, s012) hit **100.0%** on this type. This is fully solved territory.

---

## Gravitational Constant (Physics)

### Rule

Each puzzle hides a modified gravitational constant `g`. The formula
`d = 0.5 * g * t^2` is always given in the prompt. The puzzle provides several
`(t, d)` example pairs and asks for `d` at a new query `t`.

### Published solver - mohankrishnathalla (99.4% accuracy)

```python
def solve_physics(prompt):
    pairs = re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    # d = 0.5 * g * t^2  =>  g = 2d / t^2
    gs = [2 * float(d) / float(t)**2 for t, d in pairs]
    g = np.mean(gs)
    m = re.search(r'for t\s*=\s*([\d.]+)s', prompt.split('Now,')[-1])
    t = float(m.group(1))
    return round(0.5 * g * t**2, 2)
```

The 0.6% miss rate comes from rounding/format edge cases in the prompt
regex - the underlying formula is exact.

### Distribution

From mohankrishnathalla cell 20: the secret `g` ranges across many distinct
values per puzzle (`phys["secret_g"].nunique()` rather than a fixed
Wonderland constant). So the solver has to extract per-puzzle, not memorize a
table.

### konbu17 v2 reproduction

Uses "integer long division for GC/UC (instead of float operations)" -
trains the LoRA to narrate `2*d/t^2` as a long division rather than a
decimal computation. v2 hits 93-96% across runs (better than baseline LLM,
but slightly worse than the deterministic 99.4% Python solver). Local eval
ceiling: 96.3% (s012). The 4-7 pp gap below the pure Python solver suggests
some headroom on numeric CoT precision.

---

## Unit Conversion

### Rule

A fixed unit-conversion ratio per puzzle. The prompt gives several `X m
becomes Y` example pairs and asks to convert a new measurement.

### Published solver - mohankrishnathalla (99.4% accuracy)

```python
def solve_unit(prompt):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    ratios = [float(o) / float(i) for i, o in pairs if float(i) != 0]
    ratio = np.mean(ratios)
    m = re.search(r'(?:convert the following measurement:|measurement:)\s*([\d.]+)', prompt)
    if not m:
        m = re.search(r'([\d.]+)\s*m\s*$', prompt.strip())
    return round(float(m.group(1)) * ratio, 2)
```

Like physics, the 0.6% miss rate is regex edge cases; the formula is exact.

### Distribution

From mohankrishnathalla cell 20: `unit["ratio"].nunique()` is high - ratios
are continuously distributed, not drawn from a small table. So the solver
needs the per-puzzle extraction, not memorization.

### konbu17 v2 reproduction

Uses the same integer long division trick. v2 hits 97.5-100% across runs.
This type is solved at the LoRA level (100% in s012).

---

## Notebook-level headline contributions

### 1. `kalyankkr/all-6-puzzle-types-decoded-sft-training-data`

- Pure EDA + template-based CoT generation. **No actual solvers.**
- Classifies all 6 types via regex on prompt keywords.
- Produces `train_sft.jsonl` and `train_sft_balanced.jsonl` (oversampling
  hard categories 3× and medium 2×).
- The "CoT" is a *fixed-string template per puzzle type* with the answer
  inlined - e.g. for Equation Transformation:
  ```
  "Let me analyze the transformation rules applied to these equations.
   Looking at each example, I need to identify what algebraic
   transformations are being applied.
   The transformation rules I've identified are applied to the given equation:
   \boxed{<answer>}"
  ```
  This is essentially a `\boxed{}`-format teacher with zero actual reasoning.
- Headline value: the classifier regexes are reusable and the
  oversampling-by-difficulty heuristic is one of the cleanest writeups in the
  comp. But no per-type solver is implemented.

### 2. `mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers`

- The cleanest standalone deterministic-solver writeup. 4 solvers
  implemented: Roman (100%), Physics (99.4%), Unit (99.4%), Cipher (38%).
- Explicitly documents that Roman / Physics / Unit are fully deterministic
  and only Bit Manipulation + "symbol transform" (their name for Equation
  Transformation) require LLM reasoning.
- Reports a "rule-based floor" of 56.3% accuracy with zero training, with
  projected hybrid (rules + 70% LLM on the hard types) reaching ~79.6%.
- Includes a `evaluate_local` framework that mirrors the competition metric
  exactly (boxed regex → last number → last word, with `1e-4` relative
  tolerance for numeric).
- Headline value: the cleanest reusable Python solvers for the three "easy"
  types, plus a usable cipher baseline. **Does not attempt Bit Manipulation
  or Equation Transformation solvers.**

### 3. `konbu17/bit-manipulation-solver-cot-generator`

- The deepest published Bit Manipulation solver. Multi-phase architecture
  (whole-byte → per-bit candidates → context propagation → brute-force scoring)
  with the asymmetric-function discovery (INHIB / IMPL, +5.26% of bits).
- 75.5% standalone solver accuracy. Generates `bit_manipulation_cot_success.csv`
  (~1200 verified rows) + `bit_manipulation_cot_failed.csv` (~400) +
  `synthetic_bit_manipulation.csv` (3000 guaranteed-correct synthetic puzzles).
- The CoT format narrates the solver internals step-by-step (candidate lists,
  shift histograms, brute-force scoring) so the LoRA learns to imitate the
  algorithm rather than guess.
- Headline value: the **single biggest per-type solver investment** in the
  public corpus. This is what konbu17 v2 uses to drive BM from 40% (teacher
  alone) to 78.7% (LoRA after SFT on solver-CoT + synth).

### 4. `kimberleyduran/solver-verified-cryptarithm-cot-v2-dataset`

- Ports Tong Hui Kang's `cryptarithm_deduce.py` (Apache-2.0) verbatim and
  wires it into the data-prep step.
- Verifies 95 / 823 cryptarithm puzzles (~11%). For each verified row,
  emits a deterministic 5-step CoT narrating the deduced symbol→digit and
  operator→operation mappings.
- Diff vs dgxchen baseline: drops the 65 broken concat-fallback rows, replaces
  with 95 verified, upsampled 12× = 1,140.
- Reported impact: **+0.01 public LB (0.83 → 0.84)** with every other
  hyperparameter held byte-identical.
- Negative results documented: synthetic cryptarithms (-0.01), porting the
  same pattern to Bit Manipulation (-0.12, structural CoT mismatch), and
  weight-averaging three 0.84 adapters (-0.01).
- Headline value: the **only** public solver-verified pipeline that touches
  the Equation Transformation family, and quantified leaderboard impact.

### 5. `konbu17/nemotron-tong-style-cot-sft-updated-v2`

- The integration notebook: consumes Tong-style per-type CoT data
  (`/exp024-tong-style-cot-data/type_tong/`) and runs SFT with 1st-place
  hyperparameters (LR=2e-4, batch=64, 1 epoch, linear decay, no NEFTune,
  9 target modules including `lm_head`).
- Adds **priority upweighting**: a 2× upsample of low-logprob examples
  identified by running vLLM `prompt_logprobs=1` over the training set
  (threshold `min_logprob < -0.69`, which is the 50% probability boundary).
  This is the "min logprob detection" step from Tong's pipeline reproduced
  via vLLM rather than the Tinker dashboard.
- 4 data variants tested:
  - V1: downsampling (s005-like): 6,204 rows, **0.768 LB**.
  - V2: V1 + priority dups (s006-like): 0.814 LB.
  - V3: V1 + 500 synthetic-incl-wrong (s011-like): 0.810 LB.
  - V4: V3 + priority dups (s012-like): **0.834 LB** (BM 78.7%, ET 29.6%).
- v1 vs v2 SFT recipe deltas:
  - `target_modules`: v1 used `(in_proj|out_proj|up_proj|down_proj)` only.
    **v2 adds `q_proj, k_proj, v_proj, o_proj, lm_head`** (9 modules total).
  - LR: v1 used 1e-4, **v2 uses 2e-4** (or 2.4e-4 for the s011/s012 runs).
  - `gradient_accumulation_steps`: 8 → **64** (effective batch 8 → 64).
  - `lr_scheduler_type`: cosine + warmup_ratio=0.05 → **linear + warmup=0**.
  - `lora_dropout`: 0.05 → **0.0**.
  - `max_grad_norm`: 1.0 → **1e9** (effectively disabled).
  - `adam_beta2`: 0.999 (default) → **0.95**.
  - `max_length`: 4096 → 8192 desired, **stuck at 4096 on Kaggle (OOM)**.
- What v2 *could not* reproduce from Tong: SVD merging of Mamba `gate_proj`
  + `x_proj` → `in_proj`, full 8192 context (Kaggle OOM forced 4096), Tinker's
  training speed (~8 sec/step vs Kaggle ~60 sec/step).
- Headline value: documents the exact recipe + per-type accuracy under each
  data variant, making it the cleanest "ablation table" in the public corpus.
  **This is the notebook to fork if we want to match 0.834 LB locally on a
  Kaggle-equivalent GPU.**

---

## Recommendation

### Where to invest engineering effort

**Equation Transformation, specifically the `equation_numeric_guess` and
`cryptarithm_guess` sub-families, is the highest-leverage open lever.**

Evidence:

- Winner's solve rate is 15.4% / 6.7% on these two sub-families (per
  kimberleyduran cell 1). Every other sub-type is at 85%+. These two together
  are 300 puzzles ≈ 3.2% of the corpus, but their teacher pass rate is so
  low (~6-15%) that *every* additional verified row is leverage on the
  bottom-of-distribution gap.
- konbu17 v2's BM-vs-ET sweep is the smoking gun: with the same recipe and
  same synth-augmentation lever applied, BM moved from 50% → 78.7% (+28.7
  pp) while ET barely moved (27.2% → 29.6%, +2.4 pp). The lever that worked
  for BM (solver-CoT + synth + priority upweighting) has not been applied to
  ET because **no public Equation Transformation solver exists for the
  `equation_numeric` family**, and the only published Equation solver
  (kimberleyduran's cryptarithm port) covers only ~12% of cryptarithm rows.
- The kimberleyduran negative-result section explicitly warns that
  *synthetic* Equation puzzles regressed by 0.01 - so the lever is
  rejection-sampling, not synth volume. We need a *verifier* that catches more
  of the existing 823 cryptarithm + 732 equation_numeric rows, not more
  generated puzzles.

### Concrete build plan

1. **Equation Numeric solver (new build).** Tong Hui Kang's
   `investigators/equation_numeric_deduce.py` (Apache 2.0,
   github.com/tonghuikang/nemotron) is the obvious starting point - it
   handles "32 operators × 4 pairings + neg format detection" per konbu17's
   description, but is not reproduced inline in any public Kaggle notebook
   we have. Port it the same way kimberleyduran ported the cryptarithm
   solver: read JSONL-style data, run the deducer, drop rows where the
   solver's answer ≠ ground-truth answer, emit a deterministic CoT that
   narrates the deduced operator-to-arithmetic mapping. Target verified
   coverage: at least 50% of 732 = 366 rows.

2. **Cryptarithm solver coverage extension.** Push the published 95 / 823
   (~12%) upward by:
   - Relaxing the digit-uniqueness assumption further (the published
     `unique=True->False` fallback already exists; add explicit per-symbol
     ambiguity tolerance).
   - Adding a constraint-propagation pre-pass (e.g. Z3 / python-constraint)
     to handle puzzles where pure brute-force times out at `max_solutions=200`.
   - Joint deduction across cryptarithm + equation_numeric: if a puzzle's
     "symbols" turn out to be real digits at solve time, fall through to the
     numeric solver. Target verified coverage: 30-40% of 823 = 250-330 rows.

3. **CoT format compatibility.** The kimberleyduran negative-result on
   porting the BM pattern (-0.12 LB) is the warning shot: the new Equation
   CoT must structurally match the rest of the konbu17 v2 corpus
   (`<think>\n...CoT...\n</think>\n\boxed{<answer>}`, no `\boxed{}` inside
   the CoT body, no ASCII bracket markup `【...】` carryover from the
   cryptarithm narration template if that style clashes with the equation
   prompts). Test on a held-out 500-row eval split before SFT.

4. **Apply konbu17 v2 priority upweighting.** Once we have a fresh batch of
   verified Equation CoTs, run vLLM `prompt_logprobs=1` over the SFT corpus
   with the current adapter, identify the low-logprob (< -0.69) Equation
   rows, and 2× upsample them. This is the s006 / s012 trick that gave
   konbu17 v2 its +0.05 LB on BM; it should compound with the verified-CoT
   data refresh.

5. **Synthetic Equation puzzles - explicitly avoid.** Per kimberleyduran's
   ablation, synthetic cryptarithms cost -0.01 LB. The lever is verifier
   coverage on real rows, not generated volume. Skip this.

### Already-solved territory (do not spend cycles here)

- **Numeral Conversion, Gravitational Constant, Unit Conversion.**
  Deterministic Python solvers at 99-100%. konbu17 v2's CoT format
  (integer-long-division narration for GC/UC, 1-100 Roman table for NC) is
  already at the LoRA ceiling. The only marginal gain is the 4-7 pp gap on
  GC at the LoRA level; not worth it.
- **Text Encryption.** konbu17's dict-injection plus 1492-row Tong CoT puts
  the LoRA at 96-97%. The remaining 3-4% is likely teacher-CoT noise on
  edge-case cipher tables; not the high-leverage lever.
- **Bit Manipulation.** konbu17 v2 already gets 78.7% on this. There is
  still ~21% gap to a hypothetical "perfect solver" but the marginal lift is
  smaller than what's available on Equation Transformation. Push BM only if
  Equation Transformation work plateaus.

### Ranking summary

**Strongest published solvers (reuse, don't rebuild):**

1. Numeral Conversion - 100% with 13-line Python (mohankrishnathalla).
2. Gravitational Constant - 99.4% with regex + arithmetic
   (mohankrishnathalla).
3. Unit Conversion - 99.4% with regex + arithmetic (mohankrishnathalla).

**Strong-but-not-complete solvers (reuse + extend at the margins):**

4. Bit Manipulation - 75.5% standalone, 78.7% post-SFT (konbu17). Asymmetric
   boolean functions + per-bit candidate enumeration is novel and reusable;
   solver coverage is the bottleneck, not the SFT recipe.
5. Text Encryption - dict-injection 94.3% teacher / 96-97% LoRA (konbu17).
   Vocabulary injection is the trick; replicate the 77-word Wonderland list.

**Wide open for us to attack:**

6. **Equation Transformation** - only ~12% of cryptarithm rows have a
   published verifier; **0%** of `equation_numeric` rows do. konbu17 v2's
   LoRA tops out at 29.6% on this entire family. **This is where the next
   0.02-0.04 LB lives.**
