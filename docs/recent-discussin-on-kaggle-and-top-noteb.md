# Research update: Kaggle discussion + top notebooks (2026-05-27)

## Bottom line

There **is** new activity, but no clear public “secret sauce” beyond the known 0.85→0.87 playbook. The practical gap is now mostly **last-mile reliability** (conversion/alignment, trace quality on hard categories, and variance control), not a totally different method.

Also, your current situation is likely less dire than it feels: your recent submissions already reached **0.86** repeatedly, while the top cluster is **0.87** ([E2], [E1]).

## What changed since your last internal synthesis

Your internal synthesis is strong and current through 2026-05-22, including Tong Hui Kang’s write-up and the SVD misalignment diagnosis ([E10], [E11]).  
New external signals since then are mostly:

1. More teams stacking at **0.87** (many tied entries with recent timestamps) ([E1]).
2. New discussion threads focused on the **“0.86 ceiling”**, operator ambiguity, and eval/runtime friction ([E3], [E4], [E5], [E6]).
3. Hot notebooks shifting toward **replay/min-logprob loops and conversion variants**, while many are still forks/repackaging of kienngx/huikang assets ([E7], [E8], [E9]).

## Kaggle discussion signals (what people are saying now)

### 1) “0.86 ceiling” is a common complaint

In topic **702447** (“How to break 0.86 ceiling”), users report:
- 20K+ synthetic additions did not move past 0.86.
- GRPO on top of SFT also stayed flat at 0.86.
- CoT format changes often reduce score, not increase.
- Possible issue raised: duplicated/generic synthetic traces causing weak generalization ([E4]).

Interpretation: this supports your existing thesis that brute data scaling or naïve GRPO is not enough; the bottleneck is data/traces quality and token-level reliability.

### 2) 0.87 appears saturated but stable

Topic **701761** confirms repeated 0.87 reports and comments that 0.88 needs “something special.” Follow-up comments explicitly call out **cryptarithm** and **long bit-manip CoTs** as remaining pain points ([E5]).

Interpretation: this matches your internal bottleneck map (cryptarithm + complex bit manipulation), not a brand-new frontier.

### 3) Metric change is still relevant context

Topic **698106** confirms the metric extraction fix for `}` handling and community-observed ~1% shifts after the update, plus moderator notes about evaluation non-determinism and private-LB focus ([E6]).

Interpretation: part of score movement near the 0.86/0.87 boundary is noise + extraction edge behavior, not only model quality delta.

### 4) Ambiguous/underdetermined symbolic cases are being actively discussed

Topic **702304** argues some equation/symbolic items are underdetermined when query operators are unseen in examples ([E3], [E12]).

Interpretation: this reinforces why deterministic procedures plateau; some residual error may be irreducible without better priors or targeted heuristics.

## Top notebook landscape update

## What is actually new/high-signal

From current hotness feed:
- `matthewblakeward/autonotebook` (51 votes) pushes a heavy **replay + min-logprob replay** workflow and adds “glyph/byte replay” data ideas ([E7], [E9]).
- `ayomide2000/finding-nemo` (83 votes) appears focused on **Tinker adapter conversion**, including aggressive SVD and explicit `lm_head` handling decisions ([E7], [E9]).
- `mohamedamr992/*` recent notebooks continue **replay-math + Tinker-style MoE tie** experiments and packaging loops ([E7], [E9]).

These are evolutions of known themes (replay, conversion fidelity), not clearly proven 0.88 methods.

### What still looks like repackaging/fork pressure

Sampled notebooks (e.g., `minyam/nemotron-kienngx-v4`, `rohanrk1813/nvidia-comp`) reference many kienngx model versions and Tinker adapter paths directly, suggesting substantial re-use/forking of known artifacts ([E9]).

## Are you “missing something”?

Most likely you are missing **execution discipline on known weak links**, not a hidden algorithm:

1. **Conversion/alignment fidelity** (if any Tinker-style conversion remains in your path): repeated signs that `in_proj` merge/SVD and `lm_head` handling still matter materially ([E11], [E9]).
2. **Hard-category targeting**: cryptarithm + long bit-manip traces remain the commonly acknowledged ceiling drivers ([E5], [E11]).
3. **Replay selection quality over raw volume**: forum and notebook signals both point to priority/min-logprob replay quality, not just adding synthetic count ([E4], [E9]).
4. **Variance-aware eval discipline**: public-LB jitter and metric/extraction edge cases can mask small real gains near 0.86–0.87 ([E6]).

## Immediate, concrete moves (highest ROI this week)

1. Add a **daily hotness sweep** into your notebook ingestion (you currently cover voteCount leaders well, but miss many hot recent kernels) ([E8]).
2. Prioritize **A/B on conversion fidelity** over broad hyperparameter churn (same training data + two conversion paths; compare local CV + repeated LB submissions).
3. Build a **targeted replay bucket** only for low-min-logprob tokens in bit-manip/cryptarithm instead of broad synthetic expansion.
4. Run **replicate submissions per candidate** (small N) when scores are within 0.01 to separate variance from true gain.

## Evidence

- **[E1]** Leaderboard snapshot (2026-05-27): `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/leaderboard_2026-05-27.txt` (lines 4-24).
- **[E2]** Your submission history: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/our_submissions_2026-05-27.txt` (lines 3-20).
- **[E3]** New topics page 1: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/topics_new_page1_2026-05-27.json` (topic IDs 702447, 701761, 702304, 698106, etc.).
- **[E4]** Topic 702447 messages: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/topic_702447_messages_2026-05-27.json`.
- **[E5]** Topic 701761 messages: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/topic_701761_messages_2026-05-27.json`.
- **[E6]** Topic 698106 messages: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/topic_698106_messages_2026-05-27.json`.
- **[E7]** Kernel hotness snapshot: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/kernels_hotness_2026-05-27.csv` (lines 2-21, plus 27).
- **[E8]** Coverage gap vs local notebook cache: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/kernel_coverage_gap_2026-05-27.txt` (esp. hotness/dateCreated sections).
- **[E9]** Sampled notebook content probes: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/notebook_probe_summary_2026-05-27.txt`.
- **[E10]** Internal synthesis baseline: `docs/notebook_learnings/00_SYNTHESIS.md` (round-3 to round-5 updates, lines 7-14; bottleneck/corrections sections lines 95-125).
- **[E11]** Open Progress winner deep dive (SVD misalignment and frontier): `docs/notebook_learnings/10_open_progress_prize_winner.md` (lines 136-165).
- **[E12]** Topic 702304 message archive: `/Users/aleix.lopez/.copilot/session-state/52914a9b-09f4-4ecf-8e2c-f9f5f3db480a/files/research/topic_702304_messages_2026-05-27.json`.

## Direct URLs (for quick access)

- Competition discussion index: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion  
- 0.86 ceiling thread: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/702447  
- 0.87 queue thread: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701761  
- Metric update thread: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698106  
- Underdetermined symbolic thread: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/702304  
- Host metric notebook: https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook
