# NVIDIA Nemotron Model Reasoning Challenge

Solution for the Kaggle **[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)** (2026-05).

The task: submit a **LoRA adapter (rank ≤ 32)** for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` that solves rule-induction puzzles (bit manipulation, unit/numeral conversion, text encryption, equation transformation) via few-shot in-prompt examples, scored with greedy vLLM decoding on a hidden test set.

**Result:** 0.86 public LB (target was 0.87; see [the retrospective](docs/project_retrospective_2026-05-30.md) for why we stopped there).

## Approach

- **Host-metric-aligned local CV** ([`nemotron_local_cv/`](nemotron_local_cv/)) — a verbatim mirror of the host's scoring kernel (answer extraction, `\boxed{}` handling, prompt suffix) so local validation accuracy tracks the leaderboard instead of diverging from it.
- **SFT training pipeline** ([`nemotron_train/`](nemotron_train/)) — rank-32 LoRA, `completion_only_loss`, data formatting matched exactly to the host's eval prompt template.
- **Kaggle-first compute workflow** ([`kaggle/`](kaggle/)) — all GPU work (training, CV, teacher-CoT generation) runs on Kaggle Notebooks (RTX PRO 6000 Blackwell, 96GB), since that's the only environment with native parity with the host's eval kernel. The laptop only edits/tests code (90-test pytest suite, no GPU needed) and pushes it to Kaggle as a dataset.
- **No-train adapter triage + merging** ([`data/adapter_candidates.csv`](data/adapter_candidates.csv), [`data/merge_experiments.csv`](data/merge_experiments.csv)) — evaluating and weight-merging community-published adapters as a cheaper alternative to full retraining.

See [`docs/notebook_learnings/00_SYNTHESIS.md`](docs/notebook_learnings/00_SYNTHESIS.md) for a synthesis of the public leaderboard landscape, and [`docs/kaggle_workflow.md`](docs/kaggle_workflow.md) for the day-to-day train → CV → submit loop.

## Repository layout

```
nemotron_local_cv/   Host-metric mirror + local CV CLI (laptop-buildable, no GPU needed)
nemotron_train/      SFT pipeline: LoRA recipe, data formatting, TRL trainer
kaggle/              Kaggle-Notebook entry points (train/CV kernels, merge experiments)
scripts/             Kaggle sync, notebook generation, val-split building, packaging
data/                Stratified validation split + adapter/merge experiment manifests
tests/               pytest suite (90 tests, ~0.5s, no GPU)
docs/                Workflow guides, rules compliance notes, per-topic research writeups
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q          # 90 tests, ~0.5s, no GPU required
```

Fetching competition data and submitting require a Kaggle account with the CLI configured (`~/.kaggle/kaggle.json`, not included in this repo):

```bash
uv run kaggle competitions download nvidia-nemotron-model-reasoning-challenge
unzip -o nvidia-nemotron-model-reasoning-challenge.zip
```

GPU-dependent steps (training, local CV via vLLM) are designed to run on Kaggle Notebooks rather than locally — see [`docs/kaggle_workflow.md`](docs/kaggle_workflow.md).

## License

[MIT](LICENSE)
