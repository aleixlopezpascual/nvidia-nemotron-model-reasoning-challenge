# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Kaggle competition: **NVIDIA Nemotron Model Reasoning Challenge** (deadline 2026-06-15). The deliverable is a **LoRA adapter (rank ≤ 32)** for the **Nemotron-3-Nano-30B** base model, packaged as `submission.zip` (must include `adapter_config.json`). The base model is fixed by the host — only the adapter is submitted.

Evaluation: vLLM loads base + adapter, runs greedy decoding (`temperature=0.0`, `max_model_len=8192`, `max_tokens=7680`), extracts the answer from `\boxed{...}` (with fallback heuristics, last numeric value). Score = proportion correct (string match or relative numeric tolerance).

Data (`nvidia-nemotron-model-reasoning-challenge.zip`): rule-induction puzzles. Each row's `prompt` contains in-prompt few-shot examples + the instance to solve; `answer` is the ground truth (string or numeric). `test.csv` in the zip is a tiny sample — the real hidden test is several hundred problems, so **do not overfit to `test.csv`**.

More detail (eval params, prizes, timeline, current standing) lives in the assistant's memory under `.claude/projects/.../memory/`.

## Environment

Managed by **uv** (project venv at `.venv/`). Python pinned via `.python-version` (3.11).

- `uv run <cmd>` — run any command inside the project env (preferred — no activation needed)
- `uv add <pkg>` / `uv remove <pkg>` — manage deps (updates `pyproject.toml` + `uv.lock`)
- `uv sync` — recreate env from lockfile

The `kaggle` CLI is installed as a project dep, so it is **not** on the global PATH. Always invoke it as `uv run kaggle ...`. The user's Kaggle credentials live at `~/.kaggle/kaggle.json`.

## Common commands

```bash
# Competition data
uv run kaggle competitions download nvidia-nemotron-model-reasoning-challenge
unzip -o nvidia-nemotron-model-reasoning-challenge.zip

# Leaderboard & submissions
uv run kaggle competitions leaderboard nvidia-nemotron-model-reasoning-challenge -s
uv run kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge

# Submit
uv run kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission.zip -m "<message>"
```

Note: this Kaggle CLI version uses `-s/--show` for the leaderboard, **not** `--view`.

## Architecture

```
nemotron_local_cv/       # Local CV harness (laptop-buildable, runs on GPU host)
  host_metric.py         # VERBATIM mirror of metric/nvidia-nemotron-metric (host kernel)
                         # - extract_final_answer (updated 2026-05-18 — handles nested braces)
                         # - verify (binary-string bypass + rel_tol=1e-2 numeric)
                         # - BOXED_INSTRUCTION suffix with backticks (verified vs host)
                         # - generate_predictions / score (parameterized over MODEL_PATH)
  labels.py              # 6-way puzzle-type classifier (leading-clause match)
  splits.py              # build_stratified_val_split(n=600, seed=42)
  __main__.py            # CLI: python -m nemotron_local_cv --adapter ... --val ...

scripts/
  download_top_notebooks.py     # round-1 notebook scrape by votes
  download_specific_notebooks.py # round-2 hand-picked refs
  build_val_split.py            # writes data/val_split_600.csv (checked in)

data/
  val_split_600.csv      # 600-row stratified val (100 per puzzle type, seed=42)

docs/
  rules_compliance.md           # what compute/data/models are allowed
  notebook_learnings/00_SYNTHESIS.md  # READ FIRST — strategic synthesis
  notebook_learnings/01..09_*.md      # per-theme deep dives

notebooks/                # downloaded public Kaggle notebooks (gitignored if added)
notebooks/_host_metric/   # host metric kernel (source of truth for local CV)
```

**Submission contract** (mirrors what the eval layer must respect):
- Submit a LoRA adapter (rank ≤ 32) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` packaged as `submission.zip` with `adapter_config.json` + `adapter_model.safetensors` at the zip root.
- vLLM, greedy (`temperature=0.0`, `top_p=1.0`), `max_model_len=8192`, `max_tokens=7680`. Answer extracted from `\boxed{...}`.
- Eval prompt appends the boxed-instruction suffix (with **backticks**) to every user message before `apply_chat_template(..., enable_thinking=True)`. **Training prompts must match this exactly** — `nemotron_local_cv.host_metric.BOXED_INSTRUCTION` is the source of truth.

**Compute split**: laptop builds and verifies code; GPU work (running the 30B base + LoRA in vLLM, training, teacher-CoT generation) runs on Vertex AI. The base model is too large for this laptop.
