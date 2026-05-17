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

Repo is at day-zero — only `main.py` (placeholder), `pyproject.toml`, and the downloaded competition zip exist. There is no training, inference, or submission code yet. When building it out, the structure should reflect the submission contract:

1. **Data layer** — load `train.csv`/`test.csv`, parse prompts, surface in-prompt examples.
2. **Training layer** — produce a LoRA adapter (rank ≤ 32) against Nemotron-3-Nano-30B. Any framework is allowed (HF PEFT, Unsloth, Axolotl, TRL, NVIDIA recipes).
3. **Eval layer** — should mirror the host's eval exactly: vLLM + greedy decoding + `\boxed{}` extraction. Local eval that diverges from this will mislead leaderboard expectations.
4. **Packaging** — `submission.zip` must contain the adapter files including `adapter_config.json`.

The base model is large (30B); training/inference will need GPU compute outside this laptop (host suggests GCP G4 / RTX PRO 6000 Blackwell, but any compute works).
