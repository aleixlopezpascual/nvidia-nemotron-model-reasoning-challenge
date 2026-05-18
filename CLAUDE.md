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
nemotron_local_cv/       # Host-metric mirror + CV CLI (laptop-buildable, runs on any GPU)
  host_metric.py         # VERBATIM mirror of metric/nvidia-nemotron-metric (host kernel)
                         # - extract_final_answer (updated 2026-05-18 — handles nested braces)
                         # - verify (binary-string bypass + rel_tol=1e-2 numeric)
                         # - BOXED_INSTRUCTION suffix with backticks (verified vs host)
                         # - generate_predictions / score (parameterized over MODEL_PATH)
  labels.py              # 6-way puzzle-type classifier (leading-clause match)
  splits.py              # build_stratified_val_split(n=600, seed=42)
  __main__.py            # CLI: python -m nemotron_local_cv --adapter ... --val ...

nemotron_train/          # SFT pipeline (convergent public 0.85 recipe + completion_only_loss)
  recipe.py              # LoRA + training defaults (rank 32, lr 2e-4, eff batch 32, bf16)
  data.py                # CoT corpus loader + chat-template formatter (uses BOXED_INSTRUCTION)
  trainer.py             # train(...) — loads base bf16, attaches LoRA, runs TRL SFTTrainer
  __main__.py            # CLI: python -m nemotron_train --data ... --output ...

kaggle/                  # Kaggle-Notebook entry points (RTX PRO 6000 Blackwell, 96GB)
  blackwell_setup.py     # rmsnorm + fast-path + ptxas patches (load-bearing on Blackwell)
  train_kernel/          # Generated kernel: SFT training notebook + kernel-metadata.json
  cv_kernel/             # Generated kernel: local CV notebook + kernel-metadata.json

tests/                   # pytest suite (90 tests, ~0.5s) for the host-metric mirror,
                         # label classifier, val split, CLI roundtrip, SFT data formatter

scripts/
  build_val_split.py            # writes data/val_split_600.csv (checked in)
  download_top_notebooks.py     # round-1 notebook scrape
  download_specific_notebooks.py # round-2 hand-picked refs
  download_training_corpus.py   # pulls kienngx CoT corpus
  sync_code_to_kaggle.py        # packages code as private Kaggle Dataset
  build_kaggle_notebooks.py     # emits kaggle/train_kernel + kaggle/cv_kernel from Py source

data/
  val_split_600.csv      # 600-row stratified val (100 per puzzle type, seed=42)

docs/
  kaggle_workflow.md            # READ FIRST for the train/CV/submit loop
  rules_compliance.md           # what compute/data/models are allowed
  notebook_learnings/00_SYNTHESIS.md  # strategic synthesis of public landscape
  notebook_learnings/01..09_*.md      # per-theme deep dives
  archive/vertex_ai_setup.md    # deprecated — see Compute split below

notebooks/                # downloaded public Kaggle notebooks (gitignored)
notebooks/_host_metric/   # host metric kernel (source of truth for local CV)
```

**Submission contract** (mirrors what the eval layer must respect):
- Submit a LoRA adapter (rank ≤ 32) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` packaged as `submission.zip` with `adapter_config.json` + `adapter_model.safetensors` at the zip root.
- vLLM, greedy (`temperature=0.0`, `top_p=1.0`), `max_model_len=8192`, `max_tokens=7680`. Answer extracted from `\boxed{...}`.
- Eval prompt appends the boxed-instruction suffix (with **backticks**) to every user message before `apply_chat_template(..., enable_thinking=True)`. **Training prompts must match this exactly** — `nemotron_local_cv.host_metric.BOXED_INSTRUCTION` is the source of truth.

**Compute split**: laptop builds, edits, and verifies code (90-test pytest suite runs in <1s). All GPU work — SFT training, local CV via vLLM, teacher-CoT generation — runs on **Kaggle Notebooks with the RTX PRO 6000 Blackwell** accelerator (96 GB VRAM, free within Kaggle's GPU quota, native parity with the host eval kernel). See `docs/kaggle_workflow.md`. We do NOT use Vertex AI (GCP project had no 80GB quota; see `docs/archive/vertex_ai_setup.md` for the dead-end history).

**Kaggle workflow in one line per step**:
1. Edit local code → 2. `uv run python scripts/sync_code_to_kaggle.py -m "..."` → 3. `uv run python scripts/build_kaggle_notebooks.py` → 4. `uv run kaggle kernels push -p kaggle/train_kernel` → 5. Run notebook on Kaggle → 6. `... push -p kaggle/cv_kernel` → 7. Run CV → 8. Submit from training kernel's output via Kaggle UI.

**Tests**: `uv run pytest -q` runs the full 90-test suite (no GPU, ~0.5s). Always run after touching `nemotron_local_cv/` or `nemotron_train/`.
