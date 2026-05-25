#!/usr/bin/env python3
"""
Render a compact Kaggle execution state table for merge notebooks.

Example:
    uv run python scripts/kaggle_state_table.py \
      --owner aleixlopez \
      --kernel nemotron-eval-merge-1 \
      --kernel nemotron-eval-merge-2
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

DEFAULT_COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
DEFAULT_OWNER = "aleixlopez"
DEFAULT_KERNELS = [f"nemotron-eval-merge-{i}" for i in range(1, 9)]


@dataclass
class CommandResult:
    code: int
    output: str


def run(cmd: list[str]) -> CommandResult:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return CommandResult(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))


def parse_spaced_table(text: str) -> list[dict[str, str]]:
    """
    Parse Kaggle CLI tables with columns separated by 2+ spaces.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    header_idx = -1
    for i in range(len(lines) - 1):
        if re.fullmatch(r"[- ]+", lines[i + 1]):
            header_idx = i
            break
    if header_idx < 0:
        return []

    header_line = lines[header_idx]
    data_lines = lines[header_idx + 2 :]
    columns = re.split(r"\s{2,}", header_line.strip())
    rows: list[dict[str, str]] = []
    for ln in data_lines:
        parts = re.split(r"\s{2,}", ln.strip(), maxsplit=len(columns) - 1)
        if len(parts) != len(columns):
            continue
        rows.append({k: v for k, v in zip(columns, parts)})
    return rows


def parse_kernel_status(raw: str) -> str:
    m = re.search(r'status "([^"]+)"', raw)
    return m.group(1) if m else "UNKNOWN"


def status_next_action(status: str) -> str:
    s = status.upper()
    if "COMPLETE" in s:
        return "Download output / submit"
    if "RUN" in s or "QUEU" in s:
        return "Wait for completion"
    if "ERROR" in s or "FAIL" in s:
        return "Open logs + rerun"
    return "Check notebook manually"


def parse_submission_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def latest_submission_rows(
    competition: str, limit: int = 20
) -> tuple[list[dict[str, str]], str | None]:
    res = run(["kaggle", "competitions", "submissions", "-c", competition])
    if res.code != 0:
        return [], f"failed to read submissions: {res.output.strip()[:300]}"
    try:
        df = pd.read_fwf(io.StringIO(res.output))
    except Exception as exc:  # pragma: no cover - defensive
        return [], f"failed to parse submissions table: {exc}"
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), "")
    rows = [{k: str(v).strip() for k, v in row.items()} for row in df.to_dict(orient="records")]
    cleaned = [r for r in rows if r.get("ref", "").isdigit()]
    return cleaned[:limit], None


def compact_submission_summary(rows: Iterable[dict[str, str]]) -> str:
    rows = list(rows)
    if not rows:
        return "No submissions found"
    best = max(
        (r for r in rows if r.get("publicScore")),
        key=lambda r: float(r["publicScore"]),
        default=None,
    )
    latest = rows[0]
    bits = [f"Latest ref {latest.get('ref','?')} score {latest.get('publicScore','-')}"]
    if best is not None:
        bits.append(f"Best public {best['publicScore']} (ref {best.get('ref','?')})")
    return " | ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition", default=DEFAULT_COMPETITION)
    ap.add_argument("--owner", default=DEFAULT_OWNER)
    ap.add_argument(
        "--kernel",
        action="append",
        dest="kernels",
        help="Kernel slug without owner (repeatable). Defaults to merge-1..8.",
    )
    args = ap.parse_args()

    kernels = args.kernels or DEFAULT_KERNELS

    submissions, err = latest_submission_rows(args.competition)
    if err:
        print(f"Submission summary: {err}")
    else:
        print(f"Submission summary: {compact_submission_summary(submissions)}")
    print()

    print("| Kernel | Status | Next action |")
    print("|---|---|---|")
    for slug in kernels:
        full = f"{args.owner}/{slug}"
        status_res = run(["kaggle", "kernels", "status", full])
        status = parse_kernel_status(status_res.output)
        action = status_next_action(status)
        print(f"| `{full}` | `{status}` | {action} |")

    if submissions:
        print("\nRecent submissions:")
        print("| Ref | Date | Public | Status |")
        print("|---|---|---:|---|")
        for row in submissions[:8]:
            print(
                f"| `{row.get('ref','')}` | {row.get('date','')} | "
                f"{row.get('publicScore','')} | {row.get('status','')} |"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
