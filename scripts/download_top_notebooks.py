"""Download top notebooks from the NVIDIA Nemotron Model Reasoning Challenge.

Usage:
    uv run python scripts/download_top_notebooks.py [--n 20] [--sort voteCount]

Saves each notebook into notebooks/<author>__<slug>/ along with its metadata.json.
Skips notebooks that are already downloaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "notebooks"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def list_notebooks(sort_by: str, page_size: int) -> list[dict]:
    """Return a list of notebook records sorted by `sort_by`."""
    out = run(
        [
            "kaggle", "kernels", "list",
            "--competition", COMPETITION,
            "--sort-by", sort_by,
            "--page-size", str(page_size),
            "--kernel-type", "notebook",
            "--language", "python",
            "-v",
        ]
    ).stdout

    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    # The CSV header is on the line after a "Next Page Token" notice if present.
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("ref,"))
    header = lines[header_idx].split(",")
    rows = []
    for ln in lines[header_idx + 1:]:
        # Simple CSV parse — fields don't contain commas in this output.
        parts = ln.split(",")
        if len(parts) < len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def slugify_dir(ref: str) -> str:
    """Turn 'author/notebook-slug' into 'author__notebook-slug'."""
    return ref.replace("/", "__")


def download(ref: str, dest: Path) -> None:
    """Pull notebook source + metadata into `dest`."""
    if dest.exists():
        print(f"  · skip (already present): {dest.name}")
        return
    tmp = dest.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    # `kernels pull -m` also fetches the metadata file.
    run(["kaggle", "kernels", "pull", ref, "-p", str(tmp), "-m"])
    tmp.rename(dest)
    print(f"  ✓ downloaded: {dest.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="Top N notebooks to download (default 20)")
    ap.add_argument(
        "--sort",
        default="voteCount",
        choices=["voteCount", "hotness", "scoreDescending", "dateCreated"],
        help="How to rank notebooks (default voteCount)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing top {args.n} notebooks by {args.sort} for competition '{COMPETITION}'...")
    rows = list_notebooks(sort_by=args.sort, page_size=max(args.n, 20))
    rows = rows[: args.n]
    if not rows:
        print("No notebooks returned. Check that you're authenticated and the comp slug is correct.")
        return 1

    print(f"Got {len(rows)} notebooks. Downloading into {OUT_DIR.relative_to(ROOT)}/ ...")
    index = []
    for i, row in enumerate(rows, 1):
        ref = row["ref"]
        title = row.get("title", "")
        votes = row.get("totalVotes", "?")
        print(f"[{i:>2}/{len(rows)}] {ref}  ({votes} votes)  — {title}")
        dest = OUT_DIR / slugify_dir(ref)
        try:
            download(ref, dest)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ failed: {e.stderr.strip() or e.stdout.strip()}")
            continue
        index.append({"rank": i, "ref": ref, "title": title, "totalVotes": votes, "dir": dest.name})

    # Write an index for quick navigation.
    index_path = OUT_DIR / "_index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nWrote index: {index_path.relative_to(ROOT)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
