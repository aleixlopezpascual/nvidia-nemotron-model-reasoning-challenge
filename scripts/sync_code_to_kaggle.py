"""Bundle our code packages into a Kaggle Dataset and push.

Why a Dataset instead of `pip install git+...`: the GitHub repo is private,
so notebooks would need a PAT in Kaggle Secrets. Uploading the code as a
private Kaggle Dataset is simpler and free of credential plumbing.

Why a single tarball inside the Dataset: Kaggle's `--dir-mode {zip,tar,skip}`
flags either flatten or auto-archive directories in surprising ways. A single
tarball at the dataset root sidesteps that — the notebook just `tar xzf`s it
into /kaggle/working and the Python package layout is preserved.

Usage (laptop):
    uv run python scripts/sync_code_to_kaggle.py \\
        [--slug aleixlopez/nemotron-code] \\
        [--message "what changed"]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = REPO_ROOT / "kaggle" / "code_dataset"

# What goes into the tarball. `arcname` is the path inside the bundle —
# Kaggle auto-extracts uploads so it ends up as `/kaggle/input/<dataset>/<arcname>`.
# blackwell_setup.py is re-arcnamed to root so it can't clash with Kaggle's own
# `kaggle` CLI package on sys.path.
BUNDLED_PATHS: list[tuple[str, str]] = [
    ("nemotron_local_cv", "nemotron_local_cv"),
    ("nemotron_train", "nemotron_train"),
    ("data/val_split_600.csv", "data/val_split_600.csv"),
    ("data/adapter_candidates.csv", "data/adapter_candidates.csv"),
    ("kaggle/blackwell_setup.py", "blackwell_setup.py"),
]


def build_tarball(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz") as tar:
        for rel, arc in BUNDLED_PATHS:
            src = REPO_ROOT / rel
            if not src.exists():
                sys.exit(f"missing bundled path: {src}")
            print(f"  + {rel}  ->  {arc}")
            tar.add(src, arcname=arc, filter=_filter_junk)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"built {out_path.name} ({size_mb:.2f} MB)")


def _filter_junk(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = info.name.split("/")
    if "__pycache__" in parts:
        return None
    if info.name.endswith((".pyc", ".DS_Store")):
        return None
    return info


def write_metadata(staging: Path, slug: str) -> None:
    meta = {
        "title": slug.split("/", 1)[1],
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def _dataset_exists(slug: str) -> bool:
    """Probe whether a dataset is already published. `status` errors out if missing."""
    r = subprocess.run(
        ["uv", "run", "kaggle", "datasets", "status", slug],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and "ready" in (r.stdout or "").lower()


def push(staging: Path, slug: str, message: str) -> None:
    """Use `version` if dataset exists, else `create`. Probe-first avoids the
    kaggle CLI's habit of returning exit 0 on dataset-already-exists conflicts."""
    base_args = ["uv", "run", "kaggle", "datasets"]
    if _dataset_exists(slug):
        cmd = [*base_args, "version", "-p", str(staging), "-m", message, "--dir-mode", "skip"]
        action = "version"
    else:
        cmd = [*base_args, "create", "-p", str(staging), "--dir-mode", "skip"]
        action = "create"

    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    # The kaggle CLI swallows real errors as exit 0, so also scan the output.
    if r.returncode != 0 or "error" in out.lower() and "creation error" in out.lower():
        sys.exit(out + f"\nkaggle datasets {action} failed (exit {r.returncode})")
    print(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="aleixlopez/nemotron-code",
                    help="Kaggle dataset slug (owner/name)")
    ap.add_argument("-m", "--message", default="sync from laptop",
                    help="Version note for the new dataset version")
    ap.add_argument("--no-push", action="store_true",
                    help="Build staging dir + tarball but don't run kaggle datasets push")
    args = ap.parse_args()

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    print(f"staging: {STAGING_DIR}")
    print("bundling paths:")
    build_tarball(STAGING_DIR / "nemotron-code.tar.gz")
    write_metadata(STAGING_DIR, args.slug)
    print(f"wrote {STAGING_DIR / 'dataset-metadata.json'}")

    if args.no_push:
        print("\n--no-push set; skipping kaggle CLI")
        return 0

    push(STAGING_DIR, args.slug, args.message)
    print(f"\nDataset {args.slug} ready. Attach it in Kaggle notebooks at /kaggle/input/{args.slug.split('/', 1)[1]}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
