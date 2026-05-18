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

# What goes into the tarball. Paths are repo-relative.
BUNDLED_PATHS = [
    "nemotron_local_cv",
    "nemotron_train",
    "data/val_split_600.csv",
    "kaggle/blackwell_setup.py",
]


def build_tarball(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in BUNDLED_PATHS:
            src = REPO_ROOT / rel
            if not src.exists():
                sys.exit(f"missing bundled path: {src}")
            print(f"  + {rel}")
            tar.add(src, arcname=rel, filter=_filter_junk)
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


def push(staging: Path, message: str) -> None:
    """Try `version` first; if dataset doesn't exist yet, fall back to `create`."""
    base_args = ["uv", "run", "kaggle", "datasets"]
    version_cmd = [*base_args, "version", "-p", str(staging), "-m", message, "--dir-mode", "skip"]
    print("\n$", " ".join(version_cmd))
    r = subprocess.run(version_cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(r.stdout)
        return
    out = (r.stdout or "") + (r.stderr or "")
    if "404" in out or "Not Found" in out or "could not be found" in out.lower():
        print("dataset does not exist yet — creating it")
        create_cmd = [*base_args, "create", "-p", str(staging), "--dir-mode", "skip"]
        print("$", " ".join(create_cmd))
        r2 = subprocess.run(create_cmd, capture_output=True, text=True)
        print(r2.stdout)
        if r2.returncode != 0:
            sys.exit(r2.stderr or "kaggle datasets create failed")
        return
    print(out, file=sys.stderr)
    sys.exit(f"kaggle datasets version failed (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="aleixlopez/nemotron-code",
                    help="Kaggle dataset slug (owner/name)")
    ap.add_argument("--message", default="sync from laptop",
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

    push(STAGING_DIR, args.message)
    print(f"\nDataset {args.slug} ready. Attach it in Kaggle notebooks at /kaggle/input/{args.slug.split('/', 1)[1]}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
