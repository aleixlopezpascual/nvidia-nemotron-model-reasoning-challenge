#!/usr/bin/env python3
"""
Phase 2 Automation: Run merge search kernel on Kaggle, download results, 
submit all candidates with CV score ≥ 0.80.

Usage:
    uv run python scripts/phase2_auto_submit.py --run-kernel
    uv run python scripts/phase2_auto_submit.py --download
    uv run python scripts/phase2_auto_submit.py --submit-all
    uv run python scripts/phase2_auto_submit.py --full (does all 3)
"""

import subprocess
import json
import sys
import time
import os
from pathlib import Path
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_USER = "aleixlopez"
MERGE_KERNEL_SLUG = f"{KAGGLE_USER}/nemotron-phase-2-merge-search-submit"
MERGE_KERNEL_OUTPUT_DIR = REPO_ROOT / "kaggle" / "merge_kernel_output"
SUBMISSIONS_LOG = REPO_ROOT / "kaggle" / "phase2_submissions.log"

def run_kaggle_command(cmd: list[str]) -> tuple[int, str]:
    """Run a Kaggle CLI command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, f"Command timed out: {' '.join(cmd)}"
    except Exception as e:
        return 1, str(e)

def launch_kernel() -> bool:
    """Launch the merge search kernel on Kaggle."""
    print(f"Launching kernel: {MERGE_KERNEL_SLUG}")
    # Note: kaggle kernels run requires the kernel slug
    # However, this doesn't actually "run" a saved kernel; you'd need to use the web UI or a custom script
    # Instead, we'll use the output download to trigger (if not already running)
    print("⚠️  Manual action required: Go to https://www.kaggle.com/code/aleixlopez/nemotron-phase-2-merge-search-submit")
    print("    Click 'Run' to start the kernel evaluation.")
    print("    Once running, re-run this script with --download after ~1–2 hours.")
    return True

def poll_kernel_status() -> bool:
    """Check if kernel has completed and outputs are ready."""
    print(f"Checking kernel output availability...")
    output_dir = MERGE_KERNEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download the merge_eval_summary.csv
    cmd = [
        "kaggle", "kernels", "output",
        MERGE_KERNEL_SLUG,
        "--path", str(output_dir),
    ]
    rc, out = run_kaggle_command(cmd)
    if rc == 0:
        print("✓ Kernel output downloaded successfully")
        return True
    else:
        print(f"⚠️  Kernel output not yet available:")
        print(out[:500])  # Show first 500 chars of error
        return False

def download_results() -> Path | None:
    """Download merge_eval_summary.csv from the kernel output."""
    print("Downloading merge evaluation results...")
    output_dir = MERGE_KERNEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rc, out = run_kaggle_command([
        "kaggle", "kernels", "output",
        MERGE_KERNEL_SLUG,
        "--path", str(output_dir),
    ])
    
    if rc != 0:
        print(f"❌ Failed to download kernel output: {out[:200]}")
        return None
    
    summary_csv = output_dir / "merge_eval_summary.csv"
    if not summary_csv.exists():
        print(f"❌ merge_eval_summary.csv not found in kernel output")
        print(f"   Downloaded to: {output_dir}")
        print(f"   Contents: {list(output_dir.iterdir())}")
        return None
    
    print(f"✓ Downloaded: {summary_csv}")
    df = pd.read_csv(summary_csv)
    print(f"\nMerge evaluation results ({len(df)} variants):")
    print(df.sort_values("accuracy", ascending=False).to_string(index=False))
    return summary_csv

def submit_candidates(results_csv: Path, threshold: float = 0.80) -> bool:
    """Submit all candidates with CV score ≥ threshold."""
    df = pd.read_csv(results_csv)
    candidates = df[df["accuracy"] >= threshold].sort_values("accuracy", ascending=False)
    
    if len(candidates) == 0:
        print(f"❌ No candidates with CV score ≥ {threshold}")
        return False
    
    print(f"\n📊 Found {len(candidates)} candidates ≥ {threshold}:")
    print(candidates[["name", "accuracy"]].to_string(index=False))
    
    MERGE_KERNEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submissions_log = []
    
    for idx, row in candidates.iterrows():
        merge_name = row["name"]
        cv_score = row["accuracy"]
        
        # Locate the merged adapter in kernel output
        adapter_dir = MERGE_KERNEL_OUTPUT_DIR / "merged_adapters_phase2" / merge_name
        if not adapter_dir.exists():
            print(f"⚠️  Adapter dir not found: {adapter_dir}")
            continue
        
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required_files if not (adapter_dir / f).exists()]
        if missing:
            print(f"⚠️  Missing files in {merge_name}: {missing}")
            continue
        
        # Package as submission.zip
        submission_zip = REPO_ROOT / "kaggle" / f"submission_{merge_name}.zip"
        print(f"\n📦 Packaging {merge_name} (CV={cv_score:.4f})...")
        
        import zipfile
        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in required_files:
                zf.write(adapter_dir / fname, arcname=fname)
        
        size_mb = submission_zip.stat().st_size / 1024 / 1024
        print(f"   Saved: {submission_zip} ({size_mb:.2f} MB)")
        
        # Submit to competition
        print(f"   Submitting to competition...")
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", "nvidia-nemotron-model-reasoning-challenge",
            "-f", str(submission_zip),
            "-m", f"Phase 2: {merge_name} (CV={cv_score:.4f})",
        ]
        rc, out = run_kaggle_command(cmd)
        
        if rc == 0:
            print(f"   ✓ Submitted successfully")
            submissions_log.append({
                "merge_name": merge_name,
                "cv_score": cv_score,
                "submission_zip": str(submission_zip),
                "status": "submitted",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            print(f"   ❌ Submission failed: {out[:200]}")
            submissions_log.append({
                "merge_name": merge_name,
                "cv_score": cv_score,
                "submission_zip": str(submission_zip),
                "status": "failed",
                "error": out[:200],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        
        time.sleep(2)  # Rate limit
    
    # Log all submissions
    if submissions_log:
        log_df = pd.DataFrame(submissions_log)
        SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(SUBMISSIONS_LOG, index=False)
        print(f"\n✓ Submission log saved: {SUBMISSIONS_LOG}")
    
    return len(submissions_log) > 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-kernel",
        action="store_true",
        help="Print instructions to run the merge search kernel on Kaggle",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download merge evaluation results from the kernel",
    )
    ap.add_argument(
        "--submit-all",
        action="store_true",
        help="Submit all candidates with CV ≥ 0.80",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="CV score threshold for submission (default: 0.80)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run all steps: launch, download, submit",
    )
    args = ap.parse_args()
    
    if args.full:
        print("=" * 80)
        print("PHASE 2 FULL AUTOMATION")
        print("=" * 80)
        
        print("\n[1/3] Launching kernel...")
        if not launch_kernel():
            return 1
        
        print("\n[2/3] Waiting for kernel to complete (you must run it manually on Kaggle)...")
        print("      Re-run this script with --download after kernel completes.")
        return 0
    
    if args.run_kernel:
        launch_kernel()
        return 0
    
    if args.download:
        print("=" * 80)
        print("PHASE 2: DOWNLOAD RESULTS")
        print("=" * 80)
        
        # Poll for status
        if not poll_kernel_status():
            print("\n⏳ Kernel still running. Check again in a few minutes.")
            return 0
        
        # Download
        results_csv = download_results()
        if results_csv is None:
            return 1
        
        print("\n✓ Results ready for submission. Run with --submit-all to proceed.")
        return 0
    
    if args.submit_all:
        print("=" * 80)
        print(f"PHASE 2: SUBMIT ALL CANDIDATES (threshold ≥ {args.threshold})")
        print("=" * 80)
        
        # Find the most recent results CSV
        results_csv = MERGE_KERNEL_OUTPUT_DIR / "merge_eval_summary.csv"
        if not results_csv.exists():
            print(f"❌ Results not found. Run --download first.")
            return 1
        
        if not submit_candidates(results_csv, threshold=args.threshold):
            return 1
        
        print("\n" + "=" * 80)
        print("✓ PHASE 2 COMPLETE: All candidates submitted")
        print("=" * 80)
        return 0
    
    # Default: show help
    ap.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
