#!/usr/bin/env python3
"""
Phase 2 Polling Daemon: Monitor kernel completion and auto-submit.

This script runs in the background and:
1. Polls Kaggle kernel output every 5 minutes
2. When kernel completes, auto-downloads results
3. Auto-submits all candidates ≥ 0.80
4. Logs everything to kaggle/phase2_auto_runner.log

Usage:
    uv run python scripts/phase2_polling_daemon.py &    # Start in background
    uv run python scripts/phase2_polling_daemon.py --stop  # Stop daemon (if implemented)
"""

import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_USER = "aleixlopez"
MERGE_KERNEL_SLUG = f"{KAGGLE_USER}/nemotron-phase-2-merge-search-submit"
MERGE_KERNEL_OUTPUT_DIR = REPO_ROOT / "kaggle" / "merge_kernel_output"
LOG_FILE = REPO_ROOT / "kaggle" / "phase2_auto_runner.log"
AUTO_SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "phase2_auto_submit.py"

POLL_INTERVAL = 300  # 5 minutes

def log_msg(msg: str):
    """Write to both stdout and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return (returncode, stdout+stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except Exception as e:
        return 1, str(e)

def check_kernel_ready() -> bool:
    """Check if kernel output is available."""
    output_dir = MERGE_KERNEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["kaggle", "kernels", "output", MERGE_KERNEL_SLUG, "--path", str(output_dir)]
    rc, out = run_command(cmd)
    
    if rc == 0:
        summary_csv = output_dir / "merge_eval_summary.csv"
        if summary_csv.exists():
            log_msg("✓ Kernel output detected and ready!")
            return True
    
    return False

def download_and_submit():
    """Download results and submit all candidates."""
    log_msg("Starting download & submission pipeline...")
    
    # Run --download
    log_msg("Running: phase2_auto_submit.py --download")
    cmd = [sys.executable, str(AUTO_SUBMIT_SCRIPT), "--download"]
    rc, out = run_command(["uv", "run"] + cmd)
    
    if rc != 0:
        log_msg(f"❌ Download failed: {out[:200]}")
        return False
    
    log_msg(f"✓ Download successful")
    
    # Run --submit-all
    log_msg("Running: phase2_auto_submit.py --submit-all")
    cmd = [sys.executable, str(AUTO_SUBMIT_SCRIPT), "--submit-all"]
    rc, out = run_command(["uv", "run"] + cmd)
    
    if rc != 0:
        log_msg(f"❌ Submission failed: {out[:200]}")
        return False
    
    log_msg("✓ All submissions complete!")
    log_msg("=" * 80)
    log_msg("PHASE 2 AUTOMATION FINISHED!")
    log_msg("Monitor leaderboard for public scores over next 24 hours.")
    log_msg("=" * 80)
    return True

def main():
    log_msg("=" * 80)
    log_msg("PHASE 2 POLLING DAEMON STARTED")
    log_msg("=" * 80)
    log_msg(f"Kernel: {MERGE_KERNEL_SLUG}")
    log_msg(f"Poll interval: {POLL_INTERVAL}s")
    log_msg(f"Log file: {LOG_FILE}")
    log_msg("Waiting for kernel to complete...")
    log_msg("-" * 80)
    
    poll_count = 0
    while True:
        poll_count += 1
        log_msg(f"[Poll #{poll_count}] Checking kernel status...")
        
        if check_kernel_ready():
            log_msg("🎯 Kernel ready! Starting download & submission...")
            if download_and_submit():
                log_msg("✅ PHASE 2 COMPLETE")
                sys.exit(0)
            else:
                log_msg("⚠️  Submission had errors, but continuing to monitor...")
                time.sleep(60)  # Wait 1 min before retry
                continue
        
        log_msg(f"⏳ Kernel still running. Next check in {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_msg("Daemon interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        log_msg(f"❌ Fatal error: {e}")
        sys.exit(1)
