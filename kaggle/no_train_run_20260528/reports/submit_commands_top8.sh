#!/usr/bin/env bash
# Submit one at a time — copy to submission.zip first (competition requires that exact name)
set -euo pipefail

COMP=nvidia-nemotron-model-reasoning-challenge
PKGS=kaggle/no_train_run_20260528/packages
TMP=submission.zip

submit() {
  local src="$1" msg="$2"
  echo "=== Submitting: $src ==="
  cp "$PKGS/$src" "$TMP"
  .venv/bin/kaggle competitions submit -c $COMP -f "$TMP" -m "$msg"
  rm -f "$TMP"
  echo "=== Done: $src ==="
}

# Run these one at a time — uncomment one, run, wait for score, then next

submit "kienngx_tinker_adapter_v1__rewrite.zip"   "no-train-20260528 p10 tinker-rewrite"
# submit "k1e5_plus_tnorm_full_005__rewrite.zip"    "no-train-20260528 p11 k1e5+tnorm-005"
# submit "k1e5_plus_tnorm_full_008__rewrite.zip"    "no-train-20260528 p12 k1e5+tnorm-008"
# submit "k1e5_plus_tnorm_mamba_005__rewrite.zip"   "no-train-20260528 p13 k1e5+tnorm-mamba"
# submit "tnorm_plus_hnorm_full_005__rewrite.zip"   "no-train-20260528 p14 tnorm+hnorm-005"
# submit "k1e5_plus_hnorm_full_005__rewrite.zip"    "no-train-20260528 p15 k1e5+hnorm-005"
