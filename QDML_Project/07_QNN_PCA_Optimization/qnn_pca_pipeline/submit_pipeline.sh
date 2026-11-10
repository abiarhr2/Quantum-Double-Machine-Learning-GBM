#!/usr/bin/env bash
set -euo pipefail

# Run this from the 07_QNN_PCA_Optimization directory on DICO.
# Set these to your REAL filenames before running.
SELECTION_SCRIPT="${SELECTION_SCRIPT:-selection_qnn_pca.py}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-bootstrap_qnn_pca.py}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-compare_qnn_pca.py}"
VENV="${VENV:-$HOME/venv_qdml}"

ROOT="$(pwd)"
for f in "$SELECTION_SCRIPT" "$BOOTSTRAP_SCRIPT" "$COMPARE_SCRIPT" common.py; do
  [[ -f "$ROOT/$f" ]] || { echo "Missing file: $ROOT/$f" >&2; exit 1; }
done
[[ -f "$VENV/bin/activate" ]] || { echo "Missing Python environment: $VENV" >&2; exit 1; }
mkdir -p "$ROOT/logs"

export ROOT SELECTION_SCRIPT BOOTSTRAP_SCRIPT COMPARE_SCRIPT VENV
selection_id=$(sbatch --parsable --export=ALL,ROOT,SELECTION_SCRIPT,BOOTSTRAP_SCRIPT,COMPARE_SCRIPT,VENV selection_job.sh)
bootstrap_id=$(sbatch --parsable --dependency="afterok:${selection_id}" --array=0-19%8 --export=ALL,ROOT,SELECTION_SCRIPT,BOOTSTRAP_SCRIPT,COMPARE_SCRIPT,VENV bootstrap_job.sh)
comparison_id=$(sbatch --parsable --dependency="afterok:${bootstrap_id}" --export=ALL,ROOT,SELECTION_SCRIPT,BOOTSTRAP_SCRIPT,COMPARE_SCRIPT,VENV comparison_job.sh)
printf 'Selection: %s\nBootstrap array: %s\nComparison: %s\n' "$selection_id" "$bootstrap_id" "$comparison_id"
