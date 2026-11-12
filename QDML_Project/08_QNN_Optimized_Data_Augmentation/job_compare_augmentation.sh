#!/usr/bin/env bash
#SBATCH --job-name=qnn_aug_compare
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-$HOME/venv_qdml}/bin/activate"
exec python -u compare_augmentation.py "$@"
