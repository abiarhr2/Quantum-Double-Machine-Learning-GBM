#!/usr/bin/env bash
#SBATCH --job-name=qnn_aug_select
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/select_%j.out
#SBATCH --error=logs/select_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-$HOME/venv_qdml}/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u select_augmentation.py "$@"
