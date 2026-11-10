#!/bin/bash
#SBATCH --job-name=qnn_pca_boot
#SBATCH --array=0-199%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pca_boot_%A_%a.out
#SBATCH --error=logs/pca_boot_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/venv_qdml/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * 5))
END=$((START + 5))
if [ "$START" -ge 1000 ]; then exit 0; fi
if [ "$END" -gt 1000 ]; then END=1000; fi
FLAG=--skip-original
if [ "$TASK" -eq 0 ]; then FLAG=; fi
python -u bootstrap_qnn_pca.py --start "$START" --end "$END" $FLAG
