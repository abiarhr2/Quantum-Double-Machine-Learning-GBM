#!/usr/bin/env bash
#SBATCH --job-name=qnn_aug_boot
#SBATCH --array=0-199%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/bootstrap_%A_%a.out
#SBATCH --error=logs/bootstrap_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-$HOME/venv_qdml}/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

requeue() {
    trap '' USR1
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
}
start=$((SLURM_ARRAY_TASK_ID * 5))
python -u bootstrap_qnn_augmented.py --start "$start" --end "$((start + 5))" "$@" &
child=$!
trap 'requeue; exit 0' USR1
wait "$child"
