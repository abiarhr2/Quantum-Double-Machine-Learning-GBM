#!/usr/bin/env bash
#SBATCH --job-name=qnn_pca_boot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/boot_%A_%a.out
#SBATCH --error=logs/boot_%A_%a.err
set -euo pipefail
cd "$ROOT"
source "$VENV/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
start=$((SLURM_ARRAY_TASK_ID * 50))
end=$((start + 50))
on_timeout() {
  trap - USR1
  echo "Time limit approaching; requesting requeue for job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || echo "Requeue failed: submit this job again manually" >&2
  kill -TERM "$child" 2>/dev/null || true
  wait "$child" 2>/dev/null || true
  exit 1
}
trap on_timeout USR1
python -u "$BOOTSTRAP_SCRIPT" --start "$start" --end "$end" --skip-original &
child=$!
wait "$child"

