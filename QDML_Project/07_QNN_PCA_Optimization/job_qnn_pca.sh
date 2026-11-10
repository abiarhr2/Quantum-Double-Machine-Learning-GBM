
#!/bin/bash
#SBATCH --job-name=QNN_PCA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=qnn_pca_%j.out
#SBATCH --error=qnn_pca_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/07_QNN_PCA_Optimization"
source ~/venv_qdml/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Replace with the actual Python selection script filename
SCRIPT="selection_qnn_pca.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Script not found: $SCRIPT"
    exit 1
fi

restart_job() {
    trap - USR1
    echo "Time limit approaching. Submitting continuation job."
    sbatch "$SLURM_SUBMIT_DIR/job_qnn_pca_selection.sh"
    kill -TERM "$PY_PID" 2>/dev/null || true
    exit 0
}

trap restart_job USR1

python -u "$SCRIPT" --seeds 42 --epochs 120 &
PY_PID=$!
wait "$PY_PID"
