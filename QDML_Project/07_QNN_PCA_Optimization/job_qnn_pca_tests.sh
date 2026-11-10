#!/bin/bash
#SBATCH --job-name=qnn_pca_tests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pca_tests_%j.out
#SBATCH --error=logs/pca_tests_%j.err
# No partition specified: same cluster default as experiment 05.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/venv_qdml/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u select_qnn_pca.py --seeds 42 43 44
