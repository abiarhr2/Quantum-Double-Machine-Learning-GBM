#!/bin/bash
#SBATCH --job-name=QNN_stage2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qnn_stage2_%j.out
#SBATCH --error=logs/qnn_stage2_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
source ~/venv_qdml/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u 03_QNN_Hyperparameter_Optimization.py --stage 2 --data data/GBM_GBS.xlsx
EOF