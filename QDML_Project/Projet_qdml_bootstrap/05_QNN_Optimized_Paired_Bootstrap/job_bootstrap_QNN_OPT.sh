#!/bin/bash
# Bootstrap QNN_OPT uniquement (le MLP n'est pas réentraîné).
# Tableau de tâches : tâche k -> itérations [k*CHUNK, (k+1)*CHUNK).
# Reprise : relancer exactement la même commande sbatch ; chaque tâche repart de son
# fichier resultats_bootstrap_QNN_OPT/boot_<start>_<end>.csv (sauvegarde à chaque itération).
#
# Mesuré en local (1 CPU) : ~2.1 h / itération (b = 0 : 7640 s) -> ~10.5 h / tâche de 5 itérations.
#
#   cd ~/projet_qdml_bootstrap && mkdir -p logs          # logs/ doit exister AVANT sbatch
#   sbatch --array=0 job_bootstrap_QNN_OPT.sh            # test : tâche 0 (originale + b = 0..4)
#   sbatch job_bootstrap_QNN_OPT.sh                      # 1000 itérations (200 tâches x 5)
#   sbatch --array=3,17 job_bootstrap_QNN_OPT.sh         # ne relancer que certaines tâches
#
#SBATCH --job-name=boot_qnn_opt
#SBATCH --array=0-199%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qnn_opt_%A_%a.log

PROJECT=~/projet_qdml_bootstrap
SCRIPT=05_QNN_Optimized_Paired_Bootstrap.py
OUT=resultats_bootstrap_QNN_OPT
CHUNK=5
B=1000

cd "$PROJECT" || exit 1
source ~/venv_qdml/bin/activate || exit 1
mkdir -p logs "$OUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * CHUNK))
END=$((START + CHUNK))
if [ "$END" -gt "$B" ]; then END=$B; fi
if [ "$START" -ge "$B" ]; then echo "Tâche $TASK hors plage"; exit 0; fi

# Vérifications (données, plis identiques au MLP existant = scikit-learn >= 1.8, config) ; aucun entraînement
python -u "$SCRIPT" --check --out-dir "$OUT" || { echo "Échec de --check : arrêt."; exit 1; }

# Estimation sur données originales : tâche 0 seulement, et seulement si absente
FLAG="--skip-original"
if [ "$TASK" -eq 0 ] && [ ! -f "$OUT/resultats_originaux_QNN_OPT.csv" ]; then
    FLAG=""
fi

echo "Tâche $TASK : itérations $START à $((END - 1)) | $(date)"
python -u "$SCRIPT" --start "$START" --end "$END" $FLAG --out-dir "$OUT"
status=$?
echo "Fin tâche $TASK (code $status) | $(date)"
exit $status
