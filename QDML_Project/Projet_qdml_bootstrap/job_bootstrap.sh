
#!/bin/bash
#SBATCH --job-name=boot_qdml_1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

cd "$SLURM_SUBMIT_DIR" || exit 1
source ~/venv_qdml/bin/activate || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs resultats_bootstrap

pids=()

# 10 workers, 100 itérations chacun
for s in 0 100 200 300 400 500 600 700 800 900; do
    e=$((s + 100))

    FLAG="--skip-original"

    # Premier worker : estimation originale si absente
    if [ "$s" -eq 0 ] && \
       [ ! -f resultats_bootstrap/resultats_originaux.csv ]; then
        FLAG=""
    fi

    echo "Worker : $s à $((e - 1))"

    python -u QDML_bootstrap_paired.py \
        --start "$s" \
        --end "$e" \
        $FLAG \
        > "logs/worker_${s}.log" 2>&1 &

    pids+=($!)
done

# Attendre les 10 workers et vérifier leurs sorties
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

if [ "$failed" -ne 0 ]; then
    echo "Certains workers ont échoué. Relancer le job."
    exit 1
fi

# Vérifier que les 1000 indices ont été enregistrés
python - <<'PY'
import glob
import pandas as pd

files = glob.glob("resultats_bootstrap/boot_*_*.csv")
if not files:
    raise SystemExit("Aucun résultat bootstrap.")

df = pd.concat([pd.read_csv(f) for f in files])
df = df.drop_duplicates(subset="b")

expected = set(range(1000))
completed = set(df["b"].astype(int))
missing = expected - completed

if missing:
    raise SystemExit(f"Itérations manquantes : {len(missing)}")

print("1000 itérations enregistrées.")
print("Itérations valides :", df.dropna(subset=["NN", "QNN"]).shape[0])
PY

if [ "$?" -ne 0 ]; then
    echo "Bootstrap incomplet. Relancer le job."
    exit 1
fi

echo "=== Analyse finale ==="

python -u QDML_bootstrap_paired.py \
    --analyse-only \
    > logs/analyse.log 2>&1

cat logs/analyse.log
