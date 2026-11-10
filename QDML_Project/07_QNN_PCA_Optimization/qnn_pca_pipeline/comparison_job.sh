#!/usr/bin/env bash
#SBATCH --job-name=qnn_pca_compare
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
set -euo pipefail
cd "$ROOT"
source "$VENV/bin/activate"
# Original estimate is computed here only if missing. Bootstrap is already finished.
python - <<'PY'
import json
from pathlib import Path
from common import load_data, evaluate, atomic_json, HOME
out=HOME/'results/bootstrap'
orig=out/'original.json'
if not orig.exists():
    cfg=json.loads((HOME/'results/selection/selected.json').read_text())
    X,Y,D=load_data()
    atomic_json(orig,evaluate(X,Y,D,cfg))
PY
python -u "$COMPARE_SCRIPT" --pca-dir "$ROOT/results/bootstrap"
