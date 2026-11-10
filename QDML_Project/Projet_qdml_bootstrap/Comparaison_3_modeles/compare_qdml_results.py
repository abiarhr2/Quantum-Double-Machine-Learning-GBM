"""
COMPARAISON MLP (NN) / QNN original / QNN optimisé (QNN_OPT)

Lit uniquement des résultats existants (aucun entraînement) :
  - resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized/  : NN et QNN (1 couche, 40 époques, lr 0.2)
  - resultats_bootstrap_QNN_OPT/                          : QNN_OPT (3 couches, 120 époques, lr 0.05)
Les itérations sont appariées par b. L'appariement NN/QNN ↔ QNN_OPT n'est utilisé que s'il est vérifié
(empreinte des plis + hash des échantillons recalculé) ; sinon il est signalé et les différences
appariées ne sont pas calculées.

Inférence ANALYTIQUE : estimation sur données originales, SE sandwich DML, IC = theta ± 1.96 SE, p normale.
Inférence BOOTSTRAP  : distribution des theta*, SE = écart-type, IC percentile 2.5-97.5 %, p bootstrap.

Utilisation :
  python compare_qdml_results.py
  python compare_qdml_results.py --new-dir resultats_bootstrap_QNN_OPT --out-dir comparaison_3_modeles
"""

import argparse
import glob
import hashlib
import json
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOT_SEED = 0
REF_FOLD_HASH = "a56f4caa2b1e"   # plis du bootstrap MLP existant (voir QDML_Bootstrap_With_Optimized_QNN_and_MLP.py)
MODELS = ["NN", "QNN", "QNN_OPT"]
LABELS = {"NN": "MLP", "QNN": "QNN (L1, 40 ép.)", "QNN_OPT": "QNN_OPT (L3, 120 ép.)"}
COLORS = {"NN": "#2a78d6", "QNN": "#eb6834", "QNN_OPT": "#1baf7a"}   # ordre catégoriel fixe
DIFFS = [("QNN_OPT", "NN"), ("QNN_OPT", "QNN"), ("QNN", "NN")]

parser = argparse.ArgumentParser()
parser.add_argument("--prev-dir", default="resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized")
parser.add_argument("--new-dir", default="resultats_bootstrap_QNN_OPT")
parser.add_argument("--out-dir", default="comparaison_3_modeles")
parser.add_argument("--data", default="data/GBM_GBS.xlsx", help="pour recalculer les échantillons bootstrap")
parser.add_argument("--B", type=int, default=1000, help="nombre d'itérations attendu")
args = parser.parse_args()

for d in (args.prev_dir, args.new_dir):
    if os.path.abspath(args.out_dir) == os.path.abspath(d):
        raise SystemExit("--out-dir doit être différent des dossiers de résultats")
os.makedirs(args.out_dir, exist_ok=True)
warnings_ = []

def flag(msg):
    warnings_.append(msg)
    print("  [!] " + msg)

def load_boot(d, cols):
    files = sorted(glob.glob(os.path.join(d, "boot_*_*.csv")))
    if not files:
        return None
    allb = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # même b présent dans plusieurs fichiers avec des valeurs différentes -> incohérence
    conflict = allb.groupby("b")[cols].nunique(dropna=False).gt(1).any(axis=1)
    if conflict.any():
        flag(f"{d} : b présents plusieurs fois avec des valeurs différentes : {list(conflict[conflict].index[:5])}")
    return allb.drop_duplicates(subset="b", keep="last").set_index("b").sort_index()

def summarize(thetas):   # identique à QDML_Bootstrap_paired.py
    thetas = np.asarray(thetas)
    p_low = (np.sum(thetas <= 0) + 1) / (len(thetas) + 1)
    p_high = (np.sum(thetas >= 0) + 1) / (len(thetas) + 1)
    return {"moyenne_boot": thetas.mean(),
            "mediane_boot": np.median(thetas),
            "se_boot": thetas.std(ddof=1),
            "ic_low_boot": np.percentile(thetas, 2.5),
            "ic_high_boot": np.percentile(thetas, 97.5),
            "p_boot": min(1.0, 2 * min(p_low, p_high)),
            "B_valides": len(thetas)}

# ===========================
# 1. Chargement
# ===========================
print("=== Chargement ===")
old = load_boot(args.prev_dir, ["NN", "QNN"])
if old is None:
    raise SystemExit(f"Aucun résultat bootstrap dans {args.prev_dir}")
new = load_boot(args.new_dir, ["QNN_OPT"])
if new is None:
    flag(f"aucun résultat QNN_OPT dans {args.new_dir} : seuls NN et QNN sont résumés")
    new = pd.DataFrame(columns=["QNN_OPT", "QNN_OPT_se", "idx_hash", "temps_s", "erreur"])
    new.index.name = "b"

boot = old[["NN", "NN_se", "QNN", "QNN_se"]].join(new[["QNN_OPT", "QNN_OPT_se"]], how="outer")
for m in MODELS:
    done = boot[m].notna().sum()
    total = (old if m != "QNN_OPT" else new).shape[0]
    print(f"  {m:8s}: {total} itérations enregistrées, {done} valides")
    if total < args.B:
        flag(f"{m} : {total}/{args.B} itérations enregistrées (manquantes : {args.B - total})")
if "erreur" in new and new["erreur"].fillna("").astype(str).str.len().gt(0).any():
    errs = new.loc[new["erreur"].fillna("").astype(str).str.len() > 0, "erreur"]
    flag(f"QNN_OPT : {len(errs)} itérations écartées, ex. b={errs.index[0]} : {errs.iloc[0]}")

# ===========================
# 2. Vérification de l'appariement
# ===========================
print("\n=== Appariement par b ===")
paired_ok = len(new) > 0
cfg_path = os.path.join(args.new_dir, "config_QNN_OPT.json")
if len(new):
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if cfg.get("fold_hash") != REF_FOLD_HASH:
        paired_ok = False
        flag(f"empreinte des plis QNN_OPT = {cfg.get('fold_hash')} ≠ {REF_FOLD_HASH} (MLP) : "
             "plis différents, appariement INVALIDE")
    if cfg.get("boot_seed", BOOT_SEED) != BOOT_SEED:
        paired_ok = False
        flag("graine bootstrap différente : appariement INVALIDE")
    if os.path.exists(args.data):
        df = pd.read_excel(args.data, header=3)
        df.columns = df.columns.str.strip()
        n = int(df.dropna(subset=["name"]).shape[0])
        if cfg.get("n", n) != n:
            paired_ok = False
            flag(f"n différent ({cfg.get('n')} vs {n})")
        h = {b: hashlib.md5(np.random.RandomState(BOOT_SEED + b).choice(n, size=n, replace=True)
                            .astype(np.int64).tobytes()).hexdigest()[:12] for b in new.index}
        bad = [b for b in new.index if str(new.loc[b, "idx_hash"]) != h[b]]
        if bad:
            paired_ok = False
            flag(f"hash d'échantillon incohérent pour {len(bad)} itérations (ex. {bad[:5]})")
        else:
            print(f"  échantillons QNN_OPT recalculés et identiques ({len(new)} itérations)")
    else:
        flag(f"{args.data} introuvable : hash des échantillons non vérifié")
    print(f"  appariement QNN_OPT ↔ MLP/QNN : {'VALIDE' if paired_ok else 'NON VALIDE'}")

common = boot.dropna(subset=MODELS if paired_ok else ["NN", "QNN"])
print(f"  itérations valides communes : {len(common)}")

# ===========================
# 3. Inférence analytique (données originales)
# ===========================
orig = []
for path in (os.path.join(args.prev_dir, "resultats_originaux.csv"),
             os.path.join(args.new_dir, "resultats_originaux_QNN_OPT.csv")):
    if os.path.exists(path):
        orig.append(pd.read_csv(path, index_col=0))
    else:
        flag(f"{path} absent : estimation originale indisponible")
orig = pd.concat(orig) if orig else pd.DataFrame()

rows = {}
for m in MODELS:
    r = {}
    if m in orig.index:
        o = orig.loc[m]
        r.update({"theta_original": o["theta"], "SE_analytique": o["SE"],
                  "IC_low_analytique": o["IC_low"], "IC_high_analytique": o["IC_high"],
                  "p_analytique": o["p-value"], "temps_original_s": o["temps_s"]})
    th = boot[m].dropna()
    if len(th) >= 2:
        r.update(summarize(th))
        r["SE_analytique_moy_boot"] = boot[m + "_se"].dropna().mean()
    if m == "QNN_OPT" and "temps_s" in new and new["temps_s"].notna().any():
        r["temps_moy_iteration_s"] = new["temps_s"].mean()
        r["temps_total_boot_h"] = new["temps_s"].sum() / 3600
    rows[LABELS[m]] = r

table = pd.DataFrame(rows).T
cols = ["theta_original", "SE_analytique", "IC_low_analytique", "IC_high_analytique", "p_analytique",
        "moyenne_boot", "mediane_boot", "se_boot", "ic_low_boot", "ic_high_boot", "p_boot", "B_valides",
        "SE_analytique_moy_boot", "temps_original_s", "temps_moy_iteration_s", "temps_total_boot_h"]
table = table.reindex(columns=[c for c in cols if c in table.columns])
print("\n=== Par modèle (chaque modèle sur ses propres itérations valides) ===")
print("Analytique = données originales, SE sandwich DML | Bootstrap = distribution des theta*")
print(table.round(4).to_string())
print("Temps par itération : non enregistré dans l'ancien bootstrap (NN, QNN).")
table.round(6).to_csv(os.path.join(args.out_dir, "comparaison_modeles.csv"))

# ===========================
# 4. Différences appariées (bootstrap uniquement : pas de SE analytique de la différence)
# ===========================
drows = {}
for a, b in DIFFS:
    if "QNN_OPT" in (a, b) and not paired_ok:
        continue
    d = (common[a] - common[b]).dropna()
    if len(d) >= 2:
        r = summarize(d)
        r["diff_original"] = (orig.loc[a, "theta"] - orig.loc[b, "theta"]
                              if a in orig.index and b in orig.index else np.nan)
        drows[f"{a} - {b}"] = r
dtable = pd.DataFrame(drows).T
if len(dtable):
    print(f"\n=== Différences appariées (itérations communes : {len(common)}) ===")
    print(dtable.round(4).to_string())
    dtable.round(6).to_csv(os.path.join(args.out_dir, "differences_appariees.csv"))
common.to_csv(os.path.join(args.out_dir, "thetas_apparies.csv"))

# ===========================
# 5. Figures
# ===========================
avail = [m for m in MODELS if boot[m].notna().sum() >= 2]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6})

fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.histogram_bin_edges(pd.concat([boot[m].dropna() for m in avail]), bins=40)
for m in avail:
    ax.hist(boot[m].dropna(), bins=bins, alpha=0.45, color=COLORS[m], edgecolor="white", linewidth=0.5,
            label=f"{LABELS[m]} (B={boot[m].notna().sum()})")
    if m in orig.index:
        ax.axvline(orig.loc[m, "theta"], color=COLORS[m], lw=2)
ax.axvline(0, color="0.3", ls="--", lw=1)
ax.set_xlabel("θ*  (trait plein : θ sur données originales)")
ax.set_ylabel("fréquence")
ax.set_title("Distributions bootstrap de θ*")
ax.legend(frameon=False)
fig.savefig(os.path.join(args.out_dir, "hist_thetas.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

if len(dtable):
    fig, axes = plt.subplots(1, len(dtable), figsize=(4.2 * len(dtable), 3.6), squeeze=False)
    for ax, (name, r) in zip(axes[0], dtable.iterrows()):
        a, b = [s.strip() for s in name.split("-")]
        ax.hist((common[a] - common[b]).dropna(), bins=30, color="0.55", edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="0.2", ls="--", lw=1)
        ax.axvspan(r["ic_low_boot"], r["ic_high_boot"], color=COLORS[a], alpha=0.12)
        ax.set_title(f"{name}\nmoy. {r['moyenne_boot']:.3f}, IC95 [{r['ic_low_boot']:.3f}, "
                     f"{r['ic_high_boot']:.3f}], p={r['p_boot']:.3f}", fontsize=9)
        ax.set_xlabel("différence appariée de θ*")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "hist_differences.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 0.9 + 0.9 * len(MODELS)))
for i, m in enumerate(MODELS):
    r = rows[LABELS[m]]
    y = len(MODELS) - 1 - i
    if "theta_original" in r:
        ax.errorbar(r["theta_original"], y + 0.15, xerr=[[r["theta_original"] - r["IC_low_analytique"]],
                    [r["IC_high_analytique"] - r["theta_original"]]], fmt="o", ms=8, lw=2, capsize=0,
                    color=COLORS[m], label="analytique (θ original ± 1.96 SE)" if i == 0 else None)
    if "moyenne_boot" in r:
        ax.errorbar(r["moyenne_boot"], y - 0.15, xerr=[[r["moyenne_boot"] - r["ic_low_boot"]],
                    [r["ic_high_boot"] - r["moyenne_boot"]]], fmt="s", ms=8, lw=2, capsize=0, mfc="white",
                    color=COLORS[m], label="bootstrap (moyenne, IC percentile)" if i == 0 else None)
    if "theta_original" not in r and "moyenne_boot" not in r:
        ax.text(0, y, "données manquantes", ha="center", va="center", color="0.4")
ax.axvline(0, color="0.3", ls="--", lw=1)
ax.set_ylim(-0.6, len(MODELS) - 0.4)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([LABELS[m] for m in reversed(MODELS)])
ax.set_xlabel("θ (effet de l'oedème sur P(GSM))")
ax.set_title("IC à 95 % : analytique vs bootstrap")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2)
fig.savefig(os.path.join(args.out_dir, "forest_ic.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFichiers dans {args.out_dir}/ : comparaison_modeles.csv, differences_appariees.csv, "
      "thetas_apparies.csv, hist_thetas.png, hist_differences.png, forest_ic.png")
if warnings_:
    print("\n=== Avertissements ===\n  - " + "\n  - ".join(warnings_))
