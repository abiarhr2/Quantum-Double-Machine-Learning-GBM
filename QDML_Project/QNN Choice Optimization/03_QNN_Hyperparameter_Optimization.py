#!/usr/bin/env python3
"""
QNN HYPERPARAMETER SEARCH FOR THE NUISANCE FUNCTIONS OF A DML MODEL
===================================================================
Selects the QNN configuration that best predicts the two nuisance functions

    g(X) = P(Y = 1 | X)   (tumor type: GSM = 1, GBM = 0)
    m(X) = P(D = 1 | X)   (edema present = 1)

using OUT-OF-FOLD predictive performance only (log-loss as primary criterion).
theta, p-values, confidence intervals and significance are NEVER computed or used.

Selection criterion:  LL_total = LL_g + LL_m  (out-of-fold, pooled over the 5 folds).

Stage 1 : screening, 1 quantum layer, grid of learning rates x epochs, ONE seed.
Stage 2 : refinement of the top-K stage-1 configurations: layers x seeds
          (+ a longer-training variant when stage-1 convergence diagnostics justify it).

All configurations and baselines are evaluated on IDENTICAL validation folds
(for a given seed, folds come from StratifiedKFold(shuffle=True, random_state=seed), stratified on D).

Baselines (same folds): prevalence of the training fold, and the optimised MLP
(StandardScaler + MLP(32, relu, alpha=10, max_iter=500, no early stopping)).

Usage
-----
  python QNN_hyperparameter_search.py --stage selftest      # gradient / sanity checks (fast)
  python QNN_hyperparameter_search.py --stage 1 --dry-run   # show the plan, run nothing
  python QNN_hyperparameter_search.py --stage 1
  python QNN_hyperparameter_search.py --stage 2
  python QNN_hyperparameter_search.py --stage report        # rebuild figures / summary only
  python QNN_hyperparameter_search.py --stage all

Every individual training (config x seed x fold x target) is written to
<outdir>/train_log.jsonl as soon as it finishes, so an interrupted run resumes
exactly where it stopped (finished trainings are never recomputed).
"""

import argparse
import itertools
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pennylane as qml
from pennylane import numpy as pnp

warnings.filterwarnings("ignore")

# =============================================================================
# Fixed methodology (kept from the original script)
# =============================================================================
SELECTED_RADIO = [
    "original_shape_MajorAxis", "original_shape_Sphericity",
    "original_glcm_JointEntropy", "original_glcm_MaximumProbability",
    "log-sigma-3-0-mm-3D_glszm_SizeZoneNonUniformity",
    "log-sigma-5-0-mm-3D_firstorder_RootMeanSquared",
    "wavelet-LLL_firstorder_Kurtosis", "wavelet-LHH_firstorder_Variance",
    "wavelet-HHL_firstorder_Skewness", "wavelet-HHL_gldm_LowGrayLevelEmphasis",
    "squareroot_glrlm_HighGrayLevelRunEmphasis", "squareroot_glrlm_RunVariance",
]
assert len(SELECTED_RADIO) == 12
N_QUBITS = len(SELECTED_RADIO)

#These prevent probabilities from becoming exactly 0 or 1 when calculating logarithms.
TRAIN_CLIP = 1e-6          # probability clipping inside the training loss (as in the original)
EVAL_CLIP_G = (1e-6, 1 - 1e-6)   # clipping used to evaluate g(X) predictions
EVAL_CLIP_M = (0.01, 0.99)       # same clipping as the DML propensity score (PROPENSITY_CLIP = 0.01)

# MLP exactly as in QDML_bootstrap_paired.py (only the imputer is added, fitted on the training fold)
MLP_HIDDEN, MLP_ALPHA, MLP_MAX_ITER = 32, 10.0, 500

# =============================================================================
# Quantum circuit: device and QNode are built ONCE and reused for every
# configuration (the number of layers is inferred from the shape of `weights`).
# =============================================================================
_dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(_dev, diff_method="adjoint")
def _circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def _predict(weights, X):
    raw = pnp.array([_circuit(weights, x) for x in X]) # For every patient x in X, run the quantum circuit using the same weights.
    return (1 - raw) / 2


def _bce(weights, X, y): #binary cross entropy
    p = pnp.clip(_predict(weights, X), TRAIN_CLIP, 1 - TRAIN_CLIP)
    return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))


#For this circuit, every quantum layer has three trainable rotation parameters per qubit.
def n_params(layers):
    return layers * N_QUBITS * 3


def train_qnn(Xtr_q, ytr, Xte_q, layers, lr, epochs, seed):
    """Same training procedure as the original: full-batch Adam, one step per 'epoch'.
    Returns test probabilities, the per-epoch training loss (value BEFORE each update)
    and the training loss after the last update."""
    rng = np.random.RandomState(seed)
    weights = pnp.array(0.1 * rng.randn(layers, N_QUBITS, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    y = pnp.array(ytr, dtype=float)

    def cost(w):
        return _bce(w, Xtr_q, y)

    curve = []
    for ep in range(epochs):
        weights, c = opt.step_and_cost(cost, weights)
        c = float(c)
        if not np.isfinite(c):
            raise FloatingPointError(f"non-finite training loss at epoch {ep + 1}")
        curve.append(c)
    final = float(cost(weights))
    preds = np.array(_predict(weights, Xte_q), dtype=float)
    if not (np.isfinite(final) and np.all(np.isfinite(preds)) and np.all(np.isfinite(np.asarray(weights)))):
        raise FloatingPointError("non-finite final loss, weights or predictions")
    return preds, curve, final


# =============================================================================
# Data
# =============================================================================
def load_data(path):
    df = pd.read_excel(path, header=3)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["name"]).dropna(axis=1, how="all").reset_index(drop=True)
    df["Y"] = df["name"].astype(str).str.endswith("GSM").astype(int)
    df["D"] = (df["Edema"] == "Y").astype(int)
    missing = [c for c in SELECTED_RADIO if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from the file: {missing}")
    X = df[SELECTED_RADIO].apply(pd.to_numeric, errors="coerce").values.astype(float)  # NaN kept
    return X, df["Y"].values, df["D"].values


def cfg_id(lr, epochs, layers):
    return f"lr{lr:g}_ep{epochs}_L{layers}"


# =============================================================================
# Persistent store of individual trainings (resume + audit trail)
# =============================================================================
class Store:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.data[self.key(r["cfg"], r["seed"], r["fold"], r["target"])] = r
                    except Exception:
                        continue  # truncated last line after an interruption

    @staticmethod
    def key(cid, seed, fold, target):
        return f"{cid}|{seed}|{fold}|{target}"

    def get(self, cid, seed, fold, target):
        return self.data.get(self.key(cid, seed, fold, target))

    def add(self, rec):
        self.data[self.key(rec["cfg"], rec["seed"], rec["fold"], rec["target"])] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# =============================================================================
# Metrics
# =============================================================================
def loss_vec(y, p, clip):
    pc = np.clip(p, clip[0], clip[1])
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def prob_metrics(y, p, clip):
    lv = loss_vec(y, p, clip)
    pc = np.clip(p, clip[0], clip[1])
    return {"LL": lv.mean(), "AUC": roc_auc_score(y, p), "Brier": np.mean((pc - y) ** 2),
            "pmin": p.min(), "pmax": p.max(), "pmean": p.mean()}, lv


def curve_diag(curve, window):
    c = np.asarray(curve, float)
    k = max(2, min(window, len(c) // 4))
    prev, last = c[-2 * k:-k].mean(), c[-k:].mean()
    rel = (prev - last) / abs(prev) if prev != 0 else 0.0
    osc = float(np.mean(c[1:] > c[:-1] * 1.01)) if len(c) > 1 else 0.0
    return rel, osc


class Context:
    def __init__(self, args, X, Y, D, store):
        self.args, self.X, self.Y, self.D, self.store = args, X, Y, D, store
        self.n = len(Y)
        self._folds, self._base = {}, {}

    def folds(self, seed):
        if seed not in self._folds:
            skf = StratifiedKFold(n_splits=self.args.n_folds, shuffle=True, random_state=seed)
            self._folds[seed] = list(skf.split(self.X, self.D))
        return self._folds[seed]

    def target(self, t):
        return self.Y if t == "g" else self.D

    def baselines(self, seed):
        """Prevalence-only and optimised MLP, on the SAME folds as the QNN."""
        if seed in self._base:
            return self._base[seed]
        n = self.n
        out = {k: {"g": np.zeros(n), "m": np.zeros(n)} for k in ("prev", "mlp")}
        nonconv = 0
        for tr, te in self.folds(seed):
            for t, key in (("g", "g"), ("m", "m")):
                y = self.target(t)
                out["prev"][key][te] = y[tr].mean()
                clf = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), activation="relu", alpha=MLP_ALPHA,
                                  max_iter=MLP_MAX_ITER, early_stopping=False, random_state=seed))
                clf.fit(self.X[tr], y[tr])
                out["mlp"][key][te] = clf.predict_proba(self.X[te])[:, 1]
                nonconv += int(clf[-1].n_iter_ >= clf[-1].max_iter)
        res = {}
        for name in ("prev", "mlp"):
            mg, lg = prob_metrics(self.Y, out[name]["g"], EVAL_CLIP_G)
            mm, lm = prob_metrics(self.D, out[name]["m"], EVAL_CLIP_M)
            res[name] = {"g": mg, "m": mm, "lg": lg, "lm": lm}
        res["mlp_nonconverged_fits"] = nonconv
        self._base[seed] = res
        return res


# =============================================================================
# Running trainings
# =============================================================================
def run_one_training(ctx, lr, epochs, layers, seed, fold, tr, te, target):
    cid = cfg_id(lr, epochs, layers)
    rec = {"cfg": cid, "lr": lr, "epochs": epochs, "layers": layers, "seed": seed, "fold": fold,
           "target": target, "n_params": n_params(layers), "status": "ok", "error": "",
           "preds": None, "curve": None, "final_loss": None, "time_s": None}
    t0 = time.time()
    try:
        Xtr, Xte = ctx.X[tr], ctx.X[te]
        imp = SimpleImputer(strategy="median").fit(Xtr)          # fitted on the training fold only
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        sc = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(Xtr)  # fitted on the training fold only
        preds, curve, final = train_qnn(sc.transform(Xtr), ctx.target(target)[tr], sc.transform(Xte),
                                        layers, lr, epochs, seed)
        rec.update(preds=preds.tolist(), curve=curve, final_loss=final)
    except Exception as e:
        rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
        with open(os.path.join(ctx.args.outdir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%F %T')}] {cid} seed={seed} fold={fold} target={target}\n"
                    f"{traceback.format_exc()}\n")
    rec["time_s"] = time.time() - t0
    return rec


def run_config_seed(ctx, lr, epochs, layers, seed, tag=""):
    cid = cfg_id(lr, epochs, layers)
    for fold, (tr, te) in enumerate(ctx.folds(seed)):
        for target in ("g", "m"):
            old = ctx.store.get(cid, seed, fold, target)
            if old is not None and (old["status"] == "ok" or not ctx.args.retry_failed):
                continue
            rec = run_one_training(ctx, lr, epochs, layers, seed, fold, tr, te, target)
            ctx.store.add(rec)
            print(f"  {tag}{cid} seed={seed} fold={fold + 1}/{ctx.args.n_folds} {target}: "
                  f"{rec['status']} in {rec['time_s']:.0f}s", flush=True)


def build_row(ctx, lr, epochs, layers, seed):
    """One row = one configuration evaluated with one seed (all folds, both nuisances)."""
    cid = cfg_id(lr, epochs, layers)
    row = {"cfg_id": cid, "lr": lr, "epochs": epochs, "layers": layers, "n_params": n_params(layers),
           "seed": seed, "status": "ok", "error": ""}
    recs, g_hat, m_hat = [], np.zeros(ctx.n), np.zeros(ctx.n)
    for fold, (tr, te) in enumerate(ctx.folds(seed)):
        for t in ("g", "m"):
            r = ctx.store.get(cid, seed, fold, t)
            if r is None:
                row["status"] = "incomplete" if row["status"] == "ok" else row["status"]
                continue
            recs.append(r)
            if r["status"] != "ok":
                row["status"], row["error"] = "failed", row["error"] or r["error"]
            elif row["status"] != "failed":
                (g_hat if t == "g" else m_hat)[te] = r["preds"]
    row["n_trainings_done"] = len(recs)
    row["time_run_s"] = float(sum(r["time_s"] for r in recs))
    if row["status"] != "ok":
        return row

    mg, lg = prob_metrics(ctx.Y, g_hat, EVAL_CLIP_G)
    mm, lm = prob_metrics(ctx.D, m_hat, EVAL_CLIP_M)
    for k, v in mg.items():
        row[f"g_{k}"] = v
    for k, v in mm.items():
        row[f"m_{k}"] = v
    row["LL_total"] = mg["LL"] + mm["LL"]

    b = ctx.baselines(seed)
    sq = np.sqrt(ctx.n)
    for name in ("prev", "mlp"):
        row[f"{name}_LL_g"], row[f"{name}_LL_m"] = b[name]["g"]["LL"], b[name]["m"]["LL"]
        row[f"{name}_LL_total"] = b[name]["g"]["LL"] + b[name]["m"]["LL"]
    d = lm - b["prev"]["lm"]
    row["d_m_vs_prev"], row["d_m_vs_prev_se"] = d.mean(), d.std(ddof=1) / sq
    d = (lg + lm) - (b["prev"]["lg"] + b["prev"]["lm"])
    row["d_total_vs_prev"] = d.mean()
    d = (lg + lm) - (b["mlp"]["lg"] + b["mlp"]["lm"])
    row["d_total_vs_mlp"], row["d_total_vs_mlp_se"] = d.mean(), d.std(ddof=1) / sq
    row["mlp_nonconverged_fits"] = b["mlp_nonconverged_fits"]

    diags = [curve_diag(r["curve"], ctx.args.conv_window) for r in recs]
    row["loss_init"] = float(np.mean([r["curve"][0] for r in recs]))
    row["loss_final"] = float(np.mean([r["final_loss"] for r in recs]))
    row["rel_impr_last"] = float(np.mean([d[0] for d in diags]))
    row["osc_frac"] = float(np.mean([d[1] for d in diags]))
    row["not_converged"] = bool(row["rel_impr_last"] > ctx.args.conv_tol)
    row["unstable"] = bool(row["osc_frac"] > ctx.args.osc_tol)
    for t in ("g", "m"):
        row[f"train_loss_final_{t}"] = float(np.mean([r["final_loss"] for r in recs if r["target"] == t]))
    return row


# =============================================================================
# Stages
# =============================================================================
def path(ctx, name):
    return os.path.join(ctx.args.outdir, name)


def stage1_configs(a):
    return [(lr, ep, a.stage1_layers) for lr, ep in itertools.product(a.lrs, a.epochs)]


def run_stage1(ctx):
    a = ctx.args
    cfgs = stage1_configs(a)
    print(f"\n=== STAGE 1: {len(cfgs)} configurations, seed {a.stage1_seed}, "
          f"{len(cfgs) * a.n_folds * 2} trainings ===")
    if a.dry_run:
        return
    rows = []
    for lr, ep, L in cfgs:
        run_config_seed(ctx, lr, ep, L, a.stage1_seed, tag="[S1] ")
        rows.append(build_row(ctx, lr, ep, L, a.stage1_seed))
        pd.DataFrame(rows).to_csv(path(ctx, "stage1_results.csv"), index=False)  # progressive save
    print(pd.DataFrame(rows)[["cfg_id", "status", "LL_total"] if "LL_total" in rows[0] or
                              any("LL_total" in r for r in rows) else ["cfg_id", "status"]].to_string(index=False))


def run_stage2(ctx):
    a = ctx.args
    p1 = path(ctx, "stage1_results.csv")
    if not os.path.exists(p1):
        sys.exit("stage1_results.csv not found: run --stage 1 first.")
    s1 = pd.read_csv(p1)
    valid = s1[(s1["status"] == "ok")].sort_values("LL_total") if "LL_total" in s1 else s1.iloc[0:0]
    if valid.empty:
        sys.exit("No valid stage-1 configuration: stage 2 cannot start (see failures.log).")
    cands = valid.head(a.top_k)
    cfgs = []
    for _, c in cands.iterrows():
        for L in a.layers:
            cfgs.append((float(c["lr"]), int(c["epochs"]), int(L)))
            if bool(c["not_converged"]) and int(c["epochs"]) < a.long_epochs:
                cfgs.append((float(c["lr"]), int(a.long_epochs), int(L)))
    cfgs = list(dict.fromkeys(cfgs))
    plan = {"candidates": cands[["cfg_id", "LL_total", "not_converged", "unstable"]].to_dict("records"),
            "configs": cfgs, "seeds": a.seeds}
    json.dump(plan, open(path(ctx, "stage2_plan.json"), "w"), indent=1)
    todo = sum(1 for c in cfgs for s in a.seeds for f in range(a.n_folds) for t in "gm"
               if ctx.store.get(cfg_id(*c), s, f, t) is None)
    print(f"\n=== STAGE 2: {len(cfgs)} configurations x {len(a.seeds)} seeds; "
          f"{todo} trainings still to run (finished ones are reused) ===")
    for c in cfgs:
        print("  ", cfg_id(*c))
    if a.dry_run:
        return
    rows = []
    for c in cfgs:
        for s in a.seeds:
            run_config_seed(ctx, *c, s, tag="[S2] ")
            rows.append(build_row(ctx, *c, s))
            pd.DataFrame(rows).to_csv(path(ctx, "stage2_results.csv"), index=False)  # progressive save


# =============================================================================
# Aggregation, ranking, reports
# =============================================================================
MET = ["g_LL", "g_AUC", "g_Brier", "g_pmin", "g_pmax", "g_pmean", "m_LL", "m_AUC", "m_Brier", "m_pmin",
       "m_pmax", "m_pmean", "LL_total", "prev_LL_g", "prev_LL_m", "prev_LL_total", "mlp_LL_g", "mlp_LL_m",
       "mlp_LL_total", "d_m_vs_prev", "d_m_vs_prev_se", "d_total_vs_prev", "d_total_vs_mlp",
       "d_total_vs_mlp_se", "loss_init", "loss_final", "rel_impr_last", "osc_frac", "time_run_s"]


def aggregate(df):
    out = []
    for cid, g in df.groupby("cfg_id", sort=False):
        ok = g[g["status"] == "ok"]
        r = {"cfg_id": cid, "lr": g["lr"].iloc[0], "epochs": g["epochs"].iloc[0], "layers": g["layers"].iloc[0],
             "n_params": g["n_params"].iloc[0], "n_seeds_expected": len(g), "n_seeds_ok": len(ok)}
        if len(ok) == len(g):
            r["status"] = "ok"
        elif (g["status"] == "failed").any():
            r["status"] = "failed"
        else:
            r["status"] = "incomplete"
        r["error"] = next((e for e in g["error"].fillna("") if e), "")
        for m in MET:
            if m in ok:
                r[m + "_mean"] = ok[m].mean()
                r[m + "_sd"] = ok[m].std(ddof=1) if len(ok) > 1 else np.nan
        if len(ok):
            r["not_converged_frac"] = ok["not_converged"].mean()
            r["unstable_frac"] = ok["unstable"].mean()
            r["m_better_than_prev_seeds"] = int((ok["d_m_vs_prev"] < 0).sum())
        out.append(r)
    return pd.DataFrame(out)


def baseline_rows(ctx, seeds):
    rows = {"BASELINE_PREVALENCE": [], "BASELINE_MLP": []}
    for s in seeds:
        b = ctx.baselines(s)
        for name, key in (("BASELINE_PREVALENCE", "prev"), ("BASELINE_MLP", "mlp")):
            r = {}
            for t in ("g", "m"):
                for k, v in b[key][t].items():
                    r[f"{t}_{k}"] = v
            r["LL_total"] = r["g_LL"] + r["m_LL"]
            rows[name].append(r)
    out = []
    for name, lst in rows.items():
        d = pd.DataFrame(lst)
        r = {"cfg_id": name, "status": "baseline", "n_seeds_ok": len(d)}
        for c in d.columns:
            r[c + "_mean"] = d[c].mean()
            r[c + "_sd"] = d[c].std(ddof=1) if len(d) > 1 else np.nan
        out.append(r)
    return pd.DataFrame(out)


def make_reports(ctx):
    p1, p2 = path(ctx, "stage1_results.csv"), path(ctx, "stage2_results.csv")
    if not os.path.exists(p1):
        print("No stage-1 results yet: nothing to report.")
        return
    s1 = pd.read_csv(p1)
    use2 = os.path.exists(p2)
    df = pd.read_csv(p2) if use2 else s1
    source = "stage 2" if use2 else "stage 1 (single seed)"
    rk = aggregate(df)
    seeds = sorted(df["seed"].unique())
    base = baseline_rows(ctx, seeds)

    ok = rk[rk["status"] == "ok"].sort_values("LL_total_mean")
    bad = rk[rk["status"] != "ok"]
    rk = pd.concat([ok.assign(rank=range(1, len(ok) + 1)), bad], ignore_index=True)
    pd.concat([rk, base], ignore_index=True).to_csv(path(ctx, "qnn_configuration_ranking.csv"), index=False)

    # ---------------- best configuration ----------------
    best = None
    if len(ok):
        best = ok.iloc[0]
        thr = best["LL_total_mean"] + (best["LL_total_sd"] if pd.notna(best["LL_total_sd"]) else 0.0)
        simple = ok[ok["LL_total_mean"] <= thr].sort_values(["n_params", "LL_total_mean"]).iloc[0]
    pb, mb = base[base.cfg_id == "BASELINE_PREVALENCE"].iloc[0], base[base.cfg_id == "BASELINE_MLP"].iloc[0]

    def verdict(delta, se, ref):
        if delta + 1.96 * se < 0:
            return f"IMPROVES on {ref} (paired difference below 0 by more than 1.96 SE)"
        if delta < 0:
            return f"slightly lower than {ref}, but within the uncertainty (not a clear improvement)"
        return f"does NOT improve on {ref}"

    js = {"criterion": "lowest mean out-of-fold LL_total = LL_g + LL_m across seeds; failed/incomplete configurations excluded",
          "source": source, "seeds": [int(s) for s in seeds], "best": None}
    lines = ["QNN HYPERPARAMETER SEARCH: SUMMARY", "=" * 70,
             f"Source of the ranking: {source}; seeds: {seeds}; folds: {ctx.args.n_folds}",
             "Selection uses out-of-fold predictive performance only (no theta, p-value or CI).", ""]
    if best is None:
        lines += ["NO VALID CONFIGURATION: every configuration failed or is incomplete.",
                  "See failures.log and rerun with --retry-failed or fix the cause."]
    else:
        sd = lambda c: best.get(c, np.nan)
        v_m = verdict(best["d_m_vs_prev_mean"], best["d_m_vs_prev_se_mean"], "prevalence-only (edema)")
        v_mlp = verdict(best["d_total_vs_mlp_mean"], best["d_total_vs_mlp_se_mean"], "the optimised MLP (LL_total)")
        js["best"] = {"lr": float(best["lr"]), "epochs": int(best["epochs"]), "layers": int(best["layers"]),
                      "n_qubits": N_QUBITS, "n_params": int(best["n_params"]), "cfg_id": best["cfg_id"],
                      "LL_total_mean": float(best["LL_total_mean"]), "LL_total_sd": None if pd.isna(sd("LL_total_sd")) else float(sd("LL_total_sd")),
                      "LL_g_mean": float(best["g_LL_mean"]), "LL_m_mean": float(best["m_LL_mean"]),
                      "AUC_g_mean": float(best["g_AUC_mean"]), "AUC_m_mean": float(best["m_AUC_mean"]),
                      "not_converged_frac": float(best["not_converged_frac"]), "unstable_frac": float(best["unstable_frac"]),
                      "edema_vs_prevalence": v_m, "total_vs_mlp": v_mlp,
                      "parsimonious_alternative_within_1sd": simple["cfg_id"]}
        f = lambda x: "n/a" if pd.isna(x) else f"{x:.4f}"
        lines += [f"BEST CONFIGURATION: {best['cfg_id']}  (lr={best['lr']:g}, epochs={int(best['epochs'])}, "
                  f"layers={int(best['layers'])}, {int(best['n_params'])} trainable parameters, {N_QUBITS} qubits)",
                  f"  LL_total = {f(best['LL_total_mean'])} +/- {f(sd('LL_total_sd'))} (SD across seeds)",
                  f"  Parsimonious alternative within 1 SD of the best: {simple['cfg_id']} ({int(simple['n_params'])} parameters)", "",
                  "PREDICTIVE PERFORMANCE (mean over seeds, out-of-fold)",
                  f"  g(X) tumor type : LL={f(best['g_LL_mean'])}  AUC={f(best['g_AUC_mean'])}  Brier={f(best['g_Brier_mean'])}  "
                  f"p in [{f(best['g_pmin_mean'])}, {f(best['g_pmax_mean'])}], mean {f(best['g_pmean_mean'])}",
                  f"  m(X) edema      : LL={f(best['m_LL_mean'])}  AUC={f(best['m_AUC_mean'])}  Brier={f(best['m_Brier_mean'])}  "
                  f"p in [{f(best['m_pmin_mean'])}, {f(best['m_pmax_mean'])}], mean {f(best['m_pmean_mean'])}", "",
                  "BASELINES (identical folds)",
                  f"  Prevalence only : LL_g={f(pb['g_LL_mean'])}  LL_m={f(pb['m_LL_mean'])}  LL_total={f(pb['LL_total_mean'])}",
                  f"  Optimised MLP   : LL_g={f(mb['g_LL_mean'])}  LL_m={f(mb['m_LL_mean'])}  LL_total={f(mb['LL_total_mean'])}  "
                  f"(AUC_g={f(mb['g_AUC_mean'])}, AUC_m={f(mb['m_AUC_mean'])})", "",
                  "EDEMA (161 with / 22 without): does the QNN beat prevalence alone?",
                  f"  LL_m(QNN) - LL_m(prevalence) = {f(best['d_m_vs_prev_mean'])} (paired SE over patients ~ {f(best['d_m_vs_prev_se_mean'])}); "
                  f"QNN better in {int(best['m_better_than_prev_seeds'])}/{int(best['n_seeds_ok'])} seeds",
                  f"  -> {v_m}",
                  f"  LL_total(QNN) - LL_total(MLP) = {f(best['d_total_vs_mlp_mean'])} (SE ~ {f(best['d_total_vs_mlp_se_mean'])}) -> {v_mlp}", "",
                  "TRAINING STABILITY (best configuration)",
                  f"  mean training loss: first epoch {f(best['loss_init_mean'])} -> final {f(best['loss_final_mean'])}",
                  f"  relative improvement over the last {ctx.args.conv_window} epochs: {f(best['rel_impr_last_mean'])} "
                  f"(flag 'not converged' if > {ctx.args.conv_tol}; flagged in {best['not_converged_frac']:.0%} of seeds)",
                  f"  fraction of epochs where the loss rose by >1%: {f(best['osc_frac_mean'])} "
                  f"(flag 'unstable' if > {ctx.args.osc_tol}; flagged in {best['unstable_frac']:.0%} of seeds)"]
    # failures, cost
    nfail = sum(1 for r in ctx.store.data.values() if r["status"] != "ok")
    times = [r["time_s"] for r in ctx.store.data.values()]
    lines += ["", "FAILURES / COST",
              f"  failed trainings recorded: {nfail} of {len(ctx.store.data)} (details in failures.log)",
              f"  total training time recorded: {sum(times) / 3600:.2f} h; mean per training: {np.mean(times):.0f} s"
              if times else "  no training recorded",
              f"  MLP baseline fits that hit max_iter (not converged): see column mlp_nonconverged_fits in stage results"]
    if len(ok):
        lines += ["", "RANKING (top 10, lower LL_total is better)"]
        for _, r in ok.head(10).iterrows():
            lines.append(f"  {int(r['rank']) if 'rank' in r and pd.notna(r.get('rank')) else '-':>2}  {r['cfg_id']:<22} "
                         f"LL_total={r['LL_total_mean']:.4f} (sd {0 if pd.isna(r['LL_total_sd']) else r['LL_total_sd']:.4f})  "
                         f"time/run={r['time_run_s_mean']:.0f}s  not_conv={r['not_converged_frac']:.0%} unstable={r['unstable_frac']:.0%}")
    if len(bad):
        lines += ["", "EXCLUDED (cannot be selected): " + ", ".join(f"{r.cfg_id} [{r.status}]" for r in bad.itertuples())]
    open(path(ctx, "optimization_summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(js, open(path(ctx, "best_qnn_config.json"), "w"), indent=2)

    # ---------------- figures ----------------
    top = ok.head(8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, t, ttl in zip(axes, ("g", "m"), ("g(X): tumor type", "m(X): edema")):
        for _, r in top.iterrows():
            L = int(r["epochs"])
            cs = [np.array(v["curve"]) for v in ctx.store.data.values()
                  if v["cfg"] == r["cfg_id"] and v["target"] == t and v["status"] == "ok" and v["seed"] in seeds]
            if cs:
                ax.plot(range(1, L + 1), np.mean(cs, axis=0), lw=2.5 if r["cfg_id"] == best["cfg_id"] else 1.2,
                        label=r["cfg_id"])
        ax.set_title(ttl); ax.set_xlabel("epoch (one Adam step)"); ax.set_ylabel("training BCE (before the step)")
    axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path(ctx, "training_loss_curves.png"), dpi=150); plt.close(fig)

    if best is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        labels, w = ["LL_g", "LL_m", "LL_total"], 0.26
        for i, (name, row, pre) in enumerate((("best QNN", best, ""), ("MLP", mb, ""), ("prevalence", pb, ""))):
            vals = [row["g_LL_mean"], row["m_LL_mean"], row["LL_total_mean"]]
            sds = [row.get("g_LL_sd", np.nan), row.get("m_LL_sd", np.nan), row.get("LL_total_sd", np.nan)]
            ax.bar(np.arange(3) + (i - 1) * w, vals, w, yerr=[0 if pd.isna(s) else s for s in sds], label=name, capsize=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylabel("out-of-fold log-loss (lower is better)")
        ax.legend(); plt.tight_layout(); plt.savefig(path(ctx, "logloss_comparison.png"), dpi=150); plt.close(fig)
    print("\n".join(lines))


# =============================================================================
# Self-test: gradients, determinism, learning
# =============================================================================
def self_test():
    rng = np.random.RandomState(0)
    X = rng.uniform(-np.pi, np.pi, size=(8, N_QUBITS))
    y = pnp.array((rng.rand(8) > 0.5).astype(float))
    for layers in (1, 2):
        w = pnp.array(0.1 * rng.randn(layers, N_QUBITS, 3), requires_grad=True)
        g = np.asarray(qml.grad(lambda w_: _bce(w_, X, y))(w))
        idx = [tuple(rng.randint(s) for s in w.shape) for _ in range(6)]
        err, eps = 0.0, 1e-5
        for i in idx:
            wp, wm = np.array(w), np.array(w)
            wp[i] += eps; wm[i] -= eps
            fd = (float(_bce(pnp.array(wp), X, y)) - float(_bce(pnp.array(wm), X, y))) / (2 * eps)
            err = max(err, abs(fd - g[i]))
        print(f"layers={layers}: |grad|={np.linalg.norm(g):.4f}, max |adjoint - finite diff| over 6 entries = {err:.2e}")
        assert np.linalg.norm(g) > 0, "zero gradient"
        assert err < 1e-5, "adjoint gradient disagrees with finite differences"
    a, _, _ = train_qnn(X, np.asarray(y), X, 1, 0.1, 3, seed=7)
    b, curve, _ = train_qnn(X, np.asarray(y), X, 1, 0.1, 3, seed=7)
    print("same seed -> identical predictions:", np.allclose(a, b), "| loss curve:", np.round(curve, 4))
    print("SELF-TEST OK")


# =============================================================================
def parse():
    p = argparse.ArgumentParser(description="Two-stage QNN hyperparameter search (nuisance prediction).")
    p.add_argument("--stage", choices=["1", "2", "all", "report", "selftest"], required=True)
    p.add_argument("--data", default="data/GBM_GBS.xlsx")
    p.add_argument("--outdir", default="resultats_qnn_optimisation")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    p.add_argument("--epochs", type=int, nargs="+", default=[40, 80])
    p.add_argument("--stage1-layers", type=int, default=1)
    p.add_argument("--stage1-seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    p.add_argument("--long-epochs", type=int, default=120)
    p.add_argument("--conv-window", type=int, default=10, help="epochs compared at the end of training")
    p.add_argument("--conv-tol", type=float, default=0.01, help="relative loss improvement over the window above which training is 'not converged'")
    p.add_argument("--osc-tol", type=float, default=0.10, help="fraction of epochs with a >1%% loss rise above which training is 'unstable'")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    if a.stage == "selftest":
        self_test()
        return
    os.makedirs(a.outdir, exist_ok=True)
    X, Y, D = load_data(a.data)
    print(f"n = {len(Y)} | p = {X.shape[1]} | edema: {D.sum()} with / {len(D) - D.sum()} without "
          f"| GSM: {Y.sum()} | NaN cells: {int(np.isnan(X).sum())}")
    ctx = Context(a, X, Y, D, Store(os.path.join(a.outdir, "train_log.jsonl")))
    if a.stage in ("1", "all"):
        run_stage1(ctx)
    if a.stage in ("2", "all"):
        run_stage2(ctx)
    if not a.dry_run:
        make_reports(ctx)


if __name__ == "__main__":
    main()