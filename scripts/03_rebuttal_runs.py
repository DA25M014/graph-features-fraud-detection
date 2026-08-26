"""Rebuttal experiments for LoG 2026 extended abstract #127.

Addresses reviewer asks with real runs:
  1. Missing rung "aL" = raw + label-propagation features ONLY (no static
     aggregates). Separates "label features are uninformative" from "label
     features are redundant given static aggregates" (Reviewer uDZK W4).
  2. Label-rate sweep r in {0.05, 0.1, 0.2, 0.4} on YelpChi/Amazon over all
     four rungs (Reviewers od4H, FMA3: does the null result on label features
     persist at low label rates?).
  3. Elliptic, all rungs, 5 seeds: documents whether temporal-split fits are
     bit-identical across model seeds (Reviewer uDZK W5).
  4. Gain-importance shares of raw/static/label feature groups for rungs b
     and b_leaky (Reviewer uDZK Q3: leak severity vs reliance on label feats).

Writes results/rebuttal.csv and results/rebuttal_importance.csv. Never touches
results/results.csv. Reuses the exact pipeline of scripts/01_pilot_gbdt.py.

Usage: .venv/bin/python scripts/03_rebuttal_runs.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loaders import LOADERS                             # noqa: E402
from src.eval.harness import metrics, random_split, temporal_split, timer  # noqa: E402
from src.features.graph_features import (                        # noqa: E402
    label_graph_features, static_graph_features)
from src.models.tabular import get_model                         # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(HERE, "results", "rebuttal.csv")
OUT_IMP = os.path.join(HERE, "results", "rebuttal_importance.csv")
COLS = ["dataset", "rung", "model", "split", "train_rate", "seed",
        "auprc", "auroc", "best_f1", "n_train", "n_test", "n_feats", "wall_s"]
IMP_COLS = ["dataset", "rung", "model", "seed", "train_rate",
            "share_raw", "share_static", "share_label",
            "n_raw", "n_static", "n_label"]

RATES = [0.05, 0.1, 0.2, 0.4]
RUNGS = ["a", "aL", "b0", "b"]
MODELS = ["lightgbm", "xgboost"]
SEEDS = range(5)


def _writer(path, cols):
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    return f, w


def fit_with_model(mname, seed, X, y, tr, te):
    ytr = y[tr]
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    model = get_model(mname, seed, spw)
    model.fit(X[tr], ytr)
    return model, model.predict_proba(X[te])[:, 1]


def gain_shares(model, mname, groups):
    """groups: list of (name, index_array). Returns share of total gain."""
    if mname == "lightgbm":
        gain = model.booster_.feature_importance(importance_type="gain")
        gain = np.asarray(gain, dtype=np.float64)
    else:
        sc = model.get_booster().get_score(importance_type="total_gain")
        gain = np.zeros(model.n_features_in_, dtype=np.float64)
        for k, v in sc.items():
            gain[int(k[1:])] = v
    tot = gain.sum()
    if tot <= 0:
        return {name: float("nan") for name, _ in groups}
    return {name: float(gain[idx].sum() / tot) for name, idx in groups}


def build_X(gd, static_b, lab):
    F = gd.x.shape[1]
    S = static_b.shape[1]
    return {
        "a":  (gd.x, F, 0, 0),
        "aL": (np.hstack([gd.x, lab]), F, 0, lab.shape[1]),
        "b0": (np.hstack([gd.x, static_b]), F, S, 0),
        "b":  (np.hstack([gd.x, static_b, lab]), F, S, lab.shape[1]),
    }


def main():
    f1, w = _writer(OUT, COLS)
    f2, wi = _writer(OUT_IMP, IMP_COLS)

    for dname in ["yelp", "amazon", "elliptic"]:
        gd = LOADERS[dname]()
        print(f"\n== {gd.summary()}", flush=True)
        static_b, _ = static_graph_features(gd)

        rates = [None] if dname == "elliptic" else RATES
        for rate in rates:
            for seed in SEEDS:
                if dname == "elliptic":
                    tr, va, te = temporal_split(gd.y, gd.timesteps)
                    split, logged_rate = "temporal", -1.0
                else:
                    tr, va, te = random_split(gd.y, gd.eval_mask, rate, seed)
                    split, logged_rate = "random", rate

                lab, _ = label_graph_features(
                    gd, train_mask=np.isin(np.arange(gd.n), tr))
                X = build_X(gd, static_b, lab)

                for rung in RUNGS:
                    Xr, F, S, L = X[rung]
                    for mname in MODELS:
                        t = timer()
                        model, proba = fit_with_model(
                            mname, seed, Xr, gd.y, tr, te)
                        m = metrics(gd.y[te], proba)
                        w.writerow([dname, rung, mname, split, logged_rate,
                                    seed, m["auprc"], m["auroc"], m["best_f1"],
                                    len(tr), len(te), Xr.shape[1], t()])
                        f1.flush()
                        print(f"  r={logged_rate} seed{seed} {rung:3s} "
                              f"{mname:9s} AUPRC {m['auprc']:.4f} "
                              f"nfeat {Xr.shape[1]}", flush=True)

        # Importance shares at the paper's operating point (r=0.4 / temporal),
        # rungs b and b_leaky, seed 0.
        if gd.timesteps is not None:
            tr, va, te = temporal_split(gd.y, gd.timesteps)
            logged_rate = -1.0
        else:
            tr, va, te = random_split(gd.y, gd.eval_mask, 0.4, 0)
            logged_rate = 0.4
        mask = np.isin(np.arange(gd.n), tr)
        lab_clean, _ = label_graph_features(gd, train_mask=mask)
        lab_leaky, _ = label_graph_features(gd, train_mask=mask,
                                            legacy_leaky=True)
        F, S, L = gd.x.shape[1], static_b.shape[1], lab_clean.shape[1]
        idx = [("share_raw", np.arange(F)),
               ("share_static", np.arange(F, F + S)),
               ("share_label", np.arange(F + S, F + S + L))]
        for rung, lb in [("b", lab_clean), ("b_leaky", lab_leaky)]:
            Xr = np.hstack([gd.x, static_b, lb])
            for mname in MODELS:
                model, _ = fit_with_model(mname, 0, Xr, gd.y, tr, te)
                sh = gain_shares(model, mname, idx)
                wi.writerow([dname, rung, mname, 0, logged_rate,
                             sh["share_raw"], sh["share_static"],
                             sh["share_label"], F, S, L])
                f2.flush()
                print(f"  [imp] {rung} {mname}: raw {sh['share_raw']:.3f} "
                      f"static {sh['share_static']:.3f} "
                      f"label {sh['share_label']:.3f}", flush=True)

    f1.close()
    f2.close()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
