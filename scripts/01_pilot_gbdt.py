"""Pilot: rung (a) features-only, (b0) +static graph feats, (b) +label feats.

(b)-(b0) isolates the marginal value of the label-propagation features.

Usage:
    python scripts/01_pilot_gbdt.py --datasets yelp amazon \
        --models lightgbm xgboost --rungs a b0 b --train-rate 0.4 --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loaders import LOADERS                             # noqa: E402
from src.eval.harness import ResultsWriter, metrics, random_split, temporal_split, timer  # noqa: E402
from src.features.graph_features import (                        # noqa: E402
    edge_homophily, label_graph_features, static_graph_features)
from src.models.tabular import fit_predict                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["yelp", "amazon"])
    ap.add_argument("--models", nargs="+", default=["lightgbm", "xgboost"])
    ap.add_argument("--rungs", nargs="+", default=["a", "b0", "b"])
    ap.add_argument("--train-rate", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    rw = ResultsWriter()
    summary = []

    for dname in args.datasets:
        gd = LOADERS[dname]()
        print(f"\n== {gd.summary()}")
        print(f"   edge homophily (labeled pairs): {edge_homophily(gd):.3f}")

        static_b, _ = static_graph_features(gd)

        for seed in range(args.seeds):
            if dname == "elliptic":
                tr, va, te = temporal_split(gd.y, gd.timesteps)
                split = "temporal"
                if seed > 0 and "lightgbm" not in args.models:
                    break  # temporal split is deterministic; seeds vary model only
            else:
                tr, va, te = random_split(gd.y, gd.eval_mask, args.train_rate, seed)
                split = "random"

            lab_b, lab_names = label_graph_features(
                gd, train_mask=np.isin(np.arange(gd.n), tr))
            if seed == 0:
                # diagnostic: label-feature reachability across the split
                for nm, trm, tem in zip(lab_names, lab_b[tr].mean(0),
                                        lab_b[te].mean(0)):
                    print(f"   [labfeat] {nm}: train_mean={trm:.4f} "
                          f"test_mean={tem:.4f}")

            X = {"a": gd.x,
                 "b0": np.hstack([gd.x, static_b]),
                 "b": np.hstack([gd.x, static_b, lab_b])}
            if "b_leaky" in args.rungs:
                lab_leaky, _ = label_graph_features(
                    gd, train_mask=np.isin(np.arange(gd.n), tr),
                    legacy_leaky=True)
                X["b_leaky"] = np.hstack([gd.x, static_b, lab_leaky])

            for rung in args.rungs:
                for mname in args.models:
                    t = timer()
                    proba = fit_predict(mname, seed, X[rung], gd.y, tr, te)
                    m = metrics(gd.y[te], proba)
                    rw.add(dataset=dname, rung=rung, model=mname, split=split,
                           train_rate=args.train_rate, seed=seed,
                           n_train=len(tr), n_test=len(te), wall_s=t(), **m)
                    summary.append((dname, rung, mname, m["auprc"], m["auroc"]))
                    print(f"   seed{seed} rung-{rung} {mname:9s} "
                          f"AUPRC {m['auprc']:.4f}  AUROC {m['auroc']:.4f}  "
                          f"F1* {m['best_f1']:.4f}")

    print("\n== mean over seeds ==")
    arr = {}
    for d, r, mo, ap_, ro in summary:
        arr.setdefault((d, r, mo), []).append((ap_, ro))
    for k in sorted(arr):
        v = np.array(arr[k])
        print(f"   {k[0]:9s} rung-{k[1]} {k[2]:9s} "
              f"AUPRC {v[:,0].mean():.4f}±{v[:,0].std():.4f}  "
              f"AUROC {v[:,1].mean():.4f}±{v[:,1].std():.4f}")


if __name__ == "__main__":
    main()
