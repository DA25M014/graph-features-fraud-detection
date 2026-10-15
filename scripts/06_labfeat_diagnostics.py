"""Label-feature and data diagnostics behind Sec. 2, Sec. 4.1, Sec. 4.2 and
Table 2 of the LoG 2026 extended abstract (#127, camera-ready).

Prints, and writes to CSV, every number in the paper that is a property of
the data or of the label features rather than of a fitted model:

  - YelpChi (r=0.40, seeds 0-4): train-row vs test-row means of the four
    label features, with and without the self-walk leak (Sec. 4.1 quotes
    seed 0, leaky: h2 fraud fraction 0.1385 vs 0.1377, h2 train-labeled
    fraction 0.4046 vs 0.3945);
  - Elliptic (temporal split: fit t<=29, test t>=35): the same means for the
    clean label features, and whether every test value is exactly 0
    (Table 2; 04_revision_figures.py builds table2_labfeat.tex from this CSV);
  - per dataset: raw feature count, two-valued raw columns, relation count,
    rung widths (a), (b), (c) and the label-only control (Sec. 2);
  - Elliptic: connected components and edges crossing timesteps (Sec. 4.2).

Writes:
  results/labfeat_diagnostics.csv   one row per (dataset, seed, variant, feature)
  results/data_diagnostics.csv      one row per (dataset, quantity)

Usage: .venv/bin/python scripts/06_labfeat_diagnostics.py \
           | tee logs/labfeat_diagnostics.log
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loaders import LOADERS                             # noqa: E402
from src.eval.harness import random_split, temporal_split        # noqa: E402
from src.features.graph_features import (                        # noqa: E402
    label_graph_features, static_graph_features)

HERE = os.path.join(os.path.dirname(__file__), "..")
OUT_LAB = os.path.join("results", "labfeat_diagnostics.csv")
OUT_DATA = os.path.join("results", "data_diagnostics.csv")


def main():
    lab_rows, data_rows = [], []

    def note(dataset, quantity, value):
        data_rows.append([dataset, quantity, value])
        print(f"  {dataset:9s} {quantity:40s} {value}", flush=True)

    for d in ["yelp", "amazon", "elliptic"]:
        gd = LOADERS[d]()
        print(gd.summary(), flush=True)
        F, R = gd.x.shape[1], len(gd.relations)
        two = [j for j in range(F) if len(np.unique(gd.x[:, j])) == 2]
        static_b, _ = static_graph_features(gd)
        note(d, "raw features F", F)
        note(d, "two-valued raw columns", len(two))
        note(d, "relation-specific graphs R", R)
        note(d, "rung (a) width", F)
        note(d, "rung (b) width", F + static_b.shape[1])
        note(d, "rung (c) width", F + static_b.shape[1] + 4)
        note(d, "label-only control width", F + 4)
        if d == "amazon":
            note(d, "first evaluable node index", int(np.argmax(gd.eval_mask)))

        if d == "elliptic":
            ts = gd.timesteps
            coo = sp.triu(gd.adj, k=1).tocoo()
            ncomp, _ = connected_components(gd.adj, directed=False)
            note(d, "undirected edges", int(coo.nnz))
            note(d, "edges crossing timesteps",
                 int((ts[coo.row] != ts[coo.col]).sum()))
            note(d, "connected components", int(ncomp))
            note(d, "timesteps", int(len(np.unique(ts))))
            tr, _va, te = temporal_split(gd.y, ts)
            splits = [(None, tr, te)]
            variants = [("clean", False)]
        elif d == "yelp":
            splits = []
            for seed in range(5):
                tr, _va, te = random_split(gd.y, gd.eval_mask, 0.4, seed)
                splits.append((seed, tr, te))
            variants = [("leaky", True), ("clean", False)]
        else:
            continue

        for seed, tr, te in splits:
            mask = np.isin(np.arange(gd.n), tr)
            for vname, leaky in variants:
                L, names = label_graph_features(gd, train_mask=mask,
                                                legacy_leaky=leaky)
                for j, nm in enumerate(names):
                    mtr, mte = float(L[tr, j].mean()), float(L[te, j].mean())
                    test_all_zero = bool((L[te, j] == 0).all())
                    lab_rows.append([d, "" if seed is None else seed, vname, nm,
                                     f"{mtr:.6f}", f"{mte:.6f}",
                                     int(test_all_zero), len(tr), len(te)])
                    print(f"  {d:9s} seed {seed} {vname:5s} {nm:24s} "
                          f"train {mtr:.4f}  test {mte:.4f}"
                          f"{'  (every test value is 0)' if test_all_zero else ''}",
                          flush=True)

    with open(os.path.join(HERE, OUT_LAB), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "variant", "feature", "train_mean",
                    "test_mean", "test_all_zero", "n_train", "n_test"])
        w.writerows(lab_rows)
    with open(os.path.join(HERE, OUT_DATA), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "quantity", "value"])
        w.writerows(data_rows)
    print(f"\nwrote {OUT_LAB}\nwrote {OUT_DATA}", flush=True)


if __name__ == "__main__":
    main()
