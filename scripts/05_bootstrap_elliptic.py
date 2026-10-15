"""Test-set bootstrap CIs for the Elliptic contrasts (LoG 2026 #127, camera-ready).

Fulfils the rebuttal promise to Reviewer uDZK (W5): the Elliptic temporal split
is fixed and the near-default GBDTs are deterministic, so seed variation is
zero; uncertainty is instead quantified by a paired, nonparametric bootstrap
over the 16,670 labeled test nodes.

Rungs (paper labels -> code labels): (a)=a, aL=aL, (b)=b0, (c)=b,
(c) leaky=b_leaky. Every refit AUPRC is checked against results/results.csv
and results/rebuttal.csv before any bootstrap is run.

Writes:
  results/bootstrap_elliptic.csv          one row per (model, contrast)
  results/elliptic_test_predictions.npz   test-node scores per (model, rung)

Usage: .venv/bin/python scripts/05_bootstrap_elliptic.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loaders import LOADERS                             # noqa: E402
from src.eval.harness import temporal_split                      # noqa: E402
from src.features.graph_features import (                        # noqa: E402
    label_graph_features, static_graph_features)
from src.models.tabular import get_model                         # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(HERE, "results", "bootstrap_elliptic.csv")
PRED = os.path.join(HERE, "results", "elliptic_test_predictions.npz")
B = 2000
BOOT_SEED = 20261013
MODELS = ["lightgbm", "xgboost"]
# contrast name, minuend rung, subtrahend rung (code labels)
CONTRASTS = [
    ("static gain (b)-(a)", "b0", "a"),
    ("label features (c)-(b)", "b", "b0"),
    ("leaky label features (c)leaky-(b)", "b_leaky", "b0"),
    ("leak alone (c)leaky-(c)", "b_leaky", "b"),
    ("label-only control aL-(a)", "aL", "a"),
]


def published(dname, rung, mname):
    """Published AUPRC for this cell: results.csv for a/b0/b/b_leaky,
    rebuttal.csv for aL (temporal split, seed 0)."""
    for fname in ("results.csv", "rebuttal.csv"):
        path = os.path.join(HERE, "results", fname)
        with open(path) as f:
            for r in csv.DictReader(f):
                if (r["dataset"] == dname and r["rung"] == rung
                        and r["model"] == mname and r.get("seed", "0") == "0"
                        and r.get("split", "temporal") == "temporal"):
                    return float(r["auprc"])
    return None


def main():
    gd = LOADERS["elliptic"]()
    print(gd.summary(), flush=True)
    static_b, _ = static_graph_features(gd)
    tr, _va, te = temporal_split(gd.y, gd.timesteps)
    mask = np.isin(np.arange(gd.n), tr)
    lab, _ = label_graph_features(gd, train_mask=mask)
    lab_leaky, _ = label_graph_features(gd, train_mask=mask, legacy_leaky=True)
    X = {
        "a": gd.x,
        "aL": np.hstack([gd.x, lab]),
        "b0": np.hstack([gd.x, static_b]),
        "b": np.hstack([gd.x, static_b, lab]),
        "b_leaky": np.hstack([gd.x, static_b, lab_leaky]),
    }
    y_te = gd.y[te]
    ytr = gd.y[tr]
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    print(f"train {len(tr)}  test {len(te)}  test positives {int(y_te.sum())}",
          flush=True)

    preds = {}
    for mname in MODELS:
        for rung, Xr in X.items():
            model = get_model(mname, 0, spw)
            model.fit(Xr[tr], ytr)
            p = model.predict_proba(Xr[te])[:, 1]
            preds[(mname, rung)] = p
            ap = average_precision_score(y_te, p)
            pub = published("elliptic", rung, mname)
            ok = pub is not None and abs(ap - pub) < 1e-9
            print(f"  {mname:9s} {rung:8s} AUPRC {ap:.6f}  published "
                  f"{pub}  {'OK' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                sys.exit(f"refit does not reproduce the published value for "
                         f"{mname}/{rung}; refusing to bootstrap")

    np.savez_compressed(PRED, test_index=te, y_test=y_te,
                        **{f"{m}__{r}": v for (m, r), v in preds.items()})

    rng = np.random.default_rng(BOOT_SEED)
    n = len(te)
    idx = rng.integers(0, n, size=(B, n))
    rows = []
    for mname in MODELS:
        boot = {}
        for rung in X:
            p = preds[(mname, rung)]
            boot[rung] = np.array([average_precision_score(y_te[i], p[i])
                                   for i in idx])
        for cname, r1, r2 in CONTRASTS:
            point = (average_precision_score(y_te, preds[(mname, r1)])
                     - average_precision_score(y_te, preds[(mname, r2)]))
            d = boot[r1] - boot[r2]
            lo, hi = np.percentile(d, [2.5, 97.5])
            same_sign = float(np.mean(np.sign(d) == np.sign(point)))
            rows.append([mname, cname, r1, r2, point, lo, hi, same_sign, B, n,
                         int(y_te.sum()), BOOT_SEED])
            print(f"  {mname:9s} {cname:36s} {point:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  sign-consistent {same_sign:.3f}",
                  flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "contrast", "minuend", "subtrahend", "point",
                    "ci_lo_2.5", "ci_hi_97.5", "frac_same_sign", "B",
                    "n_test", "n_test_pos", "boot_seed"])
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}\n"
          f"wrote {os.path.relpath(PRED, HERE)}", flush=True)


if __name__ == "__main__":
    main()
