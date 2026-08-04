"""Splits, metrics, and the single source of truth: results/results.csv.

Metrics: AUPRC primary (heavy class imbalance makes AUROC flattering),
best-F1 and AUROC secondary for cross-paper comparability.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "results.csv")
COLS = ["dataset", "rung", "model", "split", "train_rate", "seed",
        "auprc", "auroc", "best_f1", "n_train", "n_test", "wall_s", "notes"]


def metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    p, r, _ = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return {"auprc": float(average_precision_score(y_true, proba)),
            "auroc": float(roc_auc_score(y_true, proba)),
            "best_f1": float(np.nanmax(f1))}


def random_split(y: np.ndarray, eval_mask: np.ndarray, train_rate: float,
                 seed: int, val_rate: float = 0.1):
    """Stratified train/val/test over evalable labeled nodes.
    Returns (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    idx = np.where(eval_mask & (y >= 0))[0]
    tr, va = [], []
    for cls in (0, 1):
        c = idx[y[idx] == cls]
        c = rng.permutation(c)
        ntr = int(round(train_rate * len(c)))
        nva = int(round(val_rate * len(c)))
        tr.append(c[:ntr])
        va.append(c[ntr:ntr + nva])
    train = np.concatenate(tr)
    val = np.concatenate(va)
    rest = np.setdiff1d(idx, np.concatenate([train, val]))
    return train, val, rest


def temporal_split(y: np.ndarray, timesteps: np.ndarray,
                   train_end: int = 34, val_from: int = 30):
    """Elliptic convention: train ts<=34, test ts>=35; val = tail of train."""
    lab = y >= 0
    train = np.where(lab & (timesteps <= train_end) & (timesteps < val_from))[0]
    val = np.where(lab & (timesteps >= val_from) & (timesteps <= train_end))[0]
    test = np.where(lab & (timesteps > train_end))[0]
    return train, val, test


class ResultsWriter:
    def __init__(self, path: str = RESULTS):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(COLS)

    def add(self, **kw):
        kw.setdefault("notes", "")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([kw.get(c, "") for c in COLS])


def timer():
    t0 = time.time()
    return lambda: round(time.time() - t0, 2)
