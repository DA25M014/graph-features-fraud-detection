"""Rung (a)/(b) models: GBDTs + RF behind one interface.

Deliberately near-default hyperparameters for the pilot (pre-registered in
protocol.md S6): the paper's claim is about information sources, not tuning
wizardry. A light tuning pass happens ONCE, symmetrically across rungs, in
Phase 2 - never per-cell.
"""
from __future__ import annotations

import numpy as np


def get_model(name: str, seed: int, spw: float = 1.0):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, random_state=seed, n_jobs=-1,
            eval_metric="aucpr", tree_method="hist")
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown tabular model: {name}")


def fit_predict(name: str, seed: int, X: np.ndarray, y: np.ndarray,
                train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    ytr = y[train_idx]
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    model = get_model(name, seed, spw)
    model.fit(X[train_idx], ytr)
    return model.predict_proba(X[test_idx])[:, 1]
