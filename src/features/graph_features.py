"""Rung (b): graph-derived features for tabular models.

Two families:
1. STATIC (split-independent, compute once per dataset):
   - log-degree on homo graph + each relation graph
   - 1-hop mean & max of raw features (row-normalized adjacency)
   - 2-hop mean of raw features
2. LABEL-DEPENDENT (recompute per split; uses TRAIN labels only -> leakage-safe
   by construction):
   - fraction of neighbors that are train-labeled
   - fraud fraction among train-labeled neighbors (0 where none)
   - same two at 2 hops

The amplifier hypothesis lives or dies here: if rung (b) closes the gap to
GNNs, the graph's value is feature-extractable ("amplifier"); if GNNs retain
an edge, message passing adds something aggregation can't ("source").
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.data.loaders import GraphData


def _row_norm(adj: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(adj.sum(1)).flatten()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv).dot(adj).tocsr()


def static_graph_features(gd: GraphData) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []

    deg = np.asarray(gd.adj.sum(1)).flatten()
    cols.append(np.log1p(deg))
    names.append("logdeg_homo")
    for rname, radj in gd.relations.items():
        cols.append(np.log1p(np.asarray(radj.sum(1)).flatten()))
        names.append(f"logdeg_{rname}")

    an = _row_norm(gd.adj)
    h1_mean = an.dot(gd.x)
    cols.append(h1_mean)
    names += [f"n1mean_{i}" for i in range(gd.x.shape[1])]

    # 1-hop max via per-feature sparse trick would be slow in pure scipy for
    # wide graphs; use mean of squares as a cheap dispersion proxy instead.
    h1_sq = an.dot(gd.x ** 2)
    cols.append(h1_sq)
    names += [f"n1sqmean_{i}" for i in range(gd.x.shape[1])]

    h2_mean = an.dot(h1_mean)
    cols.append(h2_mean)
    names += [f"n2mean_{i}" for i in range(gd.x.shape[1])]

    mat = np.column_stack([c if c.ndim > 1 else c[:, None] for c in cols])
    return mat.astype(np.float32), names


def label_graph_features(gd: GraphData, train_mask: np.ndarray,
                         legacy_leaky: bool = False
                         ) -> tuple[np.ndarray, list[str]]:
    """Train-label neighbor statistics. train_mask: (N,) bool.

    legacy_leaky=True reproduces the pre-Jul-23 bug (backtracking self-walks
    kept in the 2-hop counts) for the paper's M2 figures ONLY. Never use it
    for a real model."""
    lab = train_mask.astype(np.float32)                     # is train-labeled
    fraud = (train_mask & (gd.y == 1)).astype(np.float32)   # is train-labeled fraud

    out, names = [], []
    a = gd.adj
    ones = np.ones(gd.n, dtype=np.float32)
    deg = np.asarray(a.sum(1)).flatten().astype(np.float32)
    sub = 0.0 if legacy_leaky else 1.0
    # 2-hop stats via A.(A.v) walk counts -- O(nnz), never materializes A^2.
    # Backtracking walks i->j->i are subtracted (diag(A^2)=deg): otherwise a
    # train node's OWN label leaks into its own 2-hop feature (Jul-23 bug M2).
    hops = (("1", (a.dot(lab), a.dot(fraud), a.dot(ones))),
            ("2", (a.dot(a.dot(lab)) - sub * deg * lab,
                   a.dot(a.dot(fraud)) - sub * deg * fraud,
                   a.dot(a.dot(ones)) - sub * deg)))
    for hop, (n_lab, n_frd, n_tot) in hops:
        frac_lab = np.divide(n_lab, n_tot, out=np.zeros_like(n_lab),
                             where=n_tot > 0)
        frac_frd = np.divide(n_frd, n_lab, out=np.zeros_like(n_frd),
                             where=n_lab > 0)
        out += [frac_lab, frac_frd]
        names += [f"h{hop}_frac_train_labeled", f"h{hop}_train_fraud_frac"]
    return np.column_stack(out).astype(np.float32), names


def edge_homophily(gd: GraphData) -> float:
    """Fraction of edges (among labeled-labeled pairs) connecting same class.
    Places each real dataset on the synthetic camouflage axis."""
    coo = sp.triu(gd.adj, k=1).tocoo()
    yi, yj = gd.y[coo.row], gd.y[coo.col]
    both = (yi >= 0) & (yj >= 0)
    if both.sum() == 0:
        return float("nan")
    return float((yi[both] == yj[both]).mean())
