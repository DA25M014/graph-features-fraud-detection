"""Dataset loaders -> standardized GraphData container.

Conventions (match CARE-GNN / GADBench where they exist):
- Amazon: nodes 0..3304 are excluded from train/eval splits (standard convention;
  these users have too few reviews to label reliably). eval_mask handles this.
- Elliptic: temporal evaluation. Standard split = train timesteps 1-34,
  test 35-49. Unknown-class nodes excluded from splits but KEPT in the graph
  (they still carry structural signal for neighbor aggregation / GNNs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


@dataclass
class GraphData:
    name: str
    x: np.ndarray                      # (N, F) float32, raw (unscaled) features
    y: np.ndarray                      # (N,) int8: 1 fraud, 0 licit, -1 unknown
    adj: sp.csr_matrix                 # (N, N) homogeneous graph, symmetric, no self-loops
    relations: dict = field(default_factory=dict)   # name -> csr_matrix (multi-relation datasets)
    eval_mask: np.ndarray | None = None             # (N,) bool: nodes usable in train/val/test
    timesteps: np.ndarray | None = None             # (N,) int, Elliptic only

    def __post_init__(self):
        if self.eval_mask is None:
            self.eval_mask = self.y >= 0

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def summary(self) -> str:
        pos = int((self.y == 1).sum())
        lab = int((self.y >= 0).sum())
        return (f"{self.name}: N={self.n} F={self.x.shape[1]} "
                f"edges={self.adj.nnz // 2} labeled={lab} fraud={pos} "
                f"({100 * pos / max(lab, 1):.1f}%) evalable={int(self.eval_mask.sum())}")


def _clean_sym(a: sp.spmatrix) -> sp.csr_matrix:
    """Symmetrize, binarize, drop self-loops."""
    a = sp.csr_matrix(a)
    a = a.maximum(a.T)
    a.data[:] = 1.0
    a.setdiag(0)
    a.eliminate_zeros()
    return a.astype(np.float32)


def _load_mat(fname: str, name: str, rel_keys: list[str],
              exclude_first: int = 0) -> GraphData:
    m = loadmat(os.path.join(RAW, fname))
    x = np.asarray(sp.csc_matrix(m["features"]).todense(), dtype=np.float32)
    y = np.asarray(m["label"]).flatten().astype(np.int8)
    adj = _clean_sym(m["homo"])
    rels = {k: _clean_sym(m[k]) for k in rel_keys}
    eval_mask = np.ones(x.shape[0], dtype=bool)
    if exclude_first > 0:
        eval_mask[:exclude_first] = False
    return GraphData(name=name, x=x, y=y, adj=adj, relations=rels, eval_mask=eval_mask)


def load_yelp() -> GraphData:
    return _load_mat("YelpChi.mat", "yelp", ["net_rur", "net_rtr", "net_rsr"])


def load_amazon() -> GraphData:
    return _load_mat("Amazon.mat", "amazon", ["net_upu", "net_usu", "net_uvu"],
                     exclude_first=3305)


def load_elliptic(root: str | None = None) -> GraphData:
    """Expects the three Kaggle CSVs under data/raw/elliptic/ (Kaggle's nested
    elliptic_bitcoin_dataset/ subfolder is auto-detected).

    txIds are 9-digit ints > 2**24, so they must never pass through float32
    (24-bit mantissa rounds them; the Jul-23 KeyError bug). Parsed via pandas
    with explicit int64. First parse caches to elliptic_cache.npz.
    """
    import pandas as pd

    root = root or os.path.join(RAW, "elliptic")
    nested = os.path.join(root, "elliptic_bitcoin_dataset")
    if (not os.path.exists(os.path.join(root, "elliptic_txs_features.csv"))
            and os.path.exists(nested)):
        root = nested

    cache = os.path.join(root, "elliptic_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        adj = sp.csr_matrix((z["adata"], z["aindices"], z["aindptr"]),
                            shape=(z["x"].shape[0], z["x"].shape[0]))
        return GraphData(name="elliptic", x=z["x"], y=z["y"], adj=adj,
                         timesteps=z["timesteps"])

    feats = pd.read_csv(os.path.join(root, "elliptic_txs_features.csv"),
                        header=None)
    tx_ids = feats[0].to_numpy(np.int64)          # int64 only -- never float32
    timesteps = feats[1].to_numpy(np.int32)
    x = feats.iloc[:, 2:].to_numpy(np.float32)    # features are safe as f32
    id2row = {t: i for i, t in enumerate(tx_ids)}

    cls = pd.read_csv(os.path.join(root, "elliptic_txs_classes.csv"),
                      dtype={"txId": np.int64, "class": str})
    y = np.full(len(tx_ids), -1, dtype=np.int8)
    lab_map = {"1": 1, "2": 0}
    for tid, c in zip(cls["txId"].to_numpy(), cls["class"].to_numpy()):
        row = id2row.get(int(tid))
        v = lab_map.get(str(c).strip().strip('"'))
        if row is not None and v is not None:
            y[row] = v

    edges = pd.read_csv(os.path.join(root, "elliptic_txs_edgelist.csv"),
                        dtype=np.int64)
    r = edges.iloc[:, 0].map(id2row).to_numpy()
    c = edges.iloc[:, 1].map(id2row).to_numpy()
    n = len(tx_ids)
    adj = _clean_sym(sp.csr_matrix(
        (np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n)))

    np.savez(cache, x=x, y=y, timesteps=timesteps,
             adata=adj.data, aindices=adj.indices, aindptr=adj.indptr)
    return GraphData(name="elliptic", x=x, y=y, adj=adj, timesteps=timesteps)


LOADERS = {"yelp": load_yelp, "amazon": load_amazon, "elliptic": load_elliptic}
