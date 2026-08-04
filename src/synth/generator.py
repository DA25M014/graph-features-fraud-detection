"""Camouflage fraud-SBM: the controlled instrument for the regime map.

Three knobs (protocol.md S8):
  feat_strength s >= 0 : class-mean separation in units of feature std.
                         s=0 -> features carry zero class signal.
  camouflage    c in [0,1]: c=0 -> assortative fraud (fraudsters cluster,
                         high homophily); c=1 -> full camouflage (fraud-fraud
                         edges suppressed, fraud wires into normals like a
                         normal node). Sweeping c traverses the homophily
                         axis where real datasets sit (yelp 0.773, amazon 0.954
                         on labeled pairs).
  label_rate    r      : fraction of nodes with observable labels at train.
                         (Applied by the harness split, not here.)

Degree is held approximately constant across c so camouflage changes WIRING,
not exposure -- otherwise degree becomes a confound.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.data.loaders import GraphData


def make_fraud_sbm(n: int = 4000, fraud_rate: float = 0.1, feat_dim: int = 24,
                   feat_strength: float = 1.0, camouflage: float = 0.5,
                   avg_degree: float = 12.0, seed: int = 0) -> GraphData:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(np.int8)
    nf, nn_ = int(y.sum()), int(n - y.sum())

    # Features: class-conditional Gaussians, unit noise.
    mu = rng.standard_normal(feat_dim)
    mu = mu / np.linalg.norm(mu) * feat_strength
    x = rng.standard_normal((n, feat_dim)).astype(np.float32)
    x[y == 1] += mu.astype(np.float32)

    # Block edge probabilities. Base assortative structure at c=0:
    #   p_ff0 = 8q, p_fn0 = q; camouflage interpolates toward
    #   p_ff1 -> 0.1q, p_fn1 chosen to preserve fraud expected degree.
    q = avg_degree / n
    p_nn = q * (1 + 0.2)                       # normals slightly cohesive, fixed
    p_ff = (1 - camouflage) * 8 * q + camouflage * 0.1 * q
    deg_target = avg_degree
    # fraud expected degree = p_ff*nf + p_fn*nn_  -> solve p_fn
    p_fn = np.clip((deg_target - p_ff * nf) / nn_, 0, 1)

    def _block(rows, cols, p):
        # 0.5x compensates for maximum-symmetrization of two directed draws.
        # NB: homophily floor at c=1 is coupled to fraud_rate (few fraud nodes
        # -> n-n edges dominate). The Phase-2 grid sweeps fraud_rate in
        # {0.05, 0.15} to bracket amazon (6.9%) / yelp (14.5%); Day 9-10
        # calibrates knob ranges against the real-data homophily anchors.
        m = rng.random((len(rows), len(cols))) < (0.5 * p)
        r, c = np.nonzero(m)
        return rows[r], cols[c]

    idx_f, idx_n = np.where(y == 1)[0], np.where(y == 0)[0]
    R, C = [], []
    for (ri, ci, p) in ((idx_f, idx_f, p_ff), (idx_f, idx_n, p_fn),
                        (idx_n, idx_n, p_nn)):
        r, c = _block(ri, ci, p)
        R.append(r)
        C.append(c)
    rows = np.concatenate(R)
    cols = np.concatenate(C)
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()

    name = f"synth_s{feat_strength:g}_c{camouflage:g}"
    return GraphData(name=name, x=x, y=y, adj=adj)


if __name__ == "__main__":
    from src.features.graph_features import edge_homophily
    for c in (0.0, 0.5, 1.0):
        gd = make_fraud_sbm(camouflage=c, seed=1)
        deg = gd.adj.sum() / gd.n
        print(f"c={c:.1f}  homophily={edge_homophily(gd):.3f}  "
              f"avg_deg={deg:.1f}  fraud_deg="
              f"{gd.adj[gd.y == 1].sum() / max((gd.y == 1).sum(), 1):.1f}")
