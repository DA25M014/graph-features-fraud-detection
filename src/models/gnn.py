"""Rung (c): GCN + GraphSAGE-mean in pure PyTorch (sparse ops only).

Deliberately dependency-light: no PyG/DGL, so the pilot path can never be
blocked by wheel availability on Python 3.13. GAT and the fraud specialist
(BWGNN) come in Phase 2 -- via PyG if it installs cleanly, else hand-rolled.

Training protocol (symmetric with tabular rungs): masked BCE with pos_weight,
early stopping on val AUPRC, StandardScaler fit on train rows only.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.eval.harness import metrics


def _to_torch_sparse(adj: sp.csr_matrix, gcn_norm: bool = True) -> torch.Tensor:
    a = adj.tocoo().astype(np.float32)
    if gcn_norm:  # D^-1/2 (A+I) D^-1/2
        a = (adj + sp.eye(adj.shape[0], format="csr")).tocoo()
        deg = np.asarray(a.sum(1)).flatten()
        dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
        vals = dinv[a.row] * a.data * dinv[a.col]
    else:         # row-normalized A (SAGE neighbor mean)
        deg = np.asarray(adj.sum(1)).flatten()
        dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        a = adj.tocoo()
        vals = dinv[a.row] * a.data
    idx = torch.from_numpy(np.vstack([a.row, a.col])).long()
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals.astype(np.float32)),
                                   size=a.shape).coalesce()


class GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hid] * layers
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(hid, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        for lin in self.lins:
            x = torch.sparse.mm(adj_norm, self.drop(x))
            x = torch.relu(lin(x))
        return self.out(x).squeeze(-1)


class SAGE(nn.Module):
    """GraphSAGE-mean: concat(self, neighbor-mean) per layer."""

    def __init__(self, in_dim: int, hid: int = 64, layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hid] * layers
        self.lins = nn.ModuleList(nn.Linear(2 * a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(hid, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_rownorm):
        for lin in self.lins:
            x = self.drop(x)
            nb = torch.sparse.mm(adj_rownorm, x)
            x = torch.relu(lin(torch.cat([x, nb], dim=1)))
        return self.out(x).squeeze(-1)


MODELS = {"gcn": (GCN, True), "sage": (SAGE, False)}


def train_gnn(name: str, seed: int, x: np.ndarray, y: np.ndarray,
              adj: sp.csr_matrix, train_idx, val_idx, test_idx,
              epochs: int = 300, patience: int = 30, lr: float = 5e-3,
              hid: int = 64, device: str = "cpu") -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu = x[train_idx].mean(0, keepdims=True)
    sd = x[train_idx].std(0, keepdims=True) + 1e-8
    xs = torch.from_numpy(((x - mu) / sd).astype(np.float32)).to(device)

    cls, gcn_norm = MODELS[name]
    a = _to_torch_sparse(adj, gcn_norm=gcn_norm).to(device)
    model = cls(x.shape[1], hid=hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    yt = torch.from_numpy((y == 1).astype(np.float32)).to(device)
    tr = torch.from_numpy(np.asarray(train_idx)).long().to(device)
    pos_w = torch.tensor([(y[train_idx] == 0).sum() /
                          max((y[train_idx] == 1).sum(), 1)]).float().to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_val, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xs, a)
        loss = lossf(logits[tr], yt[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(xs, a)).cpu().numpy()
        v = metrics(y[val_idx], proba[val_idx])["auprc"]
        if v > best_val:
            best_val, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(xs, a)).cpu().numpy()
    return proba[test_idx]
