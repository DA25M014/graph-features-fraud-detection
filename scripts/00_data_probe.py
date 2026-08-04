"""Day-0 access probe: fetch/verify all datasets, print stats, surface blockers.

Usage:  python scripts/00_data_probe.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CARE_GNN = "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data"

ok, blocked = [], []

os.makedirs(RAW, exist_ok=True)
for zname, mat in (("YelpChi.zip", "YelpChi.mat"), ("Amazon.zip", "Amazon.mat")):
    matp = os.path.join(RAW, mat)
    if not os.path.exists(matp):
        zp = os.path.join(RAW, zname)
        if not os.path.exists(zp):
            print(f"downloading {zname} ...")
            subprocess.run(["curl", "-sL", "-o", zp, f"{CARE_GNN}/{zname}"],
                           check=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW)

from src.data.loaders import load_amazon, load_yelp  # noqa: E402
from src.features.graph_features import edge_homophily  # noqa: E402

for loader in (load_yelp, load_amazon):
    try:
        gd = loader()
        print(f"OK  {gd.summary()}  homophily={edge_homophily(gd):.3f}")
        ok.append(gd.name)
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKED {loader.__name__}: {e}")
        blocked.append(loader.__name__)

ell = os.path.join(RAW, "elliptic")
need = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv"]
missing = [f for f in need if not os.path.exists(os.path.join(ell, f))]
if missing:
    print(f"\nBLOCKED elliptic: missing {missing}")
    print("  -> Kaggle auth required. Fix (5 min):")
    print("     1. kaggle.com -> Account -> Create API Token (kaggle.json)")
    print("     2. pip install kaggle && mkdir -p ~/.kaggle && "
          "mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    print("     3. kaggle datasets download ellipticco/elliptic-data-set "
          f"-p {ell} --unzip")
    blocked.append("elliptic")
else:
    from src.data.loaders import load_elliptic  # noqa: E402
    gd = load_elliptic()
    print(f"OK  {gd.summary()}  homophily={edge_homophily(gd):.3f}")
    ok.append("elliptic")

print(f"\nprobe result: {len(ok)} ok, {len(blocked)} blocked "
      f"{'-> resolve before Aug 1' if blocked else '-> all clear for Day 1'}")
