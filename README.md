# fraud-gvt — When Does the Graph Help?

GNNs vs tabular baselines for fraud detection under controlled regimes.
Read docs/protocol.md first - it is the constitution.

## Quickstart (pilot path, no torch needed)

    python -m venv .venv && source .venv/bin/activate
    pip install numpy scipy pandas scikit-learn lightgbm xgboost
    python scripts/00_data_probe.py          # fetches Yelp/Amazon, checks Elliptic
    python scripts/01_pilot_gbdt.py --seeds 5

Results land in results/results.csv (single source of truth).
GNN rung: pip install torch, then src/models/gnn.py::train_gnn.
Status of every module: protocol.md S12.
