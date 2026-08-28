# When Do Graph Features Help Fraud Detection?

Code and results for the LoG 2026 extended-abstract submission: a three-rung
feature ladder over GBDT baselines (raw node features; + static graph
features; + label-propagation features) on YelpChi, Amazon, and Elliptic,
plus two label-leakage mechanisms (self-walk leakage in two-hop walk counts;
temporal unreachability under Elliptic's temporal split).

## Quickstart

    python -m venv .venv && source .venv/bin/activate
    pip install numpy scipy pandas scikit-learn lightgbm xgboost matplotlib
    python scripts/00_data_probe.py          # fetches Yelp/Amazon, checks Elliptic
    python scripts/01_pilot_gbdt.py --seeds 5

Results land in results/results.csv (single source of truth for the paper's
Table 1 and Figures 1-2; regenerate figures with scripts/02_plot_figures.py).
The leaky variant discussed in the paper is behind the `legacy_leaky` flag in
src/features/graph_features.py; it is never used for a real model.

## Rebuttal artifacts (Aug 2026)

- REBUTTAL.md: point-by-point author responses to the four reviews.
- scripts/03_rebuttal_runs.py: the non-nested aL rung (raw + label features
  only), the label-rate sweep r in {0.05, 0.1, 0.2, 0.4}, Elliptic across
  all rungs and 5 seeds, and gain-importance shares for rungs (b) and
  (b) leaky. Reuses the exact pipeline above; the r=0.40 rerun reproduces
  results.csv to 5 decimals.
- results/rebuttal.csv, results/rebuttal_importance.csv: the outputs.
- logs/rebuttal_runs.log: full stdout of the run.
- scripts/04_revision_figures.py: regenerates the figures and tables for the
  revised manuscript from the same results.csv, with the rungs relabeled
  a/b/c (paper) = a/b0/b (code), no error bars on the deterministic Elliptic
  panels, and Elliptic table cells printed without a vacuous +-0.000.
