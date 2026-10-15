# When Do Graph Features Help Fraud Detection?

Code and results for "When Do Graph Features Help Fraud Detection? A Ladder
Decomposition and Two Label-Leakage Mechanisms" (J. K. Shaikh), an extended
abstract presented at the Fifth Learning on Graphs Conference (LoG 2026). It
contains a three-rung feature ladder over GBDT baselines (raw node features;
+ static graph features; + label-propagation features) on YelpChi, Amazon,
and Elliptic, and two label-leakage mechanisms (self-walk leakage in two-hop
walk counts, M1; temporal unreachability under Elliptic's temporal split, M2).

Rung names: paper (a)/(b)/(c)/(c) leaky = code a/b0/b/b_leaky.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install numpy scipy pandas scikit-learn lightgbm xgboost matplotlib
    python scripts/00_data_probe.py          # fetches Yelp/Amazon, checks Elliptic

## Where each number in the paper comes from

| Paper item | Script | Output |
|---|---|---|
| Table 1, Fig. 1, Fig. 2, ladder numbers at r = 0.40 and on Elliptic | `01_pilot_gbdt.py` | `results/results.csv` |
| Label-rate sweep, aL control, gain-importance shares (Sec. 3, 4.1) | `03_rebuttal_runs.py` | `results/rebuttal.csv`, `results/rebuttal_importance.csv`, `logs/rebuttal_runs.log` |
| Elliptic paired bootstrap intervals (Sec. 3, 4.2) | `05_bootstrap_elliptic.py` | `results/bootstrap_elliptic.csv`, `results/elliptic_test_predictions.npz`, `logs/bootstrap_elliptic.log` |
| Table 2, Sec. 4.1 train/test label-feature means, Sec. 2 feature counts, Elliptic components and cross-timestep edges | `06_labfeat_diagnostics.py` | `results/labfeat_diagnostics.csv`, `results/data_diagnostics.csv`, `logs/labfeat_diagnostics.log` |
| Figure and table files | `04_revision_figures.py` | `fig1_ladder.pdf`, `fig2_mechanisms.pdf`, `table1_ladder.tex`, `table2_labfeat.tex` |

`03_rebuttal_runs.py` never touches `results.csv`; its r = 0.40 rerun
reproduces the YelpChi/Amazon cells of `results.csv` to 5 decimals.
`02_plot_figures.py` is the superseded submission-era figure script; use
`04_revision_figures.py`.

## Reproducing

`01_pilot_gbdt.py` appends to `results/results.csv`, so move the shipped file
aside first:

    mv results/results.csv results/results_shipped.csv
    python scripts/01_pilot_gbdt.py --rungs a b0 b b_leaky --seeds 5
    python scripts/01_pilot_gbdt.py --datasets elliptic --rungs a b0 b b_leaky --seeds 2
    python scripts/03_rebuttal_runs.py
    python scripts/05_bootstrap_elliptic.py | tee logs/bootstrap_elliptic.log
    python scripts/06_labfeat_diagnostics.py | tee logs/labfeat_diagnostics.log
    python scripts/04_revision_figures.py --out paper_figs

Elliptic uses its fixed temporal split (fit on timesteps 1-29, 30-34 held out
and unused, test 35-49), and the near-default GBDTs are deterministic there,
so Elliptic rows carry no seed variation; their `train_rate` column is
meaningless. `05_bootstrap_elliptic.py` refits the Elliptic models and stops
unless every refit AUPRC equals the value in `results.csv`/`rebuttal.csv`.

The leaky variant discussed in the paper is behind the `legacy_leaky` flag in
`src/features/graph_features.py`; it is never used for a real model.
`src/models/gnn.py` and `src/synth/` are unused by the paper.

## Review-period artifacts

`REBUTTAL.md` holds the point-by-point author responses posted during the
review period (Aug 2026); the note at its top lists what changed since.
