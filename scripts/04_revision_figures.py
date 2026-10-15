"""Camera-ready/rebuttal-revision figures + tables from results/results.csv.

Differences vs scripts/02_plot_figures.py, per the posted rebuttal promises:
- Rungs relabeled for display: a -> (a), b0 -> (b), b -> (c),
  b_leaky -> (c) leaky. CSV keys are unchanged (mapping note: paper (a/b/c)
  corresponds to code rungs a/b0/b).
- Fig. 2 Elliptic panel: no error bars (the temporal split is fixed and the
  fits deterministic; the old zero-length bars rendered as misleading ticks).
  Error bars are suppressed for any dataset whose stds are all zero.
- Table 1: Elliptic cells show the mean only (no vacuous +-0.000).
- Table 2 is read from results/labfeat_diagnostics.csv, written by
  scripts/06_labfeat_diagnostics.py.

Usage: .venv/bin/python scripts/04_revision_figures.py --out paper_rev
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(__file__)
DATASETS = ["yelp", "amazon", "elliptic"]
DNAMES = {"yelp": "YelpChi", "amazon": "Amazon", "elliptic": "Elliptic"}
RNAMES = {"a": "(a)", "b0": "(b)", "b": "(c)", "b_leaky": "(c) leaky"}
# camera-ready font sizes (figure points); the PDFs are placed at 0.705x
# (Fig. 1) and 0.816x (Fig. 2) of these sizes, so printed text is >= 7 pt
FS1 = {"bar": 10, "xtick": 10.5, "ytick": 10, "title": 11.5, "ylabel": 11}
FS2 = {"bar": 9, "xtick": 9, "ytick": 10, "title": 10.5, "ylabel": 10}
COLORS = {"a": "#0072B2", "b0": "#009E73", "b": "#E69F00",
          "b_leaky": "#D55E00"}



def agg(df: pd.DataFrame, model: str) -> pd.DataFrame:
    d = df[df["model"] == model]
    g = (d.groupby(["dataset", "rung"])["auprc"]
         .agg(["mean", "std", "count"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    return g


def _bars(ax, g: pd.DataFrame, dataset: str, rungs: list[str], fs: dict):
    sub = g[g["dataset"] == dataset].set_index("rung")
    xs, hs, es, cs, ls = [], [], [], [], []
    for r in (r for r in rungs if r in sub.index):
        xs.append(len(xs))
        hs.append(sub.loc[r, "mean"])
        es.append(sub.loc[r, "std"])
        cs.append(COLORS[r])
        ls.append(RNAMES[r])
    deterministic = max(es) == 0.0
    ax.bar(xs, hs, yerr=None if deterministic else es, color=cs,
           capsize=0 if deterministic else 3, width=0.7)
    # annotate above the error-bar cap, not the bar top, so the cap never
    # strikes the numerals
    for x, h, e in zip(xs, hs, es):
        ax.text(x, h + (0.0 if deterministic else e) + 0.004, f"{h:.3f}",
                ha="center", va="bottom", fontsize=fs["bar"])
    ax.set_xticks(xs)
    ax.set_xticklabels(ls, fontsize=fs["xtick"])
    ax.tick_params(axis="y", labelsize=fs["ytick"])
    lo = min(hs) - 5 * max(max(es), 0.004)
    hi = max(hs) + max(es) + 6 * max(max(es), 0.004)
    ax.set_ylim(max(0, lo), min(1.0, hi))
    ax.spines[["top", "right"]].set_visible(False)


def fig1(g: pd.DataFrame, out: str):
    ds = [d for d in DATASETS if d in set(g["dataset"])]
    fig, axes = plt.subplots(1, len(ds), figsize=(2.6 * len(ds), 2.5))
    if len(ds) == 1:
        axes = [axes]
    for ax, d in zip(axes, ds):
        _bars(ax, g, d, ["a", "b0", "b"], FS1)
        ax.set_title(DNAMES[d], fontsize=FS1["title"])
    axes[0].set_ylabel("AUPRC", fontsize=FS1["ylabel"])
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_ladder.pdf")
    fig.savefig(f"{out}/fig1_ladder.png", dpi=180)
    plt.close(fig)


def fig2(g: pd.DataFrame, out: str):
    ds = [d for d in ["yelp", "elliptic"] if d in set(g["dataset"])]
    fig, axes = plt.subplots(1, len(ds), figsize=(3.1 * len(ds), 2.6))
    if len(ds) == 1:
        axes = [axes]
    for ax, d in zip(axes, ds):
        _bars(ax, g, d, ["a", "b0", "b", "b_leaky"], FS2)
        sub = g[g["dataset"] == d].set_index("rung")
        if {"b", "b_leaky"} <= set(sub.index):
            dlt = sub.loc["b_leaky", "mean"] - sub.loc["b", "mean"]
            ax.set_title(f"{DNAMES[d]}  (leak: {dlt:+.3f})",
                         fontsize=FS2["title"])
        else:
            ax.set_title(DNAMES[d], fontsize=FS2["title"])
    axes[0].set_ylabel("AUPRC", fontsize=FS2["ylabel"])
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_mechanisms.pdf")
    fig.savefig(f"{out}/fig2_mechanisms.png", dpi=180)
    plt.close(fig)


def table1(df: pd.DataFrame, out: str):
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Dataset & Model & (a) & (b) & (c) & (c) leaky \\\\",
             "\\midrule"]
    for d in DATASETS:
        for m in ["lightgbm", "xgboost"]:
            g = agg(df, m)
            sub = g[g["dataset"] == d].set_index("rung")
            if sub.empty:
                continue
            cells = []
            for r in ["a", "b0", "b", "b_leaky"]:
                if r not in sub.index:
                    cells.append("--")
                elif sub.loc[r, "std"] == 0.0:
                    cells.append(f"{sub.loc[r,'mean']:.3f}")
                else:
                    cells.append(f"{sub.loc[r,'mean']:.3f}"
                                 f"$\\pm${sub.loc[r,'std']:.3f}")
            mm = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}[m]
            lines.append(f"{DNAMES[d]} & {mm} & " + " & ".join(cells)
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{out}/table1_ladder.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def table2(labfeat: str, out: str):
    """Elliptic reachability table from results/labfeat_diagnostics.csv
    (written by scripts/06_labfeat_diagnostics.py; clean label features,
    temporal split)."""
    lf = pd.read_csv(labfeat)
    lf = lf[(lf["dataset"] == "elliptic") & (lf["variant"] == "clean")]
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Label feature & Train mean & Test mean \\\\", "\\midrule"]
    for _, r in lf.iterrows():
        nm = r["feature"].replace("_", "\\_")
        lines.append(f"\\texttt{{{nm}}} & {r['train_mean']:.4f} & "
                     f"{r['test_mean']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{out}/table2_labfeat.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default=os.path.join(HERE, "..", "results", "results.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "paper_rev"))
    ap.add_argument("--labfeat", default=os.path.join(
        HERE, "..", "results", "labfeat_diagnostics.csv"))
    ap.add_argument("--model", default="lightgbm")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.results)
    g = agg(df, args.model)
    print(g.to_string(index=False))
    fig1(g, args.out)
    fig2(g, args.out)
    table1(df, args.out)
    table2(args.labfeat, args.out)
    print(f"\nwrote fig1/fig2 (pdf+png) + table1/table2 (.tex) -> {args.out}")


if __name__ == "__main__":
    main()
