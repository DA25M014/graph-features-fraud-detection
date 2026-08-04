"""Generate LoG-2 abstract figures + tables from results/results.csv.

Usage:
    python scripts/02_plot_figures.py            # LightGBM, default paths
    python scripts/02_plot_figures.py --model xgboost

Outputs to figures/: fig1_ladder.pdf/.png, fig2_mechanisms.pdf/.png,
table1_ladder.tex, table2_labfeat.tex. Robust to missing (dataset, rung)
combos: plots whatever the CSV contains.
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
RUNGS = ["a", "b0", "b", "b_leaky"]
RNAMES = {"a": "(a)", "b0": "(b0)", "b": "(b)", "b_leaky": "(b) leaky"}
COLORS = {"a": "#0072B2", "b0": "#009E73", "b": "#E69F00",
          "b_leaky": "#D55E00"}

# Elliptic label-feature reachability diagnostic (canonical Jul-24 run stdout;
# reproduce any time via scripts/01_pilot_gbdt.py --datasets elliptic).
LABFEAT_ELLIPTIC = [
    ("h1\\_frac\\_train\\_labeled", 0.6307, 0.0000),
    ("h1\\_train\\_fraud\\_frac", 0.0449, 0.0000),
    ("h2\\_frac\\_train\\_labeled", 0.5572, 0.0000),
    ("h2\\_train\\_fraud\\_frac", 0.0811, 0.0000),
]


def agg(df: pd.DataFrame, model: str) -> pd.DataFrame:
    d = df[df["model"] == model]
    g = (d.groupby(["dataset", "rung"])["auprc"]
         .agg(["mean", "std", "count"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    return g


def _bars(ax, g: pd.DataFrame, dataset: str, rungs: list[str]):
    sub = g[g["dataset"] == dataset].set_index("rung")
    xs, hs, es, cs, ls = [], [], [], [], []
    for i, r in enumerate(r for r in rungs if r in sub.index):
        xs.append(len(xs))
        hs.append(sub.loc[r, "mean"])
        es.append(sub.loc[r, "std"])
        cs.append(COLORS[r])
        ls.append(RNAMES[r])
    bars = ax.bar(xs, hs, yerr=es, color=cs, capsize=3, width=0.7)
    for x, h in zip(xs, hs):
        ax.text(x, h + 0.004, f"{h:.3f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(ls, fontsize=9)
    lo = min(hs) - 5 * max(max(es), 0.004)
    hi = max(hs) + 6 * max(max(es), 0.004)
    ax.set_ylim(max(0, lo), min(1.0, hi))
    ax.spines[["top", "right"]].set_visible(False)
    return bars


def fig1(g: pd.DataFrame, out: str):
    ds = [d for d in DATASETS if d in set(g["dataset"])]
    fig, axes = plt.subplots(1, len(ds), figsize=(2.6 * len(ds), 2.5))
    if len(ds) == 1:
        axes = [axes]
    for ax, d in zip(axes, ds):
        _bars(ax, g, d, ["a", "b0", "b"])
        ax.set_title(DNAMES[d], fontsize=10)
    axes[0].set_ylabel("AUPRC", fontsize=10)
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
        _bars(ax, g, d, ["a", "b0", "b", "b_leaky"])
        sub = g[g["dataset"] == d].set_index("rung")
        if {"b", "b_leaky"} <= set(sub.index):
            dlt = sub.loc["b_leaky", "mean"] - sub.loc["b", "mean"]
            ax.set_title(f"{DNAMES[d]}  (leak: {dlt:+.3f})", fontsize=10)
        else:
            ax.set_title(DNAMES[d], fontsize=10)
    axes[0].set_ylabel("AUPRC", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_mechanisms.pdf")
    fig.savefig(f"{out}/fig2_mechanisms.png", dpi=180)
    plt.close(fig)


def table1(df: pd.DataFrame, out: str):
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Dataset & Model & (a) & (b0) & (b) & (b) leaky \\\\", "\\midrule"]
    for d in DATASETS:
        for m in ["lightgbm", "xgboost"]:
            g = agg(df, m)
            sub = g[g["dataset"] == d].set_index("rung")
            if sub.empty:
                continue
            cells = []
            for r in ["a", "b0", "b", "b_leaky"]:
                if r in sub.index:
                    cells.append(f"{sub.loc[r,'mean']:.3f}"
                                 f"$\\pm${sub.loc[r,'std']:.3f}")
                else:
                    cells.append("--")
            mm = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}[m]
            lines.append(f"{DNAMES[d]} & {mm} & " + " & ".join(cells)
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{out}/table1_ladder.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def table2(out: str):
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Label feature & train mean & test mean \\\\", "\\midrule"]
    for nm, tr, te in LABFEAT_ELLIPTIC:
        lines.append(f"\\texttt{{{nm}}} & {tr:.4f} & {te:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{out}/table2_labfeat.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default=os.path.join(HERE, "..", "results", "results.csv"))
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.results)
    g = agg(df, args.model)
    print(g.to_string(index=False))
    fig1(g, args.out)
    fig2(g, args.out)
    table1(df, args.out)
    table2(args.out)
    print(f"\nwrote fig1/fig2 (pdf+png) + table1/table2 (.tex) -> {args.out}")


if __name__ == "__main__":
    main()
