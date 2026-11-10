"""Phase 5 figures from the frozen output files only (no computation of new numbers).

  reports/figures/phase5/fig1_strategies.png       Dice (image-weighted, mean +- SD over seeds) under C0, C4, C5, C6
                                                   for LANG, IMG, LR-Gate, SDC-Fallback, Fixed Average; three models
  reports/figures/phase5/fig2_mixed_reliability.png expected Dice vs assumed unreliable-text prevalence p (equal
                                                   C4/C5/C6 mixture); LR-Gate crossovers with LANG and IMG marked
Usage:  python src/phase5/make_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT = os.path.join(REPO, "reports", "figures", "phase5")
METHODS = ["LViT-T", "RecLMIS", "DD-CMD"]
SHOW = ["LANG", "IMG", "LR-Gate", "SDC-Fallback", "Fixed Average"]
COL = {"LANG": "#3b6fb6", "IMG": "#5a5a5a", "LR-Gate": "#2f9e6e", "SDC-Fallback": "#c8463d", "Fixed Average": "#8e44c9"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    T = pd.read_csv(os.path.join(REPO, "results", "phase5_test_results.csv"))
    conds = ["C0", "C4", "C5", "C6"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METHODS):
        g = T[(T.method == m) & (T.dataset == "QaTa-COV19")].set_index(["condition", "strategy"])
        for j, st in enumerate(SHOW):
            x = np.arange(len(conds)) + (j - 2) * 0.15
            ax.errorbar(x, [100 * g.dice[(c, st)] for c in conds], yerr=[100 * g.dice_sd[(c, st)] for c in conds],
                        fmt="o", ms=4.5, color=COL[st], capsize=2, lw=1, label=st)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(m, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("Dice (points)", color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("QaTa-COV19 test (post-hoc Phase 5): Dice by strategy, image-weighted, mean ± SD over 3 seeds",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig1_strategies.png"), dpi=160)
    plt.close(fig)

    M = pd.read_csv(os.path.join(REPO, "results", "phase5_mixed_reliability.csv"))
    X = pd.read_csv(os.path.join(REPO, "results", "phase5_crossover_summary.csv"), dtype={"p_star": str})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METHODS):
        for st in SHOW:
            g = M[(M.method == m) & (M.strategy == st)]
            ax.plot(g.p, 100 * g.expected_dice, color=COL[st], lw=1.6, label=st)
        lr = M[(M.method == m) & (M.strategy == "LR-Gate")].iloc[0]
        for b in ("LANG", "IMG"):
            r = X[(X.method == m) & (X.strategy == "LR-Gate") & (X.versus == b) & (X.mixture == "equal C4/C5/C6")]
            if len(r) and r.p_star.iloc[0] != "none":
                p = float(r.p_star.iloc[0])
                y = 100 * ((1 - p) * lr.D_C0 + p * lr.D_wrong)
                ax.plot([p], [y], "o", mfc="white", mec=COL["LR-Gate"], ms=7, mew=1.5)
                ax.annotate(f"vs {b}: p*={p:.2f}", (p, y), textcoords="offset points", xytext=(4, 6), fontsize=7.5, color=INK2)
        ax.set_xlabel("assumed prevalence p of unreliable text", color=INK, fontsize=8.5)
        ax.set_title(m, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("expected Dice (points)", color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Mixed text reliability (benchmark scenario: equal C4/C5/C6 mixture; not a clinical prevalence)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig2_mixed_reliability.png"), dpi=160)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
