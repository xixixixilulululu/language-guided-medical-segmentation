"""Phase 0, Tasks 0.5-0.7: summary table, pairwise comparisons, figures and the PHASE0_RESULTS.md tables.

THIS FILE IS THE FIXED ANALYSIS PLAN. It was committed and pushed before any new test-set result was
summarised. After all 18 runs finish it is run once, unchanged. Any later bug fix is a separate commit,
and PHASE0_RESULTS.md must then record what changed, why, and the results before and after.

Input: for every method x dataset x seed, the per-case metrics written by the common evaluator
(src/evaluate.py / src/lgms/runner.py / src/nnunet_select.py):
    <runs-root>/<method>/<dataset>/seed<k>/eval_<split>/per_case.csv   (sample_id, dice, miou, hd95, assd)
    <runs-root>/<method>/<dataset>/seed<k>/eval_<split>/masks/<id>.png (binarised predictions, case figure)
Every run must contain exactly the sample ids of the split (checked).

Analysis methods (fixed on 2026-10-12; plan: reports/PHASE0_PLAN.md, Tasks 0.5-0.7):

* Task 0.6 summary table, per dataset and method (results/phase0_main_results.csv):
  - For each seed, the test-set mean over cases of Dice, mIoU, HD95 and ASSD (one value per seed).
  - "Mean ± SD": mean and SAMPLE standard deviation (ddof = 1) of these 3 per-seed values. This is
    reported for all four metrics.
  - "Dice 95% CI": per-case bootstrap.
    1. Each test case's Dice is first averaged over the 3 seeds.
    2. The cases are resampled with replacement B = 10,000 times, and the mean is taken each time.
    3. The CI is the 2.5% / 97.5% percentile interval of these means.
    4. The random generator is numpy default_rng(0), created anew for every dataset x method.
  - The point estimate (mean of the seed-averaged per-case Dice) equals the mean of the 3 per-seed means.
* Task 0.7 pairwise comparisons, within each dataset (results/phase0_pairwise_comparisons.csv).
  Pairs: U-Net vs TransUNet, U-Net vs nnU-Net, TransUNet vs nnU-Net. Steps:
    1. Average each test case's Dice over the 3 seeds within each method.
    2. Align the same test cases.
    3. Compute the paired differences d = Dice_A - Dice_B.
    4. ΔDice = mean(d).
    5. 95% CI = mean(d) ± t(0.975, n-1) × SD(d) / sqrt(n).
    6. Primary p-value: two-sided paired t-test (scipy.stats.ttest_rel).
    7. Holm correction across the 3 comparisons of the dataset.
  - Supplementary sensitivity analysis, specified in advance: two-sided Wilcoxon signed-rank test
    (scipy.stats.wilcoxon, default zero handling). It gets its own Holm correction across the same 3
    comparisons, is reported in separate columns, and is not used for the main conclusions.
  - Significance is never tested on the three seed means alone.
* Limitation, stated in the report: the case bootstrap and the paired tests treat every image as an
  independent sample. Neighbouring MosMedData+ slices of one CT volume are correlated, so the MosMed CIs
  may be too narrow and the p-values too small.
* Figures (reports/figures/):
  - phase0_dice_ci_<dataset>.png: per method, the mean Dice with its 95% bootstrap CI (required figures
    1 and 2).
  - phase0_seed_stability.png: the 3 per-seed test Dice values of every method and dataset, with mean ±
    SD (required figure 3).
  - phase0_cases_<dataset>.png: real cases, selected by rule.
    - Selection score of a case = its Dice averaged over the 3 seeds and then over the 3 methods.
    - Cases are sorted by this score (ascending; ties broken by sample id).
    - The cases at the 10/30/50/70/90% quantiles are shown (position round(q × (n - 1))).
    - Each row shows: image, ground truth (yellow dashed contour), and the seed-1001 prediction of every
      method. Every panel is labelled with that method's seed-1001 Dice and HD95 for the case.

Usage (after all 18 runs):
    python src/analyze_phase0.py
Testing (never on the test split): see src/test_analyze_phase0.py (synthetic runs on the validation split).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data  # noqa: E402

METHODS = [("unet", "U-Net"), ("transunet", "TransUNet"), ("nnunet", "nnU-Net")]
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]
PAIRS = [("unet", "transunet"), ("unet", "nnunet"), ("transunet", "nnunet")]
METRICS = ["dice", "miou", "hd95", "assd"]
N_BOOT = 10_000          # bootstrap resamples for the Dice 95% CI
BOOT_SEED = 0            # numpy default_rng seed, re-created for every dataset x method
CASE_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
FIGURE_SEED = 1001       # the seed whose predictions are drawn in the case figures
# Categorical colours (reference palette, slots 1-3, fixed order: U-Net, TransUNet, nnU-Net) + markers,
# so method identity never depends on colour alone.
COLORS = {"unet": "#2a78d6", "transunet": "#eb6834", "nnunet": "#1baf7a"}
MARKERS = {"unet": "o", "transunet": "s", "nnunet": "D"}
SEED_MARKERS = {1001: "o", 1002: "s", 1003: "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
GT_COLOR = "#f2c200"     # ground truth: yellow dashed contour (CLAUDE.md figure rule)
# LViT paper (Li et al., IEEE TMI 2023), Table II, test set, Dice / mIoU in % -- for the comparison table only
LVIT_TABLE2 = {("unet", "QaTa-COV19"): (79.02, 69.46), ("unet", "MosMedData+"): (64.60, 50.73),
               ("transunet", "QaTa-COV19"): (78.63, 69.13), ("transunet", "MosMedData+"): (71.24, 58.44),
               ("nnunet", "QaTa-COV19"): (80.42, 70.81), ("nnunet", "MosMedData+"): (72.59, 60.36)}
NAME = dict(METHODS)


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def run_dir(root, method, dataset, seed):
    return os.path.join(root, method, dataset, f"seed{seed}")


def load_per_case(root, split):
    """{(method, dataset, seed): DataFrame indexed by sample_id}; every run must hold exactly the split ids."""
    out = {}
    for ds in DATASETS:
        ids = data.split_ids(ds, split)
        for m, _ in METHODS:
            for s in SEEDS:
                f = os.path.join(run_dir(root, m, ds, s), f"eval_{split}", "per_case.csv")
                if not os.path.exists(f):
                    raise FileNotFoundError(f"missing run output: {f}")
                df = pd.read_csv(f, dtype={"sample_id": str}).set_index("sample_id")
                if sorted(df.index) != sorted(ids) or df.index.duplicated().any():
                    raise ValueError(f"{f}: sample ids do not match the {split} split of {ds}")
                if df[METRICS].isna().any().any():
                    raise ValueError(f"{f}: missing metric values")
                out[(m, ds, s)] = df.loc[ids, METRICS]
    return out


def seed_averaged(per_case, method, dataset, metric="dice"):
    """Per-case metric averaged over the 3 seeds (Series indexed by sample_id, split order)."""
    return pd.concat([per_case[(method, dataset, s)][metric] for s in SEEDS], axis=1).mean(axis=1)


# ------------------------------------------------------------------------------------------------
# Statistics (each function is tested in src/test_analyze_phase0.py)
# ------------------------------------------------------------------------------------------------
def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile CI of the mean: resample the cases with replacement n_boot times."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):  # blocks of 1000 resamples keep memory small for n = 2113
        k = min(1000, n_boot - start)
        means[start:start + k] = x[rng.integers(0, len(x), size=(k, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_t(a, b, level=0.95):
    """Mean paired difference a - b, its t-distribution CI and the two-sided paired t-test p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * se
    res = stats.ttest_rel(a, b)
    return {"delta": float(mean), "ci_low": float(mean - half), "ci_high": float(mean + half),
            "t": float(res.statistic), "p": float(res.pvalue), "n": n}


def wilcoxon_p(a, b):
    """Supplementary: two-sided Wilcoxon signed-rank test on the paired differences (scipy defaults)."""
    res = stats.wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------------------------------------
# Task 0.6 / 0.7 tables
# ------------------------------------------------------------------------------------------------
def main_table(per_case):
    """One row per dataset x method. Dice / mIoU in %, HD95 / ASSD in pixels of the 224x224 image."""
    rows = []
    for ds in DATASETS:
        for m, name in METHODS:
            seed_means = {met: [per_case[(m, ds, s)][met].mean() for s in SEEDS] for met in METRICS}
            row = {"dataset": ds, "method": name, "n_seeds": len(SEEDS), "n_cases": len(per_case[(m, ds, SEEDS[0])])}
            for met in METRICS:
                scale = 100.0 if met in ("dice", "miou") else 1.0
                v = np.array(seed_means[met]) * scale
                unit = "pct" if scale == 100.0 else "px"
                row[f"{met}_mean_{unit}"] = v.mean()
                row[f"{met}_sd_{unit}"] = v.std(ddof=1)
                if met == "dice":
                    lo, hi = bootstrap_mean_ci(seed_averaged(per_case, m, ds) * 100.0)
                    row["dice_ci95_low_pct"], row["dice_ci95_high_pct"] = lo, hi
            for s, v in zip(SEEDS, seed_means["dice"]):
                row[f"dice_seed{s}_pct"] = v * 100.0
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(per_case):
    """Task 0.7: seed-averaged per-case Dice, paired differences, t CI / p (primary) + Wilcoxon (supplementary)."""
    rows = []
    for ds in DATASETS:
        block = []
        for a, b in PAIRS:
            da, db = seed_averaged(per_case, a, ds) * 100.0, seed_averaged(per_case, b, ds) * 100.0
            assert (da.index == db.index).all()          # aligned test cases
            r = paired_t(da.values, db.values)
            w_stat, w_p = wilcoxon_p(da.values, db.values)
            block.append({"dataset": ds, "method_a": NAME[a], "method_b": NAME[b], "n_cases": r["n"],
                          "delta_dice_pct": r["delta"], "ci95_low_pct": r["ci_low"], "ci95_high_pct": r["ci_high"],
                          "t_statistic": r["t"], "p_raw": r["p"],
                          "wilcoxon_statistic": w_stat, "wilcoxon_p_raw": w_p})
        for col, src in [("p_holm", "p_raw"), ("wilcoxon_p_holm", "wilcoxon_p_raw")]:
            adj = holm([r[src] for r in block])
            for r, v in zip(block, adj):
                r[col] = float(v)
        rows += block
    cols = ["dataset", "method_a", "method_b", "n_cases", "delta_dice_pct", "ci95_low_pct", "ci95_high_pct",
            "t_statistic", "p_raw", "p_holm", "wilcoxon_statistic", "wilcoxon_p_raw", "wilcoxon_p_holm"]
    return pd.DataFrame(rows)[cols]


def select_cases(per_case, dataset):
    """Rule-based case selection: quantiles of the Dice averaged over 3 seeds and 3 methods."""
    score = pd.concat([seed_averaged(per_case, m, dataset) for m, _ in METHODS], axis=1).mean(axis=1)
    ordered = score.sort_index().sort_values(kind="stable")           # ascending, ties by sample id
    n = len(ordered)
    picks = []
    for q in CASE_QUANTILES:
        i = int(round(q * (n - 1)))
        picks.append({"quantile": q, "sample_id": ordered.index[i], "score": float(ordered.iloc[i])})
    return picks


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_dice_ci(table, dataset, path):
    """Required figures 1/2: mean Dice and 95% bootstrap CI per method (one dataset)."""
    import matplotlib.pyplot as plt
    sub = table[table.dataset == dataset].set_index("method")
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    for y, (m, name) in enumerate(METHODS):
        r = sub.loc[name]
        ax.plot([r.dice_ci95_low_pct, r.dice_ci95_high_pct], [y, y], color=COLORS[m], linewidth=2,
                solid_capstyle="round")
        ax.plot(r.dice_mean_pct, y, MARKERS[m], color=COLORS[m], markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        # value labels in a column right of the axes, so the x-range covers only the data (<= 100 %)
        ax.text(1.02, y, f"{r.dice_mean_pct:.2f}  [{r.dice_ci95_low_pct:.2f}, {r.dice_ci95_high_pct:.2f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(METHODS)), [n for _, n in METHODS], color=INK, fontsize=10)
    ax.invert_yaxis()
    lo, hi = sub.dice_ci95_low_pct.min(), sub.dice_ci95_high_pct.max()
    pad = max(0.5, (hi - lo) * 0.15)
    ax.set_xlim(max(0.0, lo - pad), min(100.0, hi + pad))
    ax.set_xlabel("Test Dice (%), mean of 3 seeds, 95% CI (case bootstrap, 10,000 resamples)", color=INK2, fontsize=8.5)
    ax.set_title(f"{dataset}: Dice with 95% CI", loc="left", color=INK, fontsize=11)
    _style(ax)
    fig.tight_layout(rect=(0, 0, 0.78, 1))     # keep room for the value column
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_seed_stability(table, path):
    """Required figure 3: the 3 per-seed test Dice values per method, with mean ± SD, both datasets."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
    for ax, ds in zip(axes, DATASETS):
        sub = table[table.dataset == ds].set_index("method")
        for x, (m, name) in enumerate(METHODS):
            r = sub.loc[name]
            ax.errorbar(x, r.dice_mean_pct, yerr=r.dice_sd_pct, fmt="none", ecolor=INK2, elinewidth=1.2,
                        capsize=6, zorder=1)
            ax.plot([x - 0.22, x + 0.22], [r.dice_mean_pct] * 2, color=INK, linewidth=1.5, zorder=2)
            for k, s in enumerate(SEEDS):
                ax.plot(x + (k - 1) * 0.12, r[f"dice_seed{s}_pct"], SEED_MARKERS[s], color=COLORS[m],
                        markersize=8, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_xticks(range(len(METHODS)), [n for _, n in METHODS], color=INK, fontsize=10)
        ax.set_ylabel("Test Dice (%)", color=INK2, fontsize=9)
        ax.set_title(ds, loc="left", color=INK, fontsize=11)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    handles = [plt.Line2D([], [], marker=SEED_MARKERS[s], linestyle="none", color=INK2, markersize=7,
                          label=f"seed {s}") for s in SEEDS]
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.5, label="mean ± SD (3 seeds)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Three-seed stability of the test Dice", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_cases(per_case, root, split, dataset, picks, path):
    """Real cases (rule-based): image | ground truth | seed-1001 prediction of each method, Dice / HD95 labels."""
    import matplotlib.pyplot as plt
    cols = ["Image", "Ground truth"] + [n for _, n in METHODS]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.1 * len(cols), 2.25 * len(picks)), dpi=150)
    for r, pk in enumerate(picks):
        sid = pk["sample_id"]
        img = data.read_image(dataset, split, sid)
        gt = data.read_mask(dataset, split, sid) > 0
        for c, title in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if c >= 1:
                ax.contour(gt.astype(float), levels=[0.5], colors=[GT_COLOR], linestyles="--", linewidths=1.3)
            if c >= 2:
                m = METHODS[c - 2][0]
                pred = cv2.imread(os.path.join(run_dir(root, m, dataset, FIGURE_SEED), f"eval_{split}", "masks",
                                               f"{sid}.png"), cv2.IMREAD_GRAYSCALE) > 0
                overlay = np.zeros((*pred.shape, 4))
                overlay[pred] = (*[int(COLORS[m][i:i + 2], 16) / 255 for i in (1, 3, 5)], 0.40)
                ax.imshow(overlay)
                if pred.any():
                    ax.contour(pred.astype(float), levels=[0.5], colors=[COLORS[m]], linewidths=1.0)
                v = per_case[(m, dataset, FIGURE_SEED)].loc[sid]
                ax.text(3, 220, f"Dice {v.dice * 100:.1f}\nHD95 {v.hd95:.1f}", color="white", fontsize=7.5,
                        va="bottom", bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0))
            if r == 0:
                ax.set_title(title, fontsize=9.5, color=INK)
            if c == 0:
                wrapped = "\n".join(sid[k:k + 24] for k in range(0, len(sid), 24))   # full id, wrapped
                ax.set_ylabel(f"q{int(pk['quantile'] * 100)}\n{wrapped}", fontsize=6.5, color=INK2)
    fig.suptitle(f"{dataset}: cases at the 10/30/50/70/90% quantiles of the mean Dice (3 seeds x 3 methods); "
                 f"predictions and labels: seed {FIGURE_SEED}", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# PHASE0_RESULTS.md (tables generated; conclusions are placeholders written by hand later)
# ------------------------------------------------------------------------------------------------
def _p(p):
    return "<1e-300" if p == 0 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def git_commit():
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "src/analyze_phase0.py"],
                                        text=True).strip()
        return h + (" (src/analyze_phase0.py has uncommitted changes!)" if dirty else "")
    except Exception:  # noqa: BLE001 - report generation must not fail on git problems
        return "unknown"


def write_report(table, pairs, picks, fig_paths, path, split):
    """Generate reports/PHASE0_RESULTS.md: fixed method text, the result tables, figure links, placeholders."""
    rel = lambda p: os.path.relpath(p, os.path.dirname(path))  # noqa: E731
    L = []
    L += ["# Phase 0 — Results (Tasks 0.5–0.7)", "",
          f"Generated by `src/analyze_phase0.py` (analysis commit `{git_commit()}`) on "
          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}, split: `{split}`. The tables and figures are "
          "generated; text marked *[placeholder]* is written by hand after reviewing the results.", ""]
    with open(os.path.join(REPO, "reports", "templates", "PHASE0_RESULTS_methods.md")) as f:
        L += [f.read().rstrip(), ""]

    L += ["## 1. Main results (Task 0.6)", "",
          f"{split.capitalize()} set, 224×224 common evaluation. Each value is the mean ± sample SD over the 3 "
          f"seeds of the per-seed {split}-set mean. Dice and mIoU are in %; HD95 and ASSD are in pixels. The 95% CI is the "
          "case bootstrap of the seed-averaged per-case Dice (see Methods).", "",
          "| Dataset | Method | Dice Mean ± SD | Dice 95% CI | mIoU | HD95 | ASSD |",
          "|---|---|---|---|---:|---:|---:|"]
    for _, r in table.iterrows():
        L.append(f"| {r.dataset} | {r.method} | {r.dice_mean_pct:.2f} ± {r.dice_sd_pct:.2f} | "
                 f"[{r.dice_ci95_low_pct:.2f}, {r.dice_ci95_high_pct:.2f}] | {r.miou_mean_pct:.2f} ± {r.miou_sd_pct:.2f} | "
                 f"{r.hd95_mean_px:.2f} ± {r.hd95_sd_px:.2f} | {r.assd_mean_px:.2f} ± {r.assd_sd_px:.2f} |")
    L += ["", "Per-seed test Dice (%):", "", "| Dataset | Method | " + " | ".join(f"seed {s}" for s in SEEDS) + " |",
          "|---|---|" + "---:|" * len(SEEDS)]
    for _, r in table.iterrows():
        L.append(f"| {r.dataset} | {r.method} | " + " | ".join(f"{r[f'dice_seed{s}_pct']:.2f}" for s in SEEDS) + " |")
    L += ["", "*[placeholder: interpretation of the main results]*", ""]

    L += ["## 2. Pairwise comparisons (Task 0.7)", "",
          "Primary analysis: paired t-test on the seed-averaged per-case Dice, with a t-distribution 95% CI of "
          "the mean difference. ΔDice = Dice(A) − Dice(B), in Dice points (%). Holm correction is applied "
          "across the 3 comparisons within each dataset.", "",
          "| Dataset | Method A | Method B | ΔDice | 95% CI | Raw p | Holm p |", "|---|---|---|---:|---|---:|---:|"]
    for _, r in pairs.iterrows():
        L.append(f"| {r.dataset} | {r.method_a} | {r.method_b} | {r.delta_dice_pct:+.2f} | "
                 f"[{r.ci95_low_pct:+.2f}, {r.ci95_high_pct:+.2f}] | {_p(r.p_raw)} | {_p(r.p_holm)} |")
    L += ["", "Supplementary sensitivity analysis (specified in advance, **not used for the main conclusions**): "
          "two-sided Wilcoxon signed-rank test on the same paired differences, with its own Holm correction "
          "within each dataset.", "",
          "| Dataset | Method A | Method B | Wilcoxon raw p | Wilcoxon Holm p |", "|---|---|---|---:|---:|"]
    for _, r in pairs.iterrows():
        L.append(f"| {r.dataset} | {r.method_a} | {r.method_b} | {_p(r.wilcoxon_p_raw)} | {_p(r.wilcoxon_p_holm)} |")
    L += ["", "*[placeholder: interpretation of the pairwise comparisons]*", ""]

    L += ["## 3. Comparison with the LViT paper (Table II)", "",
          "Reference values only, used to check the pipeline; they play no role in any selection. The LViT "
          "values are single numbers from the paper, and ours are the mean over 3 seeds. Dice / mIoU in %.", "",
          "| Dataset | Method | Ours Dice | LViT Dice | Δ | Ours mIoU | LViT mIoU | Δ |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        key = ({n: m for m, n in METHODS}[r.method], r.dataset)
        ld, lm = LVIT_TABLE2[key]
        L.append(f"| {r.dataset} | {r.method} | {r.dice_mean_pct:.2f} | {ld:.2f} | {r.dice_mean_pct - ld:+.2f} | "
                 f"{r.miou_mean_pct:.2f} | {lm:.2f} | {r.miou_mean_pct - lm:+.2f} |")
    L += ["", "*[placeholder: comparison with LViT Table II, separating confirmed facts from unconfirmed "
          "possible causes, as in PHASE0_MODEL_SETUP.md section 7; including whether the U-Net MosMedData+ "
          "Dice is stable across seeds]*", ""]

    L += ["## 4. Figures", ""]
    for ds in DATASETS:
        L += [f"*[placeholder: one or two sentences stating what the {ds} Dice comparison shows]*", "",
              f"![{ds} Dice with 95% CI]({rel(fig_paths[f'ci_{ds}'])})", ""]
    L += ["*[placeholder: one or two sentences on seed stability]*", "",
          f"![Three-seed stability]({rel(fig_paths['stability'])})", ""]
    for ds in DATASETS:
        sel = ", ".join(f"q{int(p['quantile'] * 100)} = `{p['sample_id']}` ({p['score'] * 100:.1f})" for p in picks[ds])
        L += [f"*[placeholder: one or two sentences pointing to specific cases in the {ds} case figure]*", "",
              f"![{ds} cases]({rel(fig_paths[f'cases_{ds}'])})", "",
              f"Case selection rule: each test case's Dice is averaged over the 3 seeds and the 3 methods, and "
              f"the cases at the 10/30/50/70/90% quantiles of this score are shown ({sel}). **The drawn "
              f"predictions are those of seed {FIGURE_SEED}, and the Dice / HD95 printed in every panel are "
              f"also the seed-{FIGURE_SEED} values for that case.** The ground truth is the yellow dashed "
              f"contour.", ""]
    L += ["## 5. Conclusions", "", "*[placeholder: conclusions, written after reviewing all results]*", "",
          "## 6. Changes to the analysis after it was fixed", "",
          "None. (Any bug fix to `src/analyze_phase0.py` after the fixing commit is listed here: what changed, "
          "why, and the results before and after.)", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs"))
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="'test' for the final analysis; 'val' only for testing the program")
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "reports", "figures"))
    ap.add_argument("--report", default=os.path.join(REPO, "reports", "PHASE0_RESULTS.md"))
    ap.add_argument("--quiet", action="store_true", help="do not print the tables (used when testing)")
    args = ap.parse_args()
    for d in (args.results_dir, args.figures_dir, os.path.dirname(args.report)):
        os.makedirs(d, exist_ok=True)

    per_case = load_per_case(args.runs_root, args.split)                                   # Task 0.5 outputs
    table = main_table(per_case)                                                            # Task 0.6
    pairs = pairwise_table(per_case)                                                        # Task 0.7
    table.to_csv(os.path.join(args.results_dir, "phase0_main_results.csv"), index=False, float_format="%.4f")
    pairs.to_csv(os.path.join(args.results_dir, "phase0_pairwise_comparisons.csv"), index=False, float_format="%.6g")

    fig_paths = {"stability": os.path.join(args.figures_dir, "phase0_seed_stability.png")}
    picks = {}
    for ds in DATASETS:
        tag = ds.replace("+", "plus")
        fig_paths[f"ci_{ds}"] = os.path.join(args.figures_dir, f"phase0_dice_ci_{tag}.png")
        fig_paths[f"cases_{ds}"] = os.path.join(args.figures_dir, f"phase0_cases_{tag}.png")
        fig_dice_ci(table, ds, fig_paths[f"ci_{ds}"])
        picks[ds] = select_cases(per_case, ds)
        fig_cases(per_case, args.runs_root, args.split, ds, picks[ds], fig_paths[f"cases_{ds}"])
    fig_seed_stability(table, fig_paths["stability"])
    with open(os.path.join(args.results_dir, "phase0_case_selection.json"), "w") as f:
        json.dump(picks, f, indent=2)
    write_report(table, pairs, picks, fig_paths, args.report, args.split)
    if not args.quiet:
        print(table.to_string(index=False))
        print(pairs.to_string(index=False))
    print("written:", os.path.relpath(args.report, REPO))


if __name__ == "__main__":
    main()
