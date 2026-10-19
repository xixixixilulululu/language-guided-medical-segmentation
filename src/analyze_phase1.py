"""Phase 1, Tasks 1.8-1.11: six-method tables, 15 pairwise comparisons per dataset, figures, PHASE1_RESULTS.md.

THIS FILE IS THE FIXED ANALYSIS PLAN (methods text: reports/templates/PHASE1_RESULTS_methods.md). It is
fixed, committed and pushed, and its sha256 is recorded (src/phase1/analysis_lock.py), before the Phase 1
test split is linked into the data view. With --split test it refuses to run (exit code 5) unless that
program lock is open; the check happens before anything else is read. After all runs finish it is run
once, unchanged. Any later bug fix is a separate commit, and PHASE1_RESULTS.md must then record what
changed, why, and the results before and after.

Input: for every method x dataset x seed, the per-case metrics of the common evaluator (src/evaluate.py):
    <runs-root>/<method>/<dataset>/seed<k>/eval_<split>/per_case.csv   (sample_id, dice, miou, hd95, assd)
    <runs-root>/<method>/<dataset>/seed<k>/eval_<split>/masks/<id>.png (binarised predictions, case figure)
The Phase 0 runs (unet, transunet, nnunet) are read unchanged. Every run must hold exactly the split ids.

Statistics (identical functions to Phase 0, imported from src/analyze_phase0.py):
  main tables: mean ± sample SD of the 3 per-seed means; Dice 95% CI = case bootstrap (10,000, rng 0) of the
  seed-averaged per-case Dice.
  pairwise: 15 pairs per dataset (A earlier in METHODS); per-case differences of seed-averaged values;
  Dice: mean, t-CI, paired t-test, Holm across the 15 pairs, alpha = 0.05 (fixed); Wilcoxon + own Holm
  (supplementary); mIoU / HD95 / ASSD: mean difference, t-CI and case-bootstrap CI (10,000, rng 0), no p.

Outputs: results/phase1_main_results.csv, results/final_six_method_results.csv,
results/final_six_method_pairwise.csv, results/phase1_case_selection.json, reports/figures/phase1_*.png,
reports/PHASE1_RESULTS.md (tables generated; interpretation = placeholders, written by hand later).

Testing (never on the test split): src/test_analyze_phase1.py (synthetic runs on the validation split).

Usage (after all runs, once the program lock is open):
    python src/analyze_phase1.py
"""
import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase1 import analysis_lock  # noqa: E402

# (key, name, text at train, text at test); fixed order = order of the pairwise differences
METHODS = [("unet", "U-Net", False, False), ("transunet", "TransUNet", False, False),
           ("nnunet", "nnU-Net", False, False), ("lvit", "LViT-T", True, True),
           ("reclmis", "RecLMIS", True, True), ("ddcmd", "DD-CMD", True, True)]
LANGUAGE = ["lvit", "reclmis", "ddcmd"]
NAME = {m: n for m, n, _, _ in METHODS}
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]
PAIRS = list(itertools.combinations([m for m, *_ in METHODS], 2))          # 15 pairs, A before B
METRICS = ["dice", "miou", "hd95", "assd"]
ALPHA = 0.05                 # two-sided, after Holm across the 15 pairs of a dataset (fixed in advance)
N_BOOT, BOOT_SEED = 10_000, 0
CASE_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
FIGURE_SEED = 1001
FIGURE_METHODS = ["unet", "nnunet", "lvit", "reclmis", "ddcmd"]             # Phase 1 plan, optional figure
COLORS = {"unet": "#2a78d6", "transunet": "#eb6834", "nnunet": "#1baf7a",    # Phase 0 colours kept
          "lvit": "#8e44c9", "reclmis": "#d6407f", "ddcmd": "#8a7a12"}
MARKERS = {"unet": "o", "transunet": "s", "nnunet": "D", "lvit": "^", "reclmis": "v", "ddcmd": "P"}
SEED_MARKERS = {1001: "o", 1002: "s", 1003: "^"}
INK, INK2, GRID, GT_COLOR = "#0b0b0b", "#52514e", "#d9d8d4", "#f2c200"
# Published reference values (each paper's own metric definitions; orientation only, never used for selection)
REFERENCE = {("lvit", "QaTa-COV19"): "83.66 / 75.11 (LViT README / paper Table II)",
             ("lvit", "MosMedData+"): "74.57 / 61.33 (LViT README / paper Table II)",
             ("ddcmd", "QaTa-COV19"): "91.46 / 84.26 (DD-CMD paper, arXiv 2608.11335; pixel-pooled Dice)",
             ("ddcmd", "MosMedData+"): "81.95 / 69.42 (DD-CMD paper, arXiv 2608.11335; pixel-pooled Dice)",
             ("reclmis", "QaTa-COV19"): "[placeholder: RecLMIS paper value]",
             ("reclmis", "MosMedData+"): "[placeholder: RecLMIS paper value]"}


def stats_module():
    """Phase 0 statistics (bootstrap_mean_ci, paired_t, wilcoxon_p, holm), imported unchanged."""
    import analyze_phase0
    return analyze_phase0


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def run_dir(root, method, dataset, seed):
    return os.path.join(root, method, dataset, f"seed{seed}")


def load_per_case(root, split):
    """{(method, dataset, seed): DataFrame indexed by sample_id (split order)}; ids are checked."""
    from lgms import data
    out = {}
    for ds in DATASETS:
        ids = data.split_ids(ds, split)
        for m, *_ in METHODS:
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
    return pd.concat([per_case[(method, dataset, s)][metric] for s in SEEDS], axis=1).mean(axis=1)


def scale_unit(metric):
    return (100.0, "pct") if metric in ("dice", "miou") else (1.0, "px")


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------
def main_table(per_case, methods=None):
    """One row per dataset x method (Task 1.8 / 1.9)."""
    S = stats_module()
    methods = methods or [m for m, *_ in METHODS]
    info = {m: (n, tr, te) for m, n, tr, te in METHODS}
    rows = []
    for ds in DATASETS:
        for m in methods:
            name, text_train, text_test = info[m]
            row = {"dataset": ds, "method": name, "text_at_train": "Yes" if text_train else "No",
                   "text_at_test": "Yes" if text_test else "No", "n_seeds": len(SEEDS),
                   "n_cases": len(per_case[(m, ds, SEEDS[0])])}
            seed_means = {met: [per_case[(m, ds, s)][met].mean() for s in SEEDS] for met in METRICS}
            for met in METRICS:
                sc, unit = scale_unit(met)
                v = np.array(seed_means[met]) * sc
                row[f"{met}_mean_{unit}"], row[f"{met}_sd_{unit}"] = v.mean(), v.std(ddof=1)
                if met == "dice":
                    row["dice_ci95_low_pct"], row["dice_ci95_high_pct"] = S.bootstrap_mean_ci(
                        seed_averaged(per_case, m, ds) * 100.0, n_boot=N_BOOT, seed=BOOT_SEED)
            for s, v in zip(SEEDS, seed_means["dice"]):
                row[f"dice_seed{s}_pct"] = v * 100.0
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(per_case):
    """Task 1.10: 15 pairs per dataset; Dice tests with Holm across the 15; other metrics: differences + CIs."""
    S = stats_module()
    rows = []
    for ds in DATASETS:
        block = []
        for a, b in PAIRS:
            r = {"dataset": ds, "method_a": NAME[a], "method_b": NAME[b]}
            for met in METRICS:
                sc, unit = scale_unit(met)
                xa, xb = seed_averaged(per_case, a, ds, met) * sc, seed_averaged(per_case, b, ds, met) * sc
                assert (xa.index == xb.index).all()                       # aligned test cases
                t = S.paired_t(xa.values, xb.values)
                r[f"delta_{met}_{unit}"] = t["delta"]
                r[f"{met}_ci95_low_{unit}"], r[f"{met}_ci95_high_{unit}"] = t["ci_low"], t["ci_high"]
                if met == "dice":
                    w_stat, w_p = S.wilcoxon_p(xa.values, xb.values)
                    r.update(n_cases=t["n"], t_statistic=t["t"], p_raw=t["p"], wilcoxon_statistic=w_stat,
                             wilcoxon_p_raw=w_p)
                else:
                    lo, hi = S.bootstrap_mean_ci(xa.values - xb.values, n_boot=N_BOOT, seed=BOOT_SEED)
                    r[f"{met}_boot_ci95_low_{unit}"], r[f"{met}_boot_ci95_high_{unit}"] = lo, hi
            block.append(r)
        for col, src in [("p_holm", "p_raw"), ("wilcoxon_p_holm", "wilcoxon_p_raw")]:
            for r, v in zip(block, S.holm([r[src] for r in block])):
                r[col] = float(v)
        for r in block:
            r["significant_holm_alpha_0.05"] = bool(r["p_holm"] < ALPHA)
        rows += block
    cols = ["dataset", "method_a", "method_b", "n_cases", "delta_dice_pct", "dice_ci95_low_pct", "dice_ci95_high_pct",
            "t_statistic", "p_raw", "p_holm", "significant_holm_alpha_0.05", "wilcoxon_statistic", "wilcoxon_p_raw",
            "wilcoxon_p_holm"]
    for met in ["miou", "hd95", "assd"]:
        _, unit = scale_unit(met)
        cols += [f"delta_{met}_{unit}", f"{met}_ci95_low_{unit}", f"{met}_ci95_high_{unit}",
                 f"{met}_boot_ci95_low_{unit}", f"{met}_boot_ci95_high_{unit}"]
    return pd.DataFrame(rows)[cols]


def select_cases(per_case, dataset):
    """Rule-based case selection: quantiles of the Dice averaged over the 3 seeds and the 6 methods."""
    score = pd.concat([seed_averaged(per_case, m, dataset) for m, *_ in METHODS], axis=1).mean(axis=1)
    ordered = score.sort_index().sort_values(kind="stable")
    n = len(ordered)
    return [{"quantile": q, "sample_id": ordered.index[int(round(q * (n - 1)))],
             "score": float(ordered.iloc[int(round(q * (n - 1)))])} for q in CASE_QUANTILES]


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_dice_ci(table, dataset, path, split):
    """Required figures 1/2: mean Dice with 95% CI, 6 methods, one dataset."""
    import matplotlib.pyplot as plt
    sub = table[table.dataset == dataset].set_index("method")
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    for y, (m, name, *_) in enumerate(METHODS):
        r = sub.loc[name]
        ax.plot([r.dice_ci95_low_pct, r.dice_ci95_high_pct], [y, y], color=COLORS[m], linewidth=2, solid_capstyle="round")
        ax.plot(r.dice_mean_pct, y, MARKERS[m], color=COLORS[m], markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(1.02, y, f"{r.dice_mean_pct:.2f}  [{r.dice_ci95_low_pct:.2f}, {r.dice_ci95_high_pct:.2f}]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK)
    ax.axhline(2.5, color=GRID, linewidth=1, linestyle="--")                 # image-only | language-guided
    ax.set_yticks(range(len(METHODS)), [n for _, n, *_ in METHODS], color=INK, fontsize=10)
    ax.invert_yaxis()
    lo, hi = sub.dice_ci95_low_pct.min(), sub.dice_ci95_high_pct.max()
    pad = max(0.5, (hi - lo) * 0.15)
    ax.set_xlim(max(0.0, lo - pad), min(100.0, hi + pad))
    ax.set_xlabel(f"{split.capitalize()} Dice (%), mean of 3 seeds, 95% CI (case bootstrap, 10,000 resamples)",
                  color=INK2, fontsize=8.5)
    ax.set_title(f"{dataset}: Dice with 95% CI (above the line: image-only; below: language-guided)", loc="left",
                 color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_seed_stability(table, path, split):
    """Required figure 3: per-seed Dice, mean ± SD; language-guided methods, image-only ones for reference."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), dpi=200)
    for ax, ds in zip(axes, DATASETS):
        sub = table[table.dataset == ds].set_index("method")
        for x, (m, name, *_) in enumerate(METHODS):
            r = sub.loc[name]
            ax.errorbar(x, r.dice_mean_pct, yerr=r.dice_sd_pct, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=6, zorder=1)
            ax.plot([x - 0.22, x + 0.22], [r.dice_mean_pct] * 2, color=INK, linewidth=1.5, zorder=2)
            for k, s in enumerate(SEEDS):
                ax.plot(x + (k - 1) * 0.12, r[f"dice_seed{s}_pct"], SEED_MARKERS[s], color=COLORS[m], markersize=7,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.axvline(2.5, color=GRID, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(METHODS)), [n for _, n, *_ in METHODS], color=INK, fontsize=8.5)
        ax.set_ylabel(f"{split.capitalize()} Dice (%)", color=INK2, fontsize=9)
        ax.set_title(f"{ds} (left: image-only, for reference; right: language-guided)", loc="left", color=INK, fontsize=9.5)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        _style(ax, grid_axis="y")
    handles = [plt.Line2D([], [], marker=SEED_MARKERS[s], linestyle="none", color=INK2, markersize=7,
                          label=f"seed {s}") for s in SEEDS]
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.5, label="mean ± SD (3 seeds)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Three-seed stability of the Dice", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_cases(per_case, root, split, dataset, picks, path):
    """Real cases (rule-based): image | ground truth | seed-1001 prediction of the figure methods."""
    import matplotlib.pyplot as plt
    from lgms import data
    cols = ["Image", "Ground truth"] + [NAME[m] for m in FIGURE_METHODS]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.15 * len(picks)), dpi=150)
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
                m = FIGURE_METHODS[c - 2]
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
                wrapped = "\n".join(sid[k:k + 24] for k in range(0, len(sid), 24))
                ax.set_ylabel(f"q{int(pk['quantile'] * 100)}\n{wrapped}", fontsize=6.5, color=INK2)
    fig.suptitle(f"{dataset}: cases at the 10/30/50/70/90% quantiles of the mean Dice (3 seeds x 6 methods); "
                 f"predictions and labels: seed {FIGURE_SEED}", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# PHASE1_RESULTS.md (tables generated; interpretation = placeholders written by hand later)
# ------------------------------------------------------------------------------------------------
def _p(p):
    return "<1e-300" if p == 0 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def git_commit():
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "src/analyze_phase1.py"],
                                        text=True).strip()
        return h + (" (src/analyze_phase1.py has uncommitted changes!)" if dirty else "")
    except Exception:  # noqa: BLE001 - report generation must not fail on git problems
        return "unknown"


QUESTIONS = ["Which method performs best on QaTa-COV19?", "Which method performs best on MosMedData+?",
             "How strong are the image-only baselines?",
             "Do language-guided methods consistently outperform image-only methods?",
             "Does the recent DD-CMD model change the conclusion compared with LViT-T and RecLMIS?",
             "Which methods are most stable across seeds?", "Do boundary metrics agree with Dice?",
             "Which differences remain significant after Holm correction?"]


def write_report(table, pairs, picks, fig_paths, path, split, lock_info):
    rel = lambda p: os.path.relpath(p, os.path.dirname(path))  # noqa: E731
    L = ["# Phase 1 — Results (Tasks 1.7–1.11)", "",
         f"Generated by `src/analyze_phase1.py` (repository commit `{git_commit()}`; analysis fixed in commit "
         f"`{lock_info.get('commit', 'n/a (not a test-split run)')}`) on "
         f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}, split: `{split}`. The tables and figures are "
         "generated; text marked *[placeholder]* is written by hand after reviewing the results.", ""]
    with open(os.path.join(REPO, "reports", "templates", "PHASE1_RESULTS_methods.md")) as f:
        L += [f.read().rstrip(), ""]
    L += ["## 1. Six-method results (Tasks 1.8 and 1.9)", "",
          f"{split.capitalize()} set, 224×224 common evaluation. Mean ± sample SD over the 3 seeds of the per-seed "
          "means. Dice and mIoU in %, HD95 and ASSD in pixels. The Dice 95% CI is the case bootstrap of the "
          "seed-averaged per-case Dice.", "",
          "| Dataset | Method | Text at Train | Text at Test | Dice Mean ± SD | Dice 95% CI | mIoU | HD95 | ASSD |",
          "|---|---|---|---|---|---|---:|---:|---:|"]
    for _, r in table.iterrows():
        L.append(f"| {r.dataset} | {r.method} | {r.text_at_train} | {r.text_at_test} | {r.dice_mean_pct:.2f} ± "
                 f"{r.dice_sd_pct:.2f} | [{r.dice_ci95_low_pct:.2f}, {r.dice_ci95_high_pct:.2f}] | "
                 f"{r.miou_mean_pct:.2f} ± {r.miou_sd_pct:.2f} | {r.hd95_mean_px:.2f} ± {r.hd95_sd_px:.2f} | "
                 f"{r.assd_mean_px:.2f} ± {r.assd_sd_px:.2f} |")
    L += ["", "Per-seed Dice (%):", "", "| Dataset | Method | " + " | ".join(f"seed {s}" for s in SEEDS) + " |",
          "|---|---|" + "---:|" * len(SEEDS)]
    for _, r in table.iterrows():
        L.append(f"| {r.dataset} | {r.method} | " + " | ".join(f"{r[f'dice_seed{s}_pct']:.2f}" for s in SEEDS) + " |")
    L += ["", "## 2. Pairwise comparisons, Dice (Task 1.10)", "",
          f"ΔDice = Dice(A) − Dice(B) in Dice points; paired t-test on the seed-averaged per-case Dice; Holm across "
          f"the 15 comparisons of each dataset; significant = Holm p < {ALPHA} (fixed in advance).", "",
          "| Dataset | Method A | Method B | ΔDice | 95% CI | Raw p | Holm p | Significant |",
          "|---|---|---|---:|---|---:|---:|---|"]
    for _, r in pairs.iterrows():
        L.append(f"| {r.dataset} | {r.method_a} | {r.method_b} | {r.delta_dice_pct:+.2f} | [{r.dice_ci95_low_pct:+.2f}, "
                 f"{r.dice_ci95_high_pct:+.2f}] | {_p(r.p_raw)} | {_p(r.p_holm)} | "
                 f"{'yes' if r['significant_holm_alpha_0.05'] else 'no'} |")
    L += ["", "Supplementary sensitivity analysis (specified in advance, **not used for the main conclusions**): "
          "Wilcoxon signed-rank test with its own Holm correction across the 15 comparisons.", "",
          "| Dataset | Method A | Method B | Wilcoxon raw p | Wilcoxon Holm p |", "|---|---|---|---:|---:|"]
    for _, r in pairs.iterrows():
        L.append(f"| {r.dataset} | {r.method_a} | {r.method_b} | {_p(r.wilcoxon_p_raw)} | {_p(r.wilcoxon_p_holm)} |")
    L += ["", "## 3. Pairwise differences, mIoU / HD95 / ASSD", "",
          "Mean paired difference A − B with the t-based 95% CI and the case-bootstrap 95% CI (no p-values). mIoU in "
          "points (%), HD95 / ASSD in pixels (lower is better).", "",
          "| Dataset | Method A | Method B | ΔmIoU [t-CI] | ΔHD95 [t-CI] [bootstrap CI] | ΔASSD [t-CI] [bootstrap CI] |",
          "|---|---|---|---|---|---|"]
    for _, r in pairs.iterrows():
        L.append(f"| {r.dataset} | {r.method_a} | {r.method_b} | {r.delta_miou_pct:+.2f} [{r.miou_ci95_low_pct:+.2f}, "
                 f"{r.miou_ci95_high_pct:+.2f}] | {r.delta_hd95_px:+.2f} [{r.hd95_ci95_low_px:+.2f}, {r.hd95_ci95_high_px:+.2f}] "
                 f"[{r.hd95_boot_ci95_low_px:+.2f}, {r.hd95_boot_ci95_high_px:+.2f}] | {r.delta_assd_px:+.2f} "
                 f"[{r.assd_ci95_low_px:+.2f}, {r.assd_ci95_high_px:+.2f}] [{r.assd_boot_ci95_low_px:+.2f}, "
                 f"{r.assd_boot_ci95_high_px:+.2f}] |")
    L += ["", "## 4. Published reference values (Dice / mIoU, %)", "",
          "Each paper's own setting and metric definition; for orientation only, never used for any selection.", "",
          "| Dataset | Method | Ours Dice | Ours mIoU | Published |", "|---|---|---:|---:|---|"]
    for _, r in table.iterrows():
        key = ({n: m for m, n, *_ in METHODS}[r.method], r.dataset)
        if key[0] in LANGUAGE:
            L.append(f"| {r.dataset} | {r.method} | {r.dice_mean_pct:.2f} | {r.miou_mean_pct:.2f} | {REFERENCE[key]} |")
    L += ["", "## 5. Figures", ""]
    for ds in DATASETS:
        L += [f"*[placeholder: one or two sentences stating what the {ds} Dice comparison shows]*", "",
              f"![{ds} Dice with 95% CI]({rel(fig_paths[f'ci_{ds}'])})", ""]
    L += ["*[placeholder: one or two sentences on seed stability]*", "",
          f"![Three-seed stability]({rel(fig_paths['stability'])})", ""]
    for ds in DATASETS:
        sel = ", ".join(f"q{int(p['quantile'] * 100)} = `{p['sample_id']}` ({p['score'] * 100:.1f})" for p in picks[ds])
        L += [f"*[placeholder: one or two sentences pointing to specific cases in the {ds} case figure]*", "",
              f"![{ds} cases]({rel(fig_paths[f'cases_{ds}'])})", "",
              f"Case selection rule: Dice averaged over the 3 seeds and the 6 methods; the cases at the 10/30/50/70/90% "
              f"quantiles are shown ({sel}). **The drawn predictions and the Dice / HD95 in every panel are those of seed "
              f"{FIGURE_SEED}.** The ground truth is the yellow dashed contour.", ""]
    L += ["## 6. Answers to the Phase 1 questions (Task 1.11)", "",
          "No claim that the text itself causes a difference is made from comparing different architectures; "
          "this is tested in Phase 2.", ""]
    for k, q in enumerate(QUESTIONS, 1):
        L += [f"{k}. **{q}** *[placeholder]*", ""]
    L += ["## 7. Changes to the analysis after it was fixed", "",
          "None. (Any bug fix to `src/analyze_phase1.py` after the fixing commit is listed here: what changed, why, "
          "and the results before and after.)", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.join(REPO, "runs"))
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="'test' for the final analysis (needs the open program lock); 'val' only for testing")
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "reports", "figures"))
    ap.add_argument("--report", default=os.path.join(REPO, "reports", "PHASE1_RESULTS.md"))
    ap.add_argument("--lock-repo", default=REPO, help="repository whose program lock is checked (tests only)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    # Program lock FIRST: nothing of the test split is read unless the analysis is fixed on GitHub
    lock_info = analysis_lock.require_open(args.lock_repo) if args.split == "test" else {}
    for d in (args.results_dir, args.figures_dir, os.path.dirname(args.report)):
        os.makedirs(d, exist_ok=True)

    per_case = load_per_case(args.runs_root, args.split)
    table = main_table(per_case)                                               # Task 1.9 (6 methods)
    table[table.method.isin([NAME[m] for m in LANGUAGE])].drop(columns=["text_at_train", "text_at_test"]).to_csv(
        os.path.join(args.results_dir, "phase1_main_results.csv"), index=False, float_format="%.4f")   # Task 1.8
    table.to_csv(os.path.join(args.results_dir, "final_six_method_results.csv"), index=False, float_format="%.4f")
    pairs = pairwise_table(per_case)                                           # Task 1.10
    pairs.to_csv(os.path.join(args.results_dir, "final_six_method_pairwise.csv"), index=False, float_format="%.6g")

    fig_paths = {"stability": os.path.join(args.figures_dir, "phase1_seed_stability.png")}
    picks = {}
    for ds in DATASETS:
        tag = ds.replace("+", "plus")
        fig_paths[f"ci_{ds}"] = os.path.join(args.figures_dir, f"phase1_dice_ci_{tag}.png")
        fig_paths[f"cases_{ds}"] = os.path.join(args.figures_dir, f"phase1_cases_{tag}.png")
        fig_dice_ci(table, ds, fig_paths[f"ci_{ds}"], args.split)
        picks[ds] = select_cases(per_case, ds)
        fig_cases(per_case, args.runs_root, args.split, ds, picks[ds], fig_paths[f"cases_{ds}"])
    fig_seed_stability(table, fig_paths["stability"], args.split)
    with open(os.path.join(args.results_dir, "phase1_case_selection.json"), "w") as f:
        json.dump(picks, f, indent=2)
    write_report(table, pairs, picks, fig_paths, args.report, args.split, lock_info)
    if not args.quiet:
        print(table.to_string(index=False))
        print(pairs.to_string(index=False))
    print("written:", os.path.relpath(args.report, REPO))


if __name__ == "__main__":
    main()
