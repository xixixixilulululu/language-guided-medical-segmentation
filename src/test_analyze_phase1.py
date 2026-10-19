"""Tests for src/analyze_phase1.py -- synthetic data and the VALIDATION split only (never the test split).

Part A: structure and statistics
  * 15 pairs per dataset, A before B in the fixed method order; Holm over 15 == statsmodels.
Part B: synthetic end-to-end run
  * 6 methods x 2 datasets x 3 seeds of synthetic per-case metrics on the validation ids, with known
    method effects (DD-CMD best, TransUNet worst) and 2% empty predictions; binarised masks of seed 1001
    for the figure methods (shifted ground truth). Written in the real run layout.
  * analyze_phase1.py --split val into a scratch folder.
  * Every number of the three CSV files is recomputed with separate plain code (pandas / scipy /
    statsmodels / an own bootstrap loop) and compared; output structure, figures and report are checked.
Part C: input errors are refused (missing run, wrong sample ids).
The program lock (test split) is tested in src/phase1/test_analysis_lock.py.

Usage:  python src/test_analyze_phase1.py --scratch <dir>      (exit status 1 if any check fails)
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import analyze_phase1 as A  # noqa: E402
from lgms import data, metrics  # noqa: E402

FAILED = []
EFFECT = {"unet": -0.04, "transunet": -0.07, "nnunet": -0.01, "lvit": 0.00, "reclmis": 0.005, "ddcmd": 0.03}


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def part_a():
    check("15 pairs", len(A.PAIRS) == 15 and len(set(A.PAIRS)) == 15)
    order = [m for m, *_ in A.METHODS]
    check("pair direction: A earlier than B", all(order.index(a) < order.index(b) for a, b in A.PAIRS))
    from statsmodels.stats.multitest import multipletests
    S = A.stats_module()
    rng = np.random.default_rng(3)
    check("Holm over 15 == statsmodels (300 random sets)",
          all(np.allclose(S.holm(p), multipletests(p, method="holm")[1]) for p in rng.uniform(0, 0.1, size=(300, 15))))


def make_runs(root):
    for ds in A.DATASETS:
        ids = data.split_ids(ds, "val")
        base = np.random.default_rng(len(ds)).beta(6, 2, len(ids))
        for mi, (m, *_) in enumerate(A.METHODS):
            for s in A.SEEDS:
                rng = np.random.default_rng([mi, s, len(ds)])
                dice = np.clip(base + EFFECT[m] + rng.normal(0, 0.03, len(ids)), 0.0, 1.0)
                hd95 = np.abs(25 * (1 - dice) + rng.normal(0, 2, len(ids)))
                empty = rng.random(len(ids)) < 0.02
                dice[empty] = 0.0
                hd95[empty] = metrics.MAX_DIST
                df = pd.DataFrame({"sample_id": ids, "dice": dice, "miou": dice / (2 - dice), "hd95": hd95,
                                   "assd": np.where(empty, metrics.MAX_DIST, hd95 / 3)})
                out = os.path.join(A.run_dir(root, m, ds, s), "eval_val")
                os.makedirs(os.path.join(out, "masks"), exist_ok=True)
                df.to_csv(os.path.join(out, "per_case.csv"), index=False)
                if s == A.FIGURE_SEED and m in A.FIGURE_METHODS:
                    for k, i in enumerate(ids):
                        g = data.read_mask(ds, "val", i)
                        p = np.zeros_like(g) if empty[k] else np.roll(g, mi + 1, axis=1)
                        cv2.imwrite(os.path.join(out, "masks", f"{i}.png"), (p > 0).astype(np.uint8) * 255)
        print(f"  synthetic runs written: {ds}")


def boot_ci(x, n_boot=10_000, seed=0):
    """Own percentile bootstrap of the mean (same resampling scheme: default_rng(seed), blocks of 1000)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = []
    for start in range(0, n_boot, 1000):
        idx = rng.integers(0, len(x), size=(min(1000, n_boot - start), len(x)))
        means.append(x[idx].mean(axis=1))
    means = np.concatenate(means)
    return np.quantile(means, 0.025), np.quantile(means, 0.975)


def independent_check(root, out):
    from statsmodels.stats.multitest import multipletests
    res = os.path.join(out, "results")
    six = pd.read_csv(os.path.join(res, "final_six_method_results.csv"))
    lg = pd.read_csv(os.path.join(res, "phase1_main_results.csv"))
    pw = pd.read_csv(os.path.join(res, "final_six_method_pairwise.csv"))
    check("six-method table: 6 methods x 2 datasets", len(six) == 12)
    check("phase1 table: only the 3 language-guided methods", len(lg) == 6 and set(lg.method) == {"LViT-T", "RecLMIS", "DD-CMD"})
    check("text columns: image-only No/No, language-guided Yes/Yes",
          all((r.text_at_train == r.text_at_test == ("Yes" if r.method in ("LViT-T", "RecLMIS", "DD-CMD") else "No"))
              for r in six.itertuples()))
    check("pairwise: 15 rows per dataset", all((pw.dataset == ds).sum() == 15 for ds in A.DATASETS))
    worst = 0.0
    for ds in A.DATASETS:
        case = {}
        for m, name, *_ in A.METHODS:
            dfs = [pd.read_csv(os.path.join(A.run_dir(root, m, ds, s), "eval_val", "per_case.csv"), dtype={"sample_id": str})
                   .set_index("sample_id").sort_index() for s in A.SEEDS]
            row = six[(six.dataset == ds) & (six.method == name)].iloc[0]
            for met, sc, unit in [("dice", 100, "pct"), ("miou", 100, "pct"), ("hd95", 1, "px"), ("assd", 1, "px")]:
                v = np.array([d[met].mean() for d in dfs]) * sc
                worst = max(worst, abs(row[f"{met}_mean_{unit}"] - v.mean()), abs(row[f"{met}_sd_{unit}"] - np.std(v, ddof=1)))
                case[(m, met)] = sum(d[met] for d in dfs) / 3 * sc
            lo, hi = boot_ci(case[(m, "dice")].reindex(data.split_ids(ds, "val")).values)
            worst = max(worst, abs(row.dice_ci95_low_pct - lo), abs(row.dice_ci95_high_pct - hi))
        sub = pw[pw.dataset == ds].reset_index(drop=True)
        praw, wraw = [], []
        for k, (a, b) in enumerate(A.PAIRS):
            r = sub.iloc[k]
            ok = r.method_a == A.NAME[a] and r.method_b == A.NAME[b]
            for met, unit in [("dice", "pct"), ("miou", "pct"), ("hd95", "px"), ("assd", "px")]:
                xa = case[(a, met)].reindex(data.split_ids(ds, "val")).values
                xb = case[(b, met)].reindex(data.split_ids(ds, "val")).values
                d = xa - xb
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                ok &= np.isclose(r[f"delta_{met}_{unit}"], d.mean(), atol=1e-4)
                ok &= np.isclose(r[f"{met}_ci95_low_{unit}"], d.mean() - half, atol=1e-4)
                ok &= np.isclose(r[f"{met}_ci95_high_{unit}"], d.mean() + half, atol=1e-4)
                if met == "dice":
                    praw.append(stats.ttest_rel(xa, xb).pvalue)
                    wraw.append(stats.wilcoxon(xa, xb).pvalue)
                    ok &= np.isclose(r.p_raw, praw[-1], rtol=1e-4) and np.isclose(r.wilcoxon_p_raw, wraw[-1], rtol=1e-4)
                else:
                    lo, hi = boot_ci(d)
                    ok &= np.isclose(r[f"{met}_boot_ci95_low_{unit}"], lo, atol=1e-4)
                    ok &= np.isclose(r[f"{met}_boot_ci95_high_{unit}"], hi, atol=1e-4)
            check(f"{ds} {A.NAME[a]} vs {A.NAME[b]}: all differences, CIs and p-values", bool(ok))
        holm, wholm = multipletests(praw, method="holm")[1], multipletests(wraw, method="holm")[1]
        check(f"{ds}: Holm across 15 (t-test and Wilcoxon) == statsmodels",
              np.allclose(sub.p_holm, holm, rtol=1e-4) and np.allclose(sub.wilcoxon_p_holm, wholm, rtol=1e-4))
        check(f"{ds}: significance flag == (Holm p < 0.05)",
              (sub["significant_holm_alpha_0.05"] == (holm < 0.05)).all())
        order = six[six.dataset == ds].sort_values("dice_mean_pct").method.tolist()
        check(f"{ds}: synthetic effect recovered (TransUNet worst, DD-CMD best)",
              order[0] == "TransUNet" and order[-1] == "DD-CMD", str(order))
    check("main tables match independent recomputation (max abs diff < 1e-3)", worst < 1e-3, f"{worst:.2e}")
    for f in ["phase1_dice_ci_QaTa-COV19.png", "phase1_dice_ci_MosMedDataplus.png", "phase1_seed_stability.png",
              "phase1_cases_QaTa-COV19.png", "phase1_cases_MosMedDataplus.png"]:
        check(f"figure written: {f}", os.path.getsize(os.path.join(out, "figures", f)) > 10_000)
    rep = open(os.path.join(out, "PHASE1_RESULTS.md")).read()
    check("report: methods text, tables, 15-pair tables, questions, placeholders",
          all(s in rep for s in ["Methods (fixed analysis plan)", "| Dataset | Method | Text at Train", "Holm p",
                                 "Wilcoxon", "ΔHD95", "Which differences remain significant", "[placeholder"])
          and rep.count("| QaTa-COV19 | U-Net | TransUNet |") == 3)


def part_c(root, tmp):
    """Missing run and wrong sample ids must be refused."""
    bad = os.path.join(tmp, "bad_runs")
    shutil.copytree(root, bad)
    f = os.path.join(A.run_dir(bad, "ddcmd", "MosMedData+", 1003), "eval_val", "per_case.csv")
    df = pd.read_csv(f, dtype={"sample_id": str})
    df.iloc[:-1].to_csv(f, index=False)                       # one case missing
    try:
        A.load_per_case(bad, "val"); check("wrong sample ids refused", False)
    except ValueError:
        check("wrong sample ids refused", True)
    os.remove(f)
    try:
        A.load_per_case(bad, "val"); check("missing run refused", False)
    except FileNotFoundError:
        check("missing run refused", True)
    shutil.rmtree(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="folder for the synthetic runs and outputs (not in the repo)")
    args = ap.parse_args()
    print("== Part A")
    part_a()
    print("== Part B: synthetic end-to-end run on the validation split")
    root, out = os.path.join(args.scratch, "runs"), os.path.join(args.scratch, "out")
    make_runs(root)
    subprocess.run([sys.executable, os.path.join(REPO, "src", "analyze_phase1.py"), "--runs-root", root, "--split", "val",
                    "--results-dir", os.path.join(out, "results"), "--figures-dir", os.path.join(out, "figures"),
                    "--report", os.path.join(out, "PHASE1_RESULTS.md"), "--quiet"], check=True)
    independent_check(root, out)
    print("== Part C: input errors")
    part_c(root, args.scratch)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
